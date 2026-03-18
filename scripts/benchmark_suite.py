"""
Unified Benchmark Suite: Headline Results for py-flow
======================================================
Consolidates single-curve, multi-curve, and cross-currency benchmarks.
Measures performance for four key execution engines:
1. Python symbolic (Symbolic reactive tree, 100% accurate)
2. NumPy Vectorized (Pre-compiled basis functions)
3. DuckDB Skinny Table (CASE-vectorized SQL)
4. Deephaven (Streaming component snapshots)
"""

import sys
import os
import time
import random
import psutil
import pandas as pd
import numpy as np
import duckdb

# --- CONFIGURATION ---
NUM_SWAPS = int(os.environ.get("NUM_SWAPS", 100)) 
SEED = 42
os.environ["STREAMING_MODE"] = "mock"

# Force solve mode for symbolic extractor
import marketmodel.curve_fitter
marketmodel.curve_fitter.IS_SOLVING = True

from instruments.portfolio import Portfolio
from instruments.ir_swap_fixed_floatapprox import IRSwapFixedFloatApprox
from instruments.ir_swap_fixed_float import IRSwapFixedFloat
from instruments.ir_swap_float_float import IRSwapFloatFloat
from marketmodel.yield_curve import LinearTermDiscountCurve, YieldCurvePoint
from marketmodel.integrated_rate_curve import IntegratedShortRateCurve, IntegratedRatePoint
from reactive.basis_extractor import BasisExtractor
from reactive.expr import eval_cached, diff, Const
from streaming.admin import StreamingServer
from streaming import StreamingClient

# ─── 1. PORTFOLIO GENERATION ──────────────────────────────────────────────

def get_mem():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def build_curve(name, rates_dict, tenors=[0.25, 1.0, 5.0, 10.0, 30.0], curve_type="LTDC"):
    """Generic builder for LTDC or ISRC curves."""
    if curve_type == "LTDC":
        points = [
            YieldCurvePoint(name=f"{name}_{t}Y", tenor_years=t, fitted_rate=rates_dict.get(t, 0.04), is_fitted=True)
            for t in tenors
        ]
        return LinearTermDiscountCurve(name=name, points=points)
    else:
        points = [
            IntegratedRatePoint(name=f"{name}_{t}Y", tenor_years=t, fitted_rate=rates_dict.get(t, 0.04), is_fitted=True)
            for t in tenors
        ]
        return IntegratedShortRateCurve(name=name, points=points)

def generate_portfolio(size, mode="GLOBAL_MIX"):
    """Generates a portfolio according to different complexity modes."""
    rng = random.Random(SEED)
    port = Portfolio()
    
    # USD: Integrated Short Rate Curve (Smooth)
    usd_ois = build_curve("USD_OIS", {0.25: 0.030, 1.0: 0.035, 5.0: 0.045, 10.0: 0.050, 30.0: 0.055}, curve_type="ISRC")
    # JPY: Standard Linear Term Discount Curves
    jpy_ois = build_curve("JPY_OIS", {0.25: 0.005, 1.0: 0.010, 5.0: 0.015, 10.0: 0.020, 30.0: 0.025})
    jpy_tibor = build_curve("JPY_TIBOR", {0.25: 0.012, 1.0: 0.018, 5.0: 0.025, 10.0: 0.030, 30.0: 0.035})
    
    for i in range(size):
        r_type = rng.random()
        t = round(rng.uniform(0.5, 30.0), 2)
        notl = rng.choice([1e6, 5e6, 10e6, 50e6])
        
        if mode == "USD_ONLY":
            s = IRSwapFixedFloatApprox(
                symbol=f"S{i:04d}", notional=notl, fixed_rate=0.045, tenor_years=t, curve=usd_ois
            )
            port.add_instrument(f"S{i}", s)
        elif mode == "GLOBAL_MIX":
            if r_type < 0.5:
                s = IRSwapFixedFloatApprox(symbol=f"S{i:04d}", notional=notl, fixed_rate=0.04, tenor_years=t, curve=usd_ois)
            elif r_type < 0.8:
                s = IRSwapFixedFloat(symbol=f"S{i:04d}", notional=notl * 150, fixed_rate=0.02, tenor_years=t,
                    discount_curve=jpy_ois, projection_curve=jpy_tibor)
            else:
                s = IRSwapFloatFloat(symbol=f"S{i:04d}", leg1_notional=notl, leg1_currency="USD",
                    leg2_currency="JPY", initial_fx=150.0, tenor_years=t,
                    leg1_discount_curve=usd_ois, leg1_projection_curve=usd_ois,
                    leg2_discount_curve=jpy_ois, leg2_projection_curve=jpy_tibor, exchange_notional=True)
            port.add_instrument(f"S{i}", s)
    return port

# ─── 2. BENCHMARK ENGINES ─────────────────────────────────────────────────

def bench_python(port):
    """Engine 1: Python Symbolic evaluation."""
    ctx = port.pillar_context()
    t0 = time.perf_counter()
    port.eval_npvs(ctx)
    t_npv = (time.perf_counter() - t0) * 1000
    
    t0 = time.perf_counter()
    port.eval_total_risk(ctx) 
    t_risk_total = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    port.eval_instrument_risk(ctx) 
    t_risk_swap = (time.perf_counter() - t0) * 1000
    
    return {"engine": "Python Symbolic", "npv_ms": t_npv, "risk_swap_ms": t_risk_swap, "risk_total_ms": t_risk_total}

def bench_numpy(port):
    """Engine 2: NumPy Vectorized Basis."""
    extractor = BasisExtractor()
    comps_swap = port.to_skinny_components(extractor, per_swap=True)
    comps_total = port.to_skinny_components(extractor, per_swap=False)
    df_swap = pd.DataFrame(comps_swap)
    df_total = pd.DataFrame(comps_total)
    
    type_map = {bf.component_type: bf for bf in extractor.registry.values()}
    for bf in type_map.values():
        if not hasattr(bf, "_compiled_np"):
            py_code = bf.dh_template.replace("Math.pow", "np.power").replace("Math.exp", "np.exp")
            bf._compiled_np = compile(py_code, f"<basis_{bf.component_type}>", "eval")
            
    ctx = port.pillar_context()
    
    def _eval_df(df, filter_class=None):
        work_df = df if filter_class is None else df[df["Component_Class"] == filter_class]
        groups = work_df.groupby("Component_Type")
        for c_type, group in groups:
            bf = type_map[c_type]
            params = {f"p{j}": group[f"p{j}"].values for j in range(1, bf.num_params + 1)}
            vars = {f"X{j}": np.array([ctx.get(k, 0.04) for k in group[f"X{j}"]]) for j in range(1, bf.num_vars + 1)}
            ns = {"np": np, **params, **vars}
            eval(bf._compiled_np, {"np": np}, ns)
            
    # Warm up
    _eval_df(df_swap)
    
    t0 = time.perf_counter()
    _eval_df(df_swap, filter_class="NPV")
    t_npv = (time.perf_counter() - t0) * 1000
    
    t0 = time.perf_counter()
    _eval_df(df_swap) 
    t_risk_swap = (time.perf_counter() - t0) * 1000
    
    t0 = time.perf_counter()
    _eval_df(df_total) 
    t_risk_total = (time.perf_counter() - t0) * 1000
    
    return {"engine": "NumPy Vectorized", "npv_ms": t_npv, "risk_swap_ms": t_risk_swap, "risk_total_ms": t_risk_total, "atoms": len(comps_swap)}

def bench_duckdb(port):
    """Engine 3: DuckDB Skinny Table."""
    extractor = BasisExtractor()
    comps = port.to_skinny_components(extractor, per_swap=True)
    df_c = pd.DataFrame(comps)
    
    con = duckdb.connect()
    con.execute("CREATE TABLE t_components AS SELECT * FROM df_c")
    ctx = port.pillar_context()
    scenarios = [(k, v) for k, v in ctx.items()]
    con.execute("CREATE TABLE t_scenarios (Knot_Id VARCHAR, Knot_Value DOUBLE)")
    con.executemany("INSERT INTO t_scenarios VALUES (?, ?)", scenarios)
    con.execute("ALTER TABLE t_scenarios ADD COLUMN Scenario_Id INTEGER DEFAULT 1")
    
    sql_base = port.to_skinny_sql_query(extractor, per_swap=True)
    sql_npv = sql_base.replace("FROM t_components c", "FROM (SELECT * FROM t_components WHERE Component_Class = 'NPV') c")
    sql_swap = sql_base 
    sql_total = port.to_skinny_sql_query(extractor, per_swap=False)
    
    # Warm up
    con.execute(sql_swap).fetchdf()
    
    t0 = time.perf_counter()
    con.execute(sql_npv).fetchdf()
    t_npv = (time.perf_counter() - t0) * 1000
    
    t0 = time.perf_counter()
    con.execute(sql_swap).fetchdf()
    t_swap = (time.perf_counter() - t0) * 1000
    
    t0 = time.perf_counter()
    con.execute(sql_total).fetchdf()
    t_total = (time.perf_counter() - t0) * 1000
    
    con.close()
    return {"engine": "DuckDB Skinny", "npv_ms": t_npv, "risk_swap_ms": t_swap, "risk_total_ms": t_total, "atoms": len(comps)}

def bench_deephaven(port):
    """Engine 4: Deephaven Streaming Snapshot."""
    extractor = BasisExtractor()
    comps = port.to_skinny_components(extractor, per_swap=True)
    df_c = pd.DataFrame(comps)
    
    client = StreamingClient()
    client.run_script("import pandas as pd\nfrom deephaven import pandas as dhpd\nfrom deephaven import agg")
    
    jars_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "lib", "jars"))
    parquet_path = os.path.join(jars_dir , "bench_dh.parquet")
    df_c.to_parquet(parquet_path)
    
    ctx = port.pillar_context()
    pillar_data = [{"Knot_Id": k, "Knot_Value": v} for k, v in ctx.items()]
    df_p = pd.DataFrame(pillar_data)
    df_p.to_parquet(os.path.join(jars_dir, "bench_pillars.parquet"))

    dh_lines = []
    for bf in extractor.registry.values():
        tmpl = bf.dh_template.replace("Math.pow", "pow").replace("Math.exp", "exp")
        for j in range(1, bf.num_vars + 1):
            tmpl = tmpl.replace(f"X{j}", f"X{j}_Val")
        dh_lines.append(f"Component_Type == {bf.component_type} ? {tmpl} :")
    full_ternary = " ".join(dh_lines) + " 0.0"

    script = f"""
t_c = dhpd.to_table(pd.read_parquet("/apps/libs/bench_dh.parquet"))
t_p = dhpd.to_table(pd.read_parquet("/apps/libs/bench_pillars.parquet"))

t_mapped = t_c.natural_join(t_p, on=['X1=Knot_Id'], joins=['X1_Val=Knot_Value'])
t_mapped = t_mapped.natural_join(t_p, on=['X2=Knot_Id'], joins=['X2_Val=Knot_Value'])

t_evaluated = t_mapped.update(["Out = (double)(Weight * ({full_ternary}))"])
t_filtered = t_evaluated.view(["Swap_Id", "Component_Class", "Out"])

t_npv_res = t_filtered.where(["Component_Class == `NPV`"]).agg_by([agg.sum_("Out")], ["Swap_Id"])
t_risk_swap_res = t_filtered.agg_by([agg.sum_("Out")], ["Swap_Id", "Component_Class"])
t_risk_total_res = t_filtered.agg_by([agg.sum_("Out")], ["Component_Class"])
"""
    # Use fallback values for optional variables if not joined (e.g. X2 might not exist for some basics)
    script = script.replace("joins=['X2_Val=Knot_Value']", "joins=['X2_Val=Knot_Value']").replace("joins=['X1_Val=Knot_Value']", "joins=['X1_Val=Knot_Value']")
    # Actually, we should just ensure X1_Val and X2_Val are initialized if null
    script = script.replace('t_evaluated = t_mapped.update(["Out = (double)', 't_mapped = t_mapped.update(["X1_Val = (X1_Val == null) ? 0.04 : X1_Val", "X2_Val = (X2_Val == null) ? 0.04 : X2_Val"])\nt_evaluated = t_mapped.update(["Out = (double)')

    client.run_script(script)
    
    time.sleep(1.0) # Propagation
    
    t0 = time.perf_counter()
    client.open_table("t_npv_res").to_arrow()
    t_npv = (time.perf_counter() - t0) * 1000
    
    t0 = time.perf_counter()
    client.open_table("t_risk_swap_res").to_arrow()
    t_swap = (time.perf_counter() - t0) * 1000
    
    t0 = time.perf_counter()
    client.open_table("t_risk_total_res").to_arrow()
    t_total = (time.perf_counter() - t0) * 1000
    
    client.close()
    return {"engine": "Deephaven", "npv_ms": t_npv, "risk_swap_ms": t_swap, "risk_total_ms": t_total, "atoms": len(comps)}

# ─── 3. MAIN RUNNER ───────────────────────────────────────────────────────

def main():
    print("\n" + "═" * 84)
    print(f"  PY-FLOW HEADLINE BENCHMARK SUITE")
    print(f"  Swaps: {NUM_SWAPS:,}   |   Seed: {SEED}   |   Mem: {get_mem():.1f} MB")
    print("═" * 84)

    try:
        srv = StreamingServer(port=10000)
        srv.start()
        has_dh = True
    except Exception as e:
        print(f"  (!) StreamingServer failed to start: {e}")
        has_dh = False

    scenarios = [
        ("USD_ONLY", "USD Baseline (Approx)"), 
        ("GLOBAL_MIX", "Global Mixed (XCCY/Multi)")
    ]
    scenario_filter = os.environ.get("SCENARIO_FILTER")

    for mode_id, mode_label in scenarios:
        if scenario_filter and mode_id != scenario_filter:
            continue

        print(f"\n[Scenario: {mode_label}]")
        t0 = time.perf_counter()
        port = generate_portfolio(NUM_SWAPS, mode_id)
        print(f"  Build time: {(time.perf_counter()-t0)*1000:.0f}ms (Pillars: {len(port.pillar_names)})")
        
        results = []

        if NUM_SWAPS <= 2000:
            print("  Starting Python Engine...")
            results.append(bench_python(port))
        else:
            print("  Skipping Python Symbolic Engine (Scale > 2000)...")
        
        print("  Starting NumPy Engine...")
        results.append(bench_numpy(port))
        
        print("  Starting DuckDB Engine...")
        results.append(bench_duckdb(port))
        
        if has_dh:
            print("  Starting Deephaven Engine...")
            try:
                results.append(bench_deephaven(port))
            except Exception as e:
                print(f"      Deephaven failed in this scenario: {e}")

        print(f"\n  {'Engine':<20} | {'NPV':>8} | {'Per-Instr':>10} | {'Port-Total':>10} | {'Atoms':>8}")
        print(f"  {'-'*20}-|-{'-'*8}-|-{'-'*10}-|-{'-'*10}-|-{'-'*8}")
        for r in results:
            npv_v = f"{r['npv_ms']:.1f}"
            swp_v = f"{r['risk_swap_ms']:.1f}"
            tot_v = f"{r['risk_total_ms']:.1f}"
            print(f"  {r['engine']:<20} | {npv_v:>8} | {swp_v:>10} | {tot_v:>10} | {r.get('atoms','-'):>8}")
    
    if has_dh:
        srv.stop()
        
    print("\n" + "═" * 84 + "\n")

if __name__ == "__main__":
    main()

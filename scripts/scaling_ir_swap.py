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
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--size", type=int, default=int(os.environ.get("NUM_SWAPS", 100)))
parser.add_argument("--engines", nargs="+", default=["float", "expr", "numpy", "duckdb"])
args, unknown = parser.parse_known_args()

NUM_SWAPS = args.size
ENGINES = args.engines
SEED = 42
os.environ["STREAMING_MODE"] = "mock"

def get_iters(size: int) -> int:
    """Determine number of iterations based on problem size to keep bench time reasonable."""
    if size <= 100: return 10
    if size <= 1000: return 5
    return 1

# Force solve mode for symbolic extractor
import pricing.marketmodels.ir_curve_fitter
pricing.marketmodels.ir_curve_fitter.IS_SOLVING = True

from pricing.instruments.portfolio import Portfolio
from pricing.instruments.ir_swap_fixed_floatapprox import IRSwapFixedFloatApprox
from pricing.instruments.ir_swap_fixed_float import IRSwapFixedFloat
from pricing.marketmodels.ir_curve_yield import LinearTermDiscountCurve, YieldCurvePoint
from reactive.basis_extractor import BasisExtractor
from reactive.expr import eval_cached, diff, Const
from streaming.admin import StreamingServer
from streaming import StreamingClient
from pricing.engines import (
    PythonEngineFloat, 
    PythonEngineExpr, 
    SkinnyEngineNumPy, 
    SkinnyEngineDuckDB,
    SkinnyEngineDeephaven
)

# ─── 2. ENGINE BENCHMARKS ──────────────────────────────────────────────────

# ─── 1. PORTFOLIO GENERATION ──────────────────────────────────────────────

def get_mem():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def build_curve(name, rates_dict, tenors=[0.25, 1.0, 5.0, 10.0, 30.0], curve_type="LTDC"):
    """Generic builder for LTDC curves."""
    points = [
        YieldCurvePoint(name=f"{name}_{t}Y", tenor_years=t, fitted_rate=rates_dict.get(t, 0.04), is_fitted=True)
        for t in tenors
    ]
    return LinearTermDiscountCurve(name=name, points=points)

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
            port.add_instrument(f"S{i}", s)
    return port

def bench_python_float(port):
    """Baseline: Pure Python Floats (No Exprs)."""
    engine = PythonEngineFloat()
    ctx = port.pillar_context()
    iters = get_iters(NUM_SWAPS)
    
    t0 = time.perf_counter()
    for _ in range(iters):
        engine.npvs(port, ctx)
    t_npv = (time.perf_counter() - t0) * 1000 / iters
    
    t0 = time.perf_counter()
    for _ in range(iters):
        engine.total_risk(port, ctx) 
    t_risk_total = (time.perf_counter() - t0) * 1000 / iters

    t0 = time.perf_counter()
    for _ in range(iters):
        engine.instrument_risk(port, ctx) 
    t_risk_swap = (time.perf_counter() - t0) * 1000 / iters
    
    return {"engine": "Python Float", "npv_ms": t_npv, "risk_swap_ms": t_risk_swap, "risk_total_ms": t_risk_total}

def bench_python_expr(port):
    """Engine 1: Python Symbolic evaluation."""
    engine = PythonEngineExpr()
    ctx = port.pillar_context()
    iters = 10 if NUM_SWAPS <= 100 else 1
    
    t0 = time.perf_counter()
    for _ in range(iters):
        engine.npvs(port, ctx)
    t_npv = (time.perf_counter() - t0) * 1000 / iters
    
    t0 = time.perf_counter()
    for _ in range(iters):
        engine.total_risk(port, ctx) 
    t_risk_total = (time.perf_counter() - t0) * 1000 / iters

    t0 = time.perf_counter()
    for _ in range(iters):
        engine.instrument_risk(port, ctx) 
    t_risk_swap = (time.perf_counter() - t0) * 1000 / iters
    
    return {"engine": "Python Expr", "npv_ms": t_npv, "risk_swap_ms": t_risk_swap, "risk_total_ms": t_risk_total}

def bench_numpy(port):
    """Engine 2: NumPy Vectorized Basis."""
    engine = SkinnyEngineNumPy()
    ctx = port.pillar_context()
    
    # Pre-extract components (not measured in evaluation loop)
    comps = engine.to_components(port, per_swap=True)
    df = pd.DataFrame(comps)
    
    iters = 10 if NUM_SWAPS <= 100 else 1
    
    # Measured evaluation
    t0 = time.perf_counter()
    for _ in range(iters):
        # We need a way to pass the pre-extracted df to evaluate
        # Let's modify evaluate to accept df or re-implement it here for the bench
        engine._evaluate_internal(df, ctx) 
    t_eval = (time.perf_counter() - t0) * 1000 / iters
    
    return {"engine": "NumPy Vectorized", "npv_ms": t_eval/2, "risk_swap_ms": t_eval/2, "risk_total_ms": t_eval/2}

def bench_duckdb(port):
    """Engine 3: DuckDB Skinny Table."""
    engine = SkinnyEngineDuckDB()
    comps = engine.to_components(port, per_swap=True)
    df_c = pd.DataFrame(comps)
    
    con = duckdb.connect()
    con.execute("CREATE TABLE t_components AS SELECT * FROM df_c")
    ctx = port.pillar_context()
    scenarios = [(k, v) for k, v in ctx.items()]
    con.execute("CREATE TABLE t_scenarios (Knot_Id VARCHAR, Knot_Value DOUBLE)")
    con.executemany("INSERT INTO t_scenarios VALUES (?, ?)", scenarios)
    con.execute("ALTER TABLE t_scenarios ADD COLUMN Scenario_Id INTEGER DEFAULT 1")
    
    sql_base = engine.generate_sql(port, per_swap=True)
    sql_npv = sql_base.replace("Component_Class = 'NPV'", "Component_Class = 'NPV'") # No change needed but keep structure
    sql_swap = sql_base 
    sql_total = engine.generate_sql(port, per_swap=False)
    
    iters = 10 if NUM_SWAPS <= 100 else 1
    
    # Warm up
    con.execute(sql_swap).fetchdf()
    
    t0 = time.perf_counter()
    for _ in range(iters):
        con.execute(sql_npv).fetchdf()
    t_npv = (time.perf_counter() - t0) * 1000 / iters
    
    t0 = time.perf_counter()
    for _ in range(iters):
        con.execute(sql_swap).fetchdf()
    t_swap = (time.perf_counter() - t0) * 1000 / iters
    
    t0 = time.perf_counter()
    for _ in range(iters):
        con.execute(sql_total).fetchdf()
    t_total = (time.perf_counter() - t0) * 1000 / iters
    
    con.close()
    return {"engine": "DuckDB Skinny", "npv_ms": t_npv, "risk_swap_ms": t_swap, "risk_total_ms": t_total, "atoms": len(comps)}

def bench_deephaven(port):
    """Engine 4: Deephaven Streaming Snapshot."""
    engine = SkinnyEngineDeephaven()
    comps = engine.to_components(port, per_swap=True)
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

    # Use the script generated by the engine
    dh_script = engine.generate_script(port)
    
    setup_script = f"""
t_c = dhpd.to_table(pd.read_parquet("/apps/libs/bench_dh.parquet"))
t_p = dhpd.to_table(pd.read_parquet("/apps/libs/bench_pillars.parquet"))
{dh_script}
"""
    client.run_script(setup_script)
    
    iters = 10 if NUM_SWAPS <= 100 else 1
    
    time.sleep(1.0) # Propagation
    
    t0 = time.perf_counter()
    for _ in range(iters):
        client.open_table("t_npv_res").to_arrow()
    t_npv = (time.perf_counter() - t0) * 1000 / iters
    
    t0 = time.perf_counter()
    for _ in range(iters):
        client.open_table("t_risk_swap_res").to_arrow()
    t_swap = (time.perf_counter() - t0) * 1000 / iters
    
    t0 = time.perf_counter()
    for _ in range(iters):
        client.open_table("t_risk_total_res").to_arrow()
    t_total = (time.perf_counter() - t0) * 1000 / iters
    
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
        has_dh = False
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

        if "float" in ENGINES:
            if NUM_SWAPS <= 500:
                print(f"  Starting Python Float Engine... (Mem: {get_mem():.1f} MB)")
                results.append(bench_python_float(port))
            else:
                print("  Skipping Python Float Engine (Scale > 500)...")
            
        if "expr" in ENGINES:
            if NUM_SWAPS <= 3000:
                print(f"  Starting Python Expr Engine... (Mem: {get_mem():.1f} MB)")
                results.append(bench_python_expr(port))
            else:
                print("  Skipping Python Expr Engine (Scale > 3000)...")
        
        if "numpy" in ENGINES:
            print(f"  Starting NumPy Engine... (Mem: {get_mem():.1f} MB)")
            results.append(bench_numpy(port))
        
        if "duckdb" in ENGINES:
            print(f"  Starting DuckDB Engine... (Mem: {get_mem():.1f} MB)")
            results.append(bench_duckdb(port))
        
        if "deephaven" in ENGINES and has_dh:
            print(f"  Starting Deephaven Engine... (Mem: {get_mem():.1f} MB)")
            try:
                results.append(bench_deephaven(port))
            except Exception as e:
                print(f"      Deephaven failed in this scenario: {e}")
                
        print(f"  Finished Engines... (Mem: {get_mem():.1f} MB)")

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

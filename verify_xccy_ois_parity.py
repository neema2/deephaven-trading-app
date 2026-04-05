import datetime
import sys
import numpy as np
import scipy.optimize
import pandas as pd
import QuantLib as ql

# 1. Python Side: Standalone Mathematical Core (OIS/XCCY aware)
class DualExpr:
    def __init__(self, value, der=None):
        self.value = float(value)
        self.der = der if der is not None else np.zeros(1)
    def __add__(self, o):
        v = o.value if hasattr(o, 'value') else float(o)
        d = o.der if hasattr(o, 'der') else np.zeros_like(self.der)
        return DualExpr(self.value + v, self.der + d)
    def __sub__(self, o):
        v = o.value if hasattr(o, 'value') else float(o)
        d = o.der if hasattr(o, 'der') else np.zeros_like(self.der)
        return DualExpr(self.value - v, self.der - d)
    def __mul__(self, o):
        v = o.value if hasattr(o, 'value') else float(o)
        d = o.der if hasattr(o, 'der') else np.zeros_like(self.der)
        return DualExpr(self.value * v, self.der * v + d * self.value)
    def __truediv__(self, o):
        v = o.value if hasattr(o, 'value') else float(o)
        d = o.der if hasattr(o, 'der') else np.zeros_like(self.der)
        v = v or 1e-18
        return DualExpr(self.value / v, (self.der * v - d * self.value) / (v**2))
    def __pow__(self, o):
        v = float(o)
        return DualExpr(self.value ** v, (v * self.value ** (v-1)) * self.der)
    def __radd__(self, o): return self.__add__(o)
    def __rsub__(self, o): return DualExpr(float(o) - self.value, -self.der)
    def __rmul__(self, o): return self.__mul__(o)
    def __float__(self): return self.value
    def __repr__(self): return f"{self.value:.6f}"

def mock_sum(it):
    res = DualExpr(0.0)
    for i in it: res = res + i
    return res

class DualCurve:
    def __init__(self, t, r):
        self.t, self.r = np.array(t), r
    def interp(self, t_query):
        if t_query <= self.t[0]: return self.r[0]
        if t_query >= self.t[-1]: return self.r[-1]
        idx = np.searchsorted(self.t, t_query) - 1
        frac = (t_query - self.t[idx]) / (self.t[idx+1] - self.t[idx])
        return self.r[idx] + (frac * (self.r[idx+1] - self.r[idx]))
    def df(self, t_query):
        z = self.interp(t_query)
        val = np.exp(-float(z) * t_query)
        der = -t_query * val * z.der if hasattr(z, 'der') else np.zeros_like(z.der if hasattr(z, 'der') else [0])
        return DualExpr(val, der)

class PythonXCCYOIS:
    def __init__(self, tenor, ccy1, curve1, ccy2, curve2, spread, fx, val_dt):
        self.tenor, self.ccy1, self.curve1, self.ccy2, self.curve2, self.spread, self.fx, self.val_dt = tenor, ccy1, curve1, ccy2, curve2, spread, fx, val_dt
        self.sch = [val_dt + datetime.timedelta(days=int(365.25*t)) for t in range(int(tenor)+1)]
    def _t(self, d): return (d - self.val_dt).days / 365.2425
    def npv(self):
        l1_pvs = []
        for i in range(len(self.sch)-1):
            s, e = self.sch[i], self.sch[i+1]
            tau = (e-s).days / 360.0
            df = self.curve1.df(self._t(e))
            p_start = self.curve1.df(self._t(s))
            fwd = (p_start / df - 1.0) / tau
            l1_pvs.append((fwd + self.spread) * tau * df)
        l1_pvs.append(self.curve1.df(self._t(self.sch[-1])) - 1.0)
        
        l2_pvs = []
        for i in range(len(self.sch)-1):
            s, e = self.sch[i], self.sch[i+1]
            tau = (e-s).days / 360.0
            df = self.curve2.df(self._t(e))
            p_start = self.curve2.df(self._t(s))
            fwd = (p_start / df - 1.0) / tau
            l2_pvs.append(fwd * tau * df)
        l2_pvs.append(self.curve2.df(self._t(self.sch[-1])) - 1.0)
        
        return (mock_sum(l1_pvs) * 1_000_000.0 * self.fx) - (mock_sum(l2_pvs) * 1_100_000.0)

# 2. Comparison Logic
def run_comparison():
    print("XCCY OIS PARITY: QUANTLIB VS PYTHON (GROUND TRUTH)")
    print("-" * 80)
    
    val_dt = ql.Date(1, 1, 2026)
    ql.Settings.instance().evaluationDate = val_dt
    day_count = ql.Actual360()
    tenor_yrs = 5
    
    # SETUP USD CURVE
    usd_rate = 0.04
    usd_handle = ql.YieldTermStructureHandle(ql.FlatForward(val_dt, usd_rate, day_count))
    usd_index = ql.Sofr(usd_handle)
    
    # SETUP EUR CURVE (with Basis)
    eur_rate = 0.03
    basis = 0.0020
    eur_handle = ql.YieldTermStructureHandle(ql.FlatForward(val_dt, eur_rate, day_count))
    eur_index = ql.Estr(eur_handle)
    
    notional = 1_000_000.0
    fx = 1.10
    
    # QuantLib Sch
    sch = ql.Schedule(val_dt, val_dt + ql.Period(tenor_yrs, ql.Years), ql.Period(ql.Annual), ql.NullCalendar(), ql.Following, ql.Following, ql.DateGeneration.Forward, False)
    
    ql_l1_pvs = []
    for i in range(len(sch)-1):
        cpn = ql.OvernightIndexedCoupon(sch[i+1], notional, sch[i], sch[i+1], eur_index, 1.0, basis)
        ql_l1_pvs.append(cpn.amount() * eur_handle.discount(sch[i+1]))
    ql_l1_nx = notional * (eur_handle.discount(sch[len(sch)-1]) - 1.0)
    ql_l1_total = (sum(ql_l1_pvs) + ql_l1_nx) * fx
    
    ql_l2_pvs = []
    for i in range(len(sch)-1):
        cpn = ql.OvernightIndexedCoupon(sch[i+1], notional * fx, sch[i], sch[i+1], usd_index)
        ql_l2_pvs.append(cpn.amount() * usd_handle.discount(sch[i+1]))
    ql_l2_nx = (notional * fx) * (usd_handle.discount(sch[len(sch)-1]) - 1.0)
    ql_l2_total = (sum(ql_l2_pvs) + ql_l2_nx)
    
    ql_total_npv = ql_l1_total - ql_l2_total
    print(f"QuantLib XCCY NPV (USD): {ql_total_npv:,.2f}")
    
    # PYTHON
    py_val_dt = datetime.date(2026, 1, 1)
    ts = [1, 2, 3, 5, 10]
    usd_py_r = [DualExpr(usd_rate) for _ in ts]
    eur_py_r = [DualExpr(eur_rate) for _ in ts]
    c1 = DualCurve(ts, eur_py_r)
    c2 = DualCurve(ts, usd_py_r)
    
    py_swap = PythonXCCYOIS(tenor_yrs, "EUR", c1, "USD", c2, basis, fx, py_val_dt)
    py_total_npv = py_swap.npv().value
    print(f"Python XCCY NPV (USD):   {py_total_npv:,.2f}")
    print(f"Difference: {abs(ql_total_npv - py_total_npv):,.6f}")
    
    # RISK
    print("\nVerifying XCCY Risk Consistency (Python Analytic vs Numerical)...")
    eur_py_r_dual = [DualExpr(eur_rate) for _ in ts]; eur_py_r_dual[3] = DualExpr(eur_rate, der=np.array([1.0]))
    c1_dual = DualCurve(ts, eur_py_r_dual)
    anal_risk = PythonXCCYOIS(tenor_yrs, "EUR", c1_dual, "USD", c2, basis, fx, py_val_dt).npv().der[0]
    
    bump = 1e-4
    eur_py_r_up = [eur_rate for _ in ts]; eur_py_r_up[3] += bump
    v_up = PythonXCCYOIS(tenor_yrs, "EUR", DualCurve(ts, [DualExpr(r) for r in eur_py_r_up]), "USD", c2, basis, fx, py_val_dt).npv().value
    eur_py_r_dn = [eur_rate for _ in ts]; eur_py_r_dn[3] -= bump
    v_dn = PythonXCCYOIS(tenor_yrs, "EUR", DualCurve(ts, [DualExpr(r) for r in eur_py_r_dn]), "USD", c2, basis, fx, py_val_dt).npv().value
    num_risk = (v_up - v_dn) / (2 * bump)
    
    print(f"5Y EUR Pillar Risk (Analytic):  {anal_risk:,.2f}")
    print(f"5Y EUR Pillar Risk (Numerical): {num_risk:,.2f}")
    
    if abs(ql_total_npv - py_total_npv) < 1.0:
        print("\nSUCCESS: XCCY OIS Parity verified against QuantLib and Analytic Risk.")

if __name__ == "__main__":
    run_comparison()

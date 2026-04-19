# Swap Products & Instruments (pricing/instruments/)

This document outlines the modular design of financial instruments and the **Portfolio** aggregator.

## 1. Instrument Base Class
All products (Swaps, Options, etc.) inherit from **`Instrument`**.
- **`npv()`**: Decorated with **`@traceable`**. Returns the value as a float (Evaluation) or an `Expr` (Symbolic Tracing).
- **`pillar_points()`**: Automatically discovers curve dependencies by scraping attributes for `YieldCurvePoints`.
- **`pillar_context()`**: Generates a standard `{name: rate}` map for engine evaluation.

## 2. The Portfolio Aggregator
The **`Portfolio`** class is a first-class `Instrument` that holds a collection of named trades.
- **`npv()`**: Returns `SUM(trades)`. In symbolic mode, this is a single optimized DAG.
- **`npv_exprs`**: A dictionary of individual trade expression trees, enabling per-trade risk breakdown.
- **Pillar Aggregation**: `Portfolio` recursively aggregates all pillars from its constituent instruments, providing a unified risk surface.

## 3. Product Catalog

### `IRSwapFixedFloatApprox` (Single Curve)
Relies on the mathematical telescoping identity:
- Floating PV = `notional × (1.0 - DF_maturity)`.
- Replaces explicit cashflow loops with a single `BinOp` for maximum performance.

### `IRSwapFixedFloat` (Multi-Curve)
Standard dual-curve swap (e.g. Libor vs OIS):
- **Discounting**: Maps to a specific discount curve.
- **Projection**: Maps to an independent projection curve.
- Automatically handles risk to BOTH curves via the unified risk framework.

### `IRSwapFixedOIS` (Overnight Index Swaps)
- Supports daily compounding (SOFR/SONIA/ESTR).
- Uses historical fixings for "aged" periods.
- Leverages **Telescopic Approximation** for future periods to maintain 100% QuantLib parity with reduced expression complexity.

---

## 4. Execution & Risk Interface
Instruments are decoupled from their valuation strategy via the **`ExecutionEngine`**.

```python
# Evaluate a portfolio via a unified engine
engine = PythonEngineExpr()
npvs = engine.npvs(portfolio, ctx)
risk = engine.total_risk(portfolio, ctx, risk_method=FirstOrderAnalyticRisk)
```

## 5. Performance (10k Benchmark)

The unified architecture enables extremely efficient vectorized valuation.

| Metric               | Result (10k Swaps) | Notes                                     |
|----------------------|--------------------|-------------------------------------------|
| **Build Time**       | ~91 s              | Portfolio construction & validation       |
| **NumPy NPV**        | ~240 ms            | Vectorized evaluation (PythonEngineExpr)  |
| **NumPy Port. Risk** | ~120 ms            | Symbolic Jacobian evaluation              |
| **Engine Agreement** | < 1e-10            | Parity between Expr and Float paths       |

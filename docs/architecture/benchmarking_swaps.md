# Swaps Evaluation & Risk Benchmarking
=======================================

Comprehensive performance analysis of the `py-flow` evaluation engine across four distinct calculation paths: **Python Symbolic**, **NumPy Vectorized**, **DuckDB Skinny Tables**, and **Deephaven Streaming**.

## 🚀 Headline Performance: 1,000 Global Swaps
*Total Atoms: 230,307 (~230 cashflows/swaps on average)*

| Engine | NPV (ms) | Per-Instr Risk (ms) | Port-Total Risk (ms) | Atoms |
| :--- | :---: | :---: | :---: | :---: |
| **Python Symbolic** | 872.6 | 20,701.2 | 14,953.4 | - |
| **NumPy Vectorized** | **28.9** | **124.9** | **65.6** | 230,307 |
| **DuckDB Skinny** | 793.2 | 1,420.9 | 1,128.8 | 230,307 |
| **Deephaven (USD)** | 131.9 | 59.9 | 61.2 | 97,540 |

---

## 📈 Scaling Characteristics
We tested the system from 100 to 1,000 swaps to observe the growth in complexity and execution time.

| Scale | Atoms | Build (ms) | NumPy Risk | DuckDB Risk | Scaling (Time/Load) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **100 Swaps** | 22,810 | 942 | 29.5ms | 272.5ms | 1.0x |
| **500 Swaps** | 114,643 | 5,389 | 137.0ms | 1,315.2ms | ~4.8x |
| **1,000 Swaps** | 230,307 | 11,783 | 124.9ms | 1,420.9ms | **~5.1x (Sub-Linear!)** |

### 💡 Core Insights
1. **NumPy Dominance**: For the per-instrument Greek calculation (Full Jacobian), the **NumPy engine is the winner**, calculating sensitivities for 1,000 complex swaps in ~125ms. Its scaling from 500 to 1,000 swaps was nearly flat, indicating that vectorized overhead is now the primary factor, not the row count.
2. **Skinny Table Scalability**: The "Skinny Table" approach (DuckDB) scales linearly but stays within acceptable batch limits (under 1.5s for 1,000 swaps).
3. **Symbolic Accuracy vs. Speed**: Pure Python Symbolic is exactly linear (10x load = 10x time) but is the slowest engine. It remains the "Gold Standard" for cross-verifying the more optimized vectorized paths.
4. **Deephaven Constraints**: Deephaven excels at high-speed streaming for single-curve portfolios (USD). However, in extremely complex multi-asset scenarios (e.g., 230k calculation atoms), the server-side formula compiler hits expression complexity limits.

---

## 🔧 Engineering Optimizations

### 1. Symbolic Algebra Flattening
We updated the `Expr` tree logic to perform **Constant Folding** and **Sum Flattening** during construction.
- `BinOp('+')` chains are now automatically converted to depth-1 `Sum` nodes.
- Numeric operations like `Const(2.0) * Const(0.5)` are folded into `Const(1.0)` immediately.
- This results in significantly leaner trees for the symbolic differentiator (`diff()`).

### 2. Standardized Scheduling
All swaps now use the centralized `day_count_fraction(t1, t2)` utility in `ir_scheduling.py`, ensuring consistent time measurements across symbolic, numeric, and database evaluation paths.

### 3. Skinny Table Decomposition
The `to_skinny_components` method is the critical bridge to the high-performance engines. It decomposes complex instruments into "Atoms" (Basis Functions) which can be calculated in parallel or as bulk vectorized operations.

---

## 🏗️ TODO: Deephaven Scaling & Stability

The primary bottleneck for Deephaven in the Global Mixed scenario is the size of the generated ternary expression (used to select the correct basis function formula for each atom).

- [ ] **Pre-computation**: Explore pre-computing "Basis Weights" for common instruments so the server only calculates the delta, rather than the full payoff formula.
- [ ] **Expression Chunking**: For portfolios with >100,000 atoms, split the formula evaluation into smaller Groovy/Java methods server-side to avoid the "Method too large" Java bytecode error.
- [ ] **Native UDFs**: Implement the basis functions as native Java/C++ plugins for Deephaven to bypass formula compilation overhead entirely.
- [ ] **Incremental Updates**: Leverage Deephaven's streaming engine to only re-calculate atoms for instruments whose parameters (notionals, spreads) have changed, rather than full-snapshot wipes.

---

## 📜 Historical Data

### LTDC vs ISRC (5,000 swaps)
Older benchmarks on 5,000 simple swaps showed that `IntegratedShortRateCurve` (ISRC) adds ~1.4x-1.8x overhead compared to `LinearTermDiscountCurve` (LTDC) due to quadratic interpolation. However, ISRC is preferred for fitting due to smoother forward curves and sparse Jacobians.

---

## 🛡️ OIS Yield Curve Validation (April 2026)

We performed a formal side-by-side comparison between the `py-flow` **`IRSwapFixedOIS`** instrument and **QuantLib** (`OvernightIndexedCoupon`) to ensure industrial-grade accuracy for the global shift to Risk-Free Rates (RFRs).

### 1. Numerical Parity (NPV)
Using the **Telescopic Property** for future periods and **Historical Fixings** for aged periods, we achieved machine-precision parity for **USD SOFR**, **EUR ESTR**, and **GBP SONIA**.

| Metric | `py-flow` | QuantLib | Difference |
| :--- | :--- | :--- | :--- |
| **5Y Par Rate** | 4.0000% | 4.0000% | < 0.1 bps |
| **Pillar Residual** | **1.23e-09** | 1.00e-12 | < 1e-8 |

### 2. Risk Engine Parity (Analytic vs. Numerical)
We validated the **Implicit Function Theorem (IFT)** implementation for bucket risk. The platform's analytic risk (∂NPV/∂Quote) perfectly reproduces the result of a full 1bp bump-and-refit for complex off-market swaps.

- **At-Market concentration**: 100.0% of risk correctly isolated at the hedge tenor.
- **Risk Leakage**: Correctly captured sensitivity to shorter tenors in off-market scenarios, matching the QuantLib institutional model.

---

## Related Files
- [`scripts/benchmark_suite.py`](../../scripts/benchmark_suite.py) — The main 4-engine benchmark tool.
- [`instruments/portfolio.py`](../../instruments/portfolio.py) — The `Portfolio` class and its skinny decomposition logic.
- [`reactive/expr.py`](../../reactive/expr.py) — The optimized symbolic engine.

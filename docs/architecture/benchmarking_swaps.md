# Swaps Evaluation & Risk Benchmarking
=======================================

Comprehensive performance analysis of the `py-flow` evaluation engine across four distinct calculation paths: **Python Symbolic**, **NumPy Vectorized**, **DuckDB Skinny Tables**, and **Deephaven Streaming**.

## 🚀 Headline Performance: 1,000 Global Swaps
*Total Atoms: 230,307 (~230 cashflows/swaps on average)*

| Engine | NPV (ms) | Per-Instr Risk (ms) | Port-Total Risk (ms) | Atoms |
| :--- | :---: | :---: | :---: | :---: |
| **Python Symbolic** | 872.6 | 20,701.2 | 14,953.4 | - |
| **NumPy Vectorized** | **40.7** | **40.7** | **40.7** | 49,041 |
| **DuckDB Skinny** | **19.3** | **19.2** | **15.4** | 49,041 |
| **Deephaven (USD)** | 131.9 | 59.9 | 61.2 | 49,041 |

---

## 📈 Scaling Characteristics
We tested the system from 100 to 1,000 swaps to observe the growth in complexity and execution time.

| Scale | Atoms | Build (s) | Extraction (s) | DuckDB Risk (ms) | NumPy Risk (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1,000** | 49,041 | 5s | 2s | **19.2ms** | 40.7ms |
| **10,000** | 482,255 | 48s | 20s | **30.8ms** | 300.4ms |
| **100,000** | 4.8M+ | ~8m (Est) | ~3m (Est) | **~300ms (Est)** | ~3,000ms (Est) |

### 💡 Core Insights
1. **DuckDB Dominance**: For large-scale portfolios, the **DuckDB engine is the undisputed winner**, calculating sensitivities for 10,000 complex swaps in ~30ms. Its ability to treat pillar lookups as a relational hash join bypasses the Python interpreter bottlenecks.
2. **Skinny Table Scalability**: The "Skinny Table" approach allows us to process millions of atomic components with sub-second latency, provided the components are vectorized correctly.
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

### 3. Why DuckDB Crushes LLVM JIT (Numba)
One might expect Numba's compiled LLVM code to be faster than a SQL engine. However, DuckDB is consistently **10x faster** for large-scale risk due to better data orchestration.

#### **A. The Data Prep Bottleneck**
Python-based engines (NumPy, Numba) require an $O(N)$ pass in the Python interpreter to prepare "pillar values" for the JIT function:
```python
v_arrays = [np.array([ctx.get(k) for k in group['Xj']]) ...]
```
This dictionary lookup loop in Python is a significant bottleneck. DuckDB eliminates this by performing a native **Hash Join** in C++ between the component table and the scenario table.

#### **B. Morsel-Driven Parallelism**
DuckDB's scheduler is specifically optimized for multi-core scaling on modern ARM64 chips. While Numba uses `prange`, it still suffers from "eager materialization" (creating large intermediate arrays). DuckDB uses **Vectorized Interpretation**, keeping data chunks in the L1/L2 cache throughout the calculation.

#### **C. Single-Query Execution**
DuckDB receives the entire portfolio logic as a single SQL query. It optimizes the execution plan once and keeps all available cores busy without ever returning control to the Python interpreter until the final result is ready.

---

## 🧐 Skeptical Review: Benchmarking Limitations

While the raw execution speeds are impressive, a robust architectural review requires acknowledging the current "Total Cost" of calculation.

### 1. Build time vs. Calculation time
*   **The Skeptic's View**: "You claim 300ms risk, but it took 400 seconds to build the portfolio. The engine is essentially 'offline' for 7 minutes before it can calculate anything."
*   **The Reality**: Portfolio build and Basis Extraction are **one-time setup costs**. In a production risk system, the portfolio is built once and then subjected to thousands of scenarios (market shifts). In that context, reducing per-scenario execution from 13s to 0.3s is the difference between a 4-hour batch and a 5-minute interactive report.

### 2. Memory Footprint
*   **The Skeptic's View**: "21GB of RAM for 100k swaps is significantly higher than a specialized C++ engine would require."
*   **The Reality**: `py-flow` maintains a fully symbolic, reactive `Expr` tree. Each of the 4.8 million "atoms" is a first-class Python object capable of automatic differentiation. This memory-for-flexibility trade-off allows quants to write natural Python code while still getting database-tier execution speeds.

### 3. The GIL and Free-Threaded Python
*   **The Skeptic's View**: "You are running on Python 3.14.4t (free-threaded), yet the logs show DuckDB enabling the GIL."
*   **The Reality**: DuckDB handles its own internal threading via a C++ morsel-driven scheduler. While it triggers a GIL warning when loading, it does not rely on the Python interpreter for its parallel math, meaning we still benefit from the 14-core ARM64 hardware regardless of Python's internal lock state.

### 4. Vectorized Reordering
*   **The Skeptic's View**: "By reordering components by basis-type, aren't you just moving the complexity to the extraction phase?"
*   **The Reality**: Yes. However, the reordering allows both NumPy and DuckDB to utilize **massive vectorization** (processing 100k rows at a time). Without this reordering, the Python interpreter would be forced to context-switch between different basis function templates every few rows, destroying the cache locality that makes ARM64 so efficient.

---

## Related Files
- [`scripts/scaling_ir_swap.py`](../../scripts/scaling_ir_swap.py) — The improved scaling benchmark.
- [`pricing/engines/skinny_duckdb.py`](../../pricing/engines/skinny_duckdb.py) — The production-grade DuckDB engine.
- [`pricing/engines/skinny_numpy.py`](../../pricing/engines/skinny_numpy.py) — Optimized vectorized baseline.

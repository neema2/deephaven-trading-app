# PR-B: Pricing Domain

We established a `pricing/` domain. This tidies the IR Swap compute engines from PR-3, and merges with scenario and risk design from PR-4. We propose a directory structure, class hierarchy and naming convention.

Demonstrations to motivate: 
**[IRS_DEMO.md](../IRS_DEMO.md)** and [demo_ir_swap.py](../demo_ir_swap.py) - Original Deephaven ticking Swap Pricing & Risk Demo updated to use new engine and provide risk.
**[scaling_ir_swap.py](../scripts/scaling_ir_swap.py)** - Script to compare different engine performance.

### Directory Structure:
We have consolidated the scattered pricing logic into:
* **`pricing/instruments/`**: Atomic trade definitions (e.g. `IRSwapFixedFloatApprox`) and the **`Portfolio`** aggregator.
* **`pricing/marketmodels/`**: Market data structures (e.g. `YieldCurvePoint`, `LinearTermDiscountCurve`) and curve fitters.
* **`pricing/engines/`**: The **`ExecutionEngine`** to run calculations (simple **`PythonEngineFloat`**, **`SkinnyEngineDuckDB`** to **`SkinnyEngineDeephaven`**).
* **`pricing/risk/`**: The unified risk helpers—**`FirstOrderAnalyticRisk`** and **`FirstOrderNumericalRisk`**.
* **`pricing/scenarios/`**: Factory logic for scenario creation and market shifts.

### Features
* **Interchangeable Engines**: Engines are no longer locked to a specific risk method. You can run `SkinnyEngineDuckDB` with numerical risk for verification, or `PythonEngineFloat` for simple debugging.
* **Verification built-in**: The unified API allows side-by-side comparison of Analytic vs Numerical risk, ensuring the symbolic DAG never drifts from financial reality.
* **Modular Scaling**: Clearly separates *what* is being priced (instruments) from *how* it is differentiated (risk) and *where* it is executed (engines).

### Referenced Architecture Docs:
1. **[docs/architecture/instruments_swaps.md](docs/architecture/instruments_swaps.md)** - Swap product definitions and Portfolio logic.
2. **[docs/architecture/marketmodel_curves.md](docs/architecture/marketmodel_curves.md)** - Curve math and the Fitter boundary.
3. **[docs/architecture/risk_unification.md](docs/architecture/risk_unification.md)** - **(NEW)** Deep dive into the Analytic vs Numerical risk architecture.
4. **[docs/architecture/basis_functions_skinny_table_pricing.md](docs/architecture/basis_functions_skinny_table_pricing.md)** - High-performance vectorized pricing via Basis Extraction.

### What's not? 
We purposely removed richer instruments, curves, calendars, to keep code size down. Next PR-C will add some of notions of market state and model management, that useful in Notebooks and Dashboards.

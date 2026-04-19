# Risk Unification Architecture

This document outlines the design of the **Unified Risk Framework** in `py-flow`, which bridges symbolic (Analytic) and finite-difference (Numerical) sensitivity calculations.

## 1. Motivation
In a reactive, expression-based system, analytic derivatives (AAD) are extremely fast but complex to implement correctly for every new node type. Numerical "bump-and-reprice" risk is slow but mathematically simple and serves as the "gold standard" for correctness. 

The Unified Risk Framework ensures both methods use an identical API, allowing execution engines to switch between them for performance or verification purposes.

## 2. The `FirstOrderRiskBase` Interface
All first-order risk calculators inherit from a shared base that provides standardized discovery of pillars and instruments.

| Method | Description |
| :--- | :--- |
| **`evaluate(ctx)`** | Returns `dict[str, float]`. The primary unified entry point for all engines. |
| **`total_risk()`** | Returns the aggregate sensitivity for the whole portfolio. |
| **`instrument_risk()`** | Returns a mapping of `{InstrumentName: {Pillar: Sensitivity}}`. |
| **`jacobian()`** | Returns the normalized sensitivity `∂(NPV / Notional) / ∂Pillar` for solvers. |

## 3. Two Paths, One API

### Analytic Path (`FirstOrderAnalyticRisk`)
Uses the symbolic DAG to calculate exact derivatives.
- **Implementation**: Leverages `reactive.expr.diff()`.
- **Output**: Returns new `Expr` trees representing the derivative.
- **Advantage**: Instantaneous evaluation once compiled; exact results (no discretization error).

### Numerical Path (**`FirstOrderNumericalRisk`**)
Uses "bump-and-reprice" logic on the live reactive graph.
- **Implementation**: Bumps a `YieldCurvePoint`, triggers a graph `flush()`, and measures the delta in `npv()`.
- **Output**: Returns `float` values.
- **Advantage**: Works even for non-symbolic or complex code; provides a baseline to verify Analytic derivatives.
- **Features**: Supports **forward** (`O(h)`) and **central** (`O(h^2)`) difference methods.

## 4. Engine Integration
Execution engines are designed to be "Risk Method Agnostic".

```python
# Analytic risk via NumPy
engine = SkinnyEngineNumPy()
risk_a = engine.total_risk(portfolio, ctx, risk_method=FirstOrderAnalyticRisk)

# Numerical risk (central diff) via standard Python loop
engine = PythonEngineFloat()
risk_n = engine.total_risk(portfolio, ctx, risk_method=FirstOrderNumericalRisk, method="central")
```

## 5. Verification: The "Golden Run"
The platform encourages a **Refinement Workflow**:
1. Implement new instrument logic using `@computed` (Float path).
2. Calculate and verify risk using `FirstOrderNumericalRisk`.
3. Enable `@traceable` and `FirstOrderAnalyticRisk`.
4. Validate that `abs(Analytic - Numerical) < tolerance`.

This ensures that as the codebase evolves, the high-performance symbolic paths remain financially accurate.

---

## 6. Mathematical Consistency
To maintain parity, both paths handle "Pillar Discovery" identically by scraping the `pillar_points()` of the target portfolio. This ensures that even if a swap depends on multiple curves (e.g. Discount and Projection), risk to both is captured automatically by the Unified API.

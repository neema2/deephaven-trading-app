# Reactive Expression DAG Architecture (reactive/)

This document outlines an architectural extension in py-flow from Python Abstract Syntax Tree (AST) parsing (`@computed`) to potential to have Execution Tracing via Directed Acyclic Graphs (`@traceable`). 

## 1. The AST Limitation vs. Execution Tracing
Historically, the system attempted to generate SQL/pure expression equivalents by statically parsing Python source code via `ast.parse` in `reactive/computed.py`. 
- **The AST Limitation**: AST parsing can struggle with complex logic (loops over arrays, cross-entity method resolutions across multiple curves, dictionaries) since Python syntax is vast. In these cases, the system silently bails out, falling back to native python scalar evaluations which limits SQL generation and symbolic risk differentiation capabilities.
- **A Proposed Extension - `@traceable`**: To augment the existing AST capabilities, we introduce an execution tracing alternative (similar to QuantLib's AutoDiff, PyTorch 2.0, Ibis). Both paradigms remain fundamentally available. Instead of parsing the code text, the python function executes *once* locally. Under normal execution, it natively runs using standard `float`s. When tracing is requested, the system injects `Expr` node wrapper objects (e.g., `Const(100.0)` or `Variable("yield")`). As Python executes basic arithmetic (`+`, `*`, `If()`), it builds an explicit structural DAG memory map.

## 2. DAG Execution Targets
Once a property returns an `Expr` DAG, we can execute that explicit map against three completely different compiler backends for free:
1. **Live Dynamic Ticking**: We traverse the DAG using `.eval_cached(ctx)` locally, replacing `Variables` with live data feeds in micro-seconds.
2. **Database Pushdown (DAG-Aware SQL Generation)**: We compile the tree using `.to_sql()`. Instead of generating repetitive SQL code, the engine identifies common sub-expressions (e.g., interpolated discount factors used by multiple swaps) and compiles them into **Common Table Expressions (CTEs)**. This results in dramatic SQL size reduction and allows database engines (like DuckDB) to optimize the evaluation of shared nodes once per row.
3. **Symbolic Risk Differentiation (Jacobian Pruning)**: The `diff(expr, "PILLAR")` engine algebraically differentiates the polynomial tree to produce an exact Jacobian derivative map `∂npv/∂pillar_rate` mathematically. Sensitivities are computed exactly without the overhead of finite difference (bump-and-grind). Columns with guaranteed zero sensitivity are automatically pruned from the SQL generation.

## 3. Sub-Expression Sharing (`eval_cached`)
When you take the symbolic derivative of a massive function composed of 60 floating periods, you create mathematical expressions that reuse the exact same sub-trees repeatedly (e.g. `df_T`). 
- **The Risk**: Calling a naive recursive `.eval(ctx)` on highly derivative trees results in exponential time-complexity (O(2^N)) re-evaluations of the *same* static branch.
- **The Fix**: `eval_cached(expr, ctx, _cache)` tracks the memory address `id(expr)` of every traversed node. An identical node branch is only evaluated once globally during a pricing tick, reducing the traversal complexity back down to O(V+E) linear time.

## 4. The Magic `_TracedCallable`
To ensure cross-compatibility between standard scalars and pure expression trees, the `@traceable` descriptor acts exactly like dual-mode C++ template auto-differentiation. It automatically gracefully degrades directly to plain Python outputs (dictionaries, strings) for non-arithmetic functions. However, when returning floats:
- `swap.dv01` → Acts like a standard numerical float object, evaluated instantaneously without any tree generation overhead.
- `swap.dv01()` → Invokes the `__call__` dunder method to re-run the calculation natively in Trace mode and structurally compose the explicit `Expr` tree DAG.

## 5. Natural Code Ergonomics — Writing Instruments Like Plain Python
`@traceable` is designed so that instrument pricing methods look and read like ordinary float arithmetic. Three properties of the `Expr` framework make this work transparently:

### 5.1 Implicit Scalar Wrapping via `_wrap()`
Every arithmetic and comparison operator on `Expr` calls the internal `_wrap(other)` helper before constructing a node. `_wrap` promotes any plain Python `int` or `float` to a `Const` node, and passes `Expr` instances through unchanged:
```python
# instrument code — no Const() needed:
rate = projection_curve.fwd(start, end) + self.float_spread   
pv   = df * dt * notional * 0.0001                            
```
This is why `If(expr > 0, "PROFIT", "LOSS")` works without wrapping the literal `0` or the string arguments.

### 5.2 The Accumulator Pattern — Looks Like Float, Builds a Flat Sum
A common pricing loop accumulates coupon present values:
```python
pv = 0.0
for i in range(len(targets)):          
    df = discount_curve.df(targets[i])
    pv += df * dt * notional           
return pv
```
This looks like it might build a left-recursive `BinOp` tree of depth N, which would be extremely slow.
In reality, `__add__` in `Expr` structurally flattens it directly into a single `Sum([t1, t2, ..., tN])` evaluating optimally in O(N).
- **Zero-identity:** `Const(0) + x → x`. The initial `pv = 0.0` evaporates physically from the generated tree.
- **Constant folding:** Two adjacent `Const` nodes reduce mathematically immediately.

## 6. Analytic Differentiation & The Variable Mixin
Tracing natively generates the literal mathematical formula (`Expr` DAG) constructed by the Python execution. By traversing this algebraic DAG using calculus rules (Adjoint Algorithmic Differentiation), we dynamically produce the exact expression formula for the differentials to inputs in at most 4 times the computational cost, if caching of interim calculations is optimized. 

### Why do we need Explicit Variables instead of just `@traceable`?
Fundamentally, `@computed` and `@traceable` are designed to govern intermediate, deterministic logic functions. They evaluate explicit inputs to compute dependent mathematical outcomes. 
When computing Jacobians, the DAG tracing engine needs to know exactly which *edge nodes* (independent degrees of freedom) the output is sensitive to. If we apply standard tracing to a base class property (like an underlying market data quote), trace execution effortlessly extracts the numerical scalar (e.g. `150.0`) and aggressively optimizes it away natively into an untracked static `Const` node. This destroys its identity, blinding the analytical calculus engine. 
Instead, we need the execution tracer to structurally drop an explicit, identifiable leaf node representing exactly that variable into the graph. We achieve this using a specific **Variable Mixin** architecture (currently flagged natively via the targeted `@traceable(is_variable=True)` decorator).

### Parameterized Nodes and Streaming Integration
The Variable Mixin approach becomes important in two ways:
1. **Instance Property Tracking (The Core)**: Similar to how `@computed` extracts target names mapping exactly into explicitly governed schema Column objects utilizing exactly Class Instance names as database Primary Keys natively. Variables track atomic independent fields explicitly (such as tracking isolated boundary values mapped onto underlying static trades vs fluid market parameters).
2. **Parameterized Function Calls (Curve Interpolation)**: A dynamic calculation executing across `df_T(time)` builds massive unique mathematical tree structures parameterized exclusively upon runtime variables natively! The Variable Mixin uniquely allows trees to define exact functional variables (such as explicitly embedding the `time` parameter) rather than simply mapping static property definitions statically.

This fits with the system's objective to integrate functional graphs with streaming capabilities. Because the analytical engine gracefully isolates all dynamic independent boundaries utilizing Variable node object arrays, downstream event systems can arbitrarily stream these independent variables precisely over extremely efficient skinny schemas. We seamlessly adopt the explicit functional identity (the Variable Mixin) scaling securely onto arbitrary functional parameters (like specific interpolated timeline pillars) as dynamically governed secondary indexing keys bridging flawlessly onto upstream primary tables.

---

## Appendix: The Dual Architecture Approach

The integration of both AST parsing (`@computed`) and Execution Tracing (`@traceable`) mirrors one of the most famous architectural paradigm shifts in modern compiler design across both Quantitative Finance and Machine Learning. By retaining both, we support ergonomic parsing alongside explicit, scalable tracing.

| Platform | Paradigm Approach | Description | Strategy Match |
| :--- | :--- | :--- | :--- |
| **Pony ORM** | **AST Decompilation** | Relies dynamically on Python's Abstract Syntax Tree hooks to translate generator comprehensions sequentially into SQL. Extremely powerful locally but highly brittle against unfamiliar looping semantics or dictionaries. | Legacy `@computed` SQL Generator |
| **QuantLib** | **C++ Template Tracing** | Standard industry AAD (Adjoint Algorithmic Differentiation). Uses `typedef Real` where `Real` can compile globally as either native `double` for instant calculation or `std::shared_ptr<Node>` for taping operation histories and building dynamic Jacobians. | The new `@traceable` |
| **PyTorch 2** | **JIT Tracing** | PyTorch previously attempted AST parsing (`TorchScript`), but the dynamic nature of Python meant the engine failed consistently. PyTorch 2.0 abandoned AST parsing to introduce `TorchDynamo`, relying purely on object tracing to map explicit memory architectures. | `TracedFloat` object injection |
| **JAX** | **Virtual Tracers** | Instead of processing Python AST structures, JAX natively pumps `Tracer` objects into ordinary python math functions. The overloaded functions emit structural `jaxpr` objects perfectly matching mathematical operator intent automatically without the backend developer being completely aware. | The new `@traceable` |

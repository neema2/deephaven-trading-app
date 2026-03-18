# Reactive Expression DAG Architecture (reactive/)

This document outlines the core architectural shift in py-flow from Python Abstract Syntax Tree (AST) parsing (`@computed`) to Execution Tracing via Directed Acyclic Graphs (`@computed_expr`). 

## 1. The AST Limitation vs. Execution Tracing
Historically, the system attempted to generate SQL/pure expression equivalents by statically parsing Python source code via `ast.parse` in `reactive/computed.py`. 
- **The Problem**: AST parsing fails rapidly on complex logic (loops over arrays, cross-entity method resolutions across multiple curves, dictionaries). Python syntax is too vast to maintain a parallel compiler. The system would silently bail out (`cp.expr = None`), forcing the engine back to native python scalar evaluations and destroying SQL generation and symbolic risk differentiation capabilities.
- **The Solution - `@computed_expr`**: We shifted to execution tracing (identical to PyTorch 2.0 / JAX / Ibis workflows). Instead of parsing the code text, the python function executes *once* locally. However, instead of passing numbers (e.g., `notional = 100.0`), the system injects `Expr` node wrapper objects (e.g., `Const(100.0)` or `Variable("yield")`). As Python executes basic arithmetic (`+`, `*`, `If()`), it builds an explicit structural DAG memory map.

## 2. DAG Execution Targets
Once a property returns an `Expr` DAG, we can execute that explicit map against three completely different compiler backends for free:
1. **Live Dynamic Ticking**: We traverse the DAG using `.eval_cached(ctx)` locally, replacing `Variables` with live data feeds in micro-seconds.
2. **Database Pushdown (DAG-Aware SQL Generation)**: We compile the tree using `.to_sql()`. Instead of generating repetitive SQL code, the engine identifies common sub-expressions (e.g., interpolated discount factors used by multiple swaps) and compiles them into **Common Table Expressions (CTEs)**. This results in dramatic SQL size reduction and allows database engines (like DuckDB) to optimize the evaluation of shared nodes once per row.
3. **Symbolic Risk Differentiation (Jacobian Pruning)**: The `diff(expr, "PILLAR")` engine algebraically differentiates the polynomial tree to produce an exact Jacobian derivative map `∂npv/∂pillar_rate` mathematically. Sensitivities are computed exactly without the overhead of finite difference (bump-and-grind). Columns with guaranteed zero sensitivity (e.g., a 1Y swap is not sensitive to the 30Y pillar) are automatically pruned from the SQL generation. This minimizes data transfer and simplifies the final query result.

## 3. Sub-Expression Sharing (`eval_cached`)
When you take the symbolic derivative of a massive function composed of 60 floating periods, you create mathematical expressions that reuse the exact same sub-trees repeatedly (e.g. `df_T`). 
- **The Risk**: Calling a naive recursive `.eval(ctx)` on highly derivative trees results in exponential time-complexity (O(2^N)) re-evaluations of the *same* static branch.
- **The Fix**: `eval_cached(expr, ctx, _cache)` tracks the memory address `id(expr)` of every traversed node. An identical node branch is only evaluated once globally during a pricing tick, reducing the traversal complexity back down to O(V+E) linear time.

## 4. The Magic `CallableFloat`
To ensure cross-compatibility between standard scalars (`@computed`) and complex trees (`@computed_expr`), the `@computed_expr` descriptor produces an overloaded `CallableFloat` return value.
- `swap.dv01` → Acts like a standard lazily evaluated numerical float object.
- `swap.dv01()` → Invokes the `__call__` dunder method to expose the raw underlying `Expr` tree DAG for SQL and derivative transformations.

However, explicitly expanding this pattern to `CallableStr` and `CallableDict` objects induces severe upstream instability when paired natively with PyArrow streaming. During `@ticking` serialization, PyArrow's schema inference frequently fails on these structures, attempting to forcibly coerce them into `double` native arrays (`ArrowInvalid: Could not convert 'LOSS' with type CallableStr: tried to convert to double`).

**Best Practices for Streaming:**
1. **Complex Dictionaries:** Multi-dimensional maps evaluated via `Expr` derivatives (like `risk` arrays) must be explicitly excluded from continuous stream ingestion: `@ticking(exclude={"curve", "risk"})`.
2. **Text State Signals:** Standard string emissions (like `pnl_status` returning "PROFIT"/"LOSS") must avoid `@computed_expr` entirely and fall back to native `@computed` scalar execution returning raw native strings to align safely with PyArrow types.

---

## Appendix: The Architectural Shift

The journey from AST parsing (`@computed`) to Execution Tracing (`@computed_expr`) mirrors one of the most famous architectural paradigm shifts in modern compiler design, specifically within Machine Learning and Data Engineering.

### 1. PyTorch 2.0: The Failure of AST Parsing (`TorchScript` vs `TorchDynamo`)
PyTorch went through this exact evolution, culminating in the release of PyTorch 2.0:
* **The AST Approach (`TorchScript`)**: Historically, PyTorch engineers built `@torch.jit.script` (similar to our AST parser). It parsed Python source code's AST to build a static C++ graph. The problem? Python's dynamic nature is vast. `TorchScript` continually struggled to handle Python dictionaries, list comprehensions, or dynamic `if` statements, forcing PyTorch engineers to maintain a fragile parallel Python compiler.
* **The Tracing Approach (`TorchDynamo`)**: In PyTorch 2.0, they entirely abandoned the AST approach. They introduced **TorchDynamo**, which uses "JIT Tracing". Instead of reading source code, it hooks into Python's native execution. When you run a function, Torch tracks operations dynamically to build the graph (just like our `@computed_expr` nodes do). If it hits something it doesn't understand, it does a "graph break" and falls back to standard Python.

### 2. JAX: Pure Tracing by Design
Google's **JAX** recognized the limitations of AST parsing and decided never to build one. As detailed in the famous *"Autodiff Cookbook"*, JAX's tracing architecture works elegantly:
When you decorate a function with `@jax.jit` and call it, JAX doesn't pass in real data. It passes in "Tracer" objects. These Tracers look and act exactly like Python floats (much like our `CallableFloat`), but when you add them together, they silently record the operation to an internal graph called a `jaxpr` (JAX Expression). JAX proved that Operator Overloading + Execution Tracing is vastly superior and easier to maintain than AST parsing.

### 3. Frameworks for Deriving SQL
In the data engineering world, several frameworks mirror these exact two approaches for SQL generation:

**The Tracing / Operator Overloading Frameworks**
* **Ibis (by Voltron Data):** The industry standard for explicit expression building. Ibis allows you to write Python code that dynamically chains together mathematical operations. Under the hood, it builds an explicit expression tree that compiles flawlessly into DuckDB, Snowflake, or Postgres SQL.
* **SQLAlchemy 2.0:** SQLAlchemy's Expression Language works exactly like our `Expr` tracking. It overloads arithmetic operators to return expression nodes, which later compile to SQL string fragments.

**The AST Parsing Frameworks**
* **Pony ORM:** One of the few famous frameworks that went the AST compilation route. In Pony ORM, you write literal Python generator expressions. Pony hooks into Python's AST, decompiles your generator into an abstract tree, and attempts to translate it to SQL. Just like our `@computed`, it feels like absolute magic when it works, but it famously crashes if you use an unsupported Python feature.

By implementing *both* in our `py-flow` framework, we essentially have the magic of **Pony ORM** for simple column-level logic, and the unconstrained scale of **Ibis / JAX** for deep, telescoping financial pricing graphs!

---

## 5. Natural Code Ergonomics — Writing Instruments Like Plain Python

`@computed_expr` is designed so that instrument pricing methods look and read like
ordinary float arithmetic. Three properties of the `Expr` framework make this work
transparently. Tests for all three live in `tests/test_expr_ergonomics.py`.

### 5.1 Implicit Scalar Wrapping via `_wrap()`

Every arithmetic and comparison operator on `Expr` calls the internal `_wrap(other)`
helper before constructing a node. `_wrap` promotes any plain Python `int` or `float`
to a `Const` node, and passes `Expr` instances through unchanged:

```python
# instrument code — no Const() needed:
rate = projection_curve.fwd(start, end) + self.float_spread   # float auto-wrapped
pv   = df * dt * notional * 0.0001                            # all scalars wrapped

# exactly equivalent to:
rate = projection_curve.fwd(start, end) + Const(self.float_spread)
pv   = df * dt * Const(notional) * Const(0.0001)
```

`_wrap` is invoked by `__mul__`, `__add__`, `__sub__`, `__truediv__`, `__rmul__`,
`__radd__`, and all comparison operators — so scalars may appear on **either side**
of an expression. This is also why `If(expr > 0, "PROFIT", "LOSS")` works without
wrapping the literal `0` or the string arguments.

`get_expr()` inside `computed_expr` extends this to **return values**: a `@computed_expr`
body returning a bare `0.0` for an early-exit guard is auto-promoted to `Const(0.0)`,
so callers never need to write `return Const(0.0)`.

### 5.2 The Accumulator Pattern — Looks Like Float, Builds a Flat Sum

A common pricing loop accumulates coupon present values:

```python
pv = 0.0
for i in range(len(targets)):          # could be 20, 40, 60 coupon periods
    df = discount_curve.df(targets[i])
    pv += df * dt * notional           # looks like float accumulation
return pv
```

This **looks** like it might build a left-recursive `BinOp` tree of depth N —
which would be slow to differentiate and even slower to evaluate naively.
In reality, `__add__` in `Expr` maintains a flat structure throughout:

| Step | Operation | Result type | Tree depth |
|------|-----------|-------------|------------|
| seed | `pv = 0.0` | `float` (not yet an Expr) | — |
| 1st `+=` | `_wrap(0.0) + BinOp` → zero-identity fires | `BinOp` (the term) | 1 |
| 2nd `+=` | `BinOp + BinOp` → default branch | `Sum([t1, t2])` | **1** |
| 3rd `+=` | `Sum + BinOp` → flatten branch | `Sum([t1, t2, t3])` | **1** |
| Nth `+=` | `Sum + BinOp` → flatten extends | `Sum([t1…tN])` | **1** |

Key rules in `__add__` that keep this flat:

- **Zero-identity (line 48–49):** `Const(0) + x → x`. The seed zero disappears
  on the first accumulation — no dead weight in the tree.
- **Constant folding (line 52–54):** Two adjacent `Const` nodes are reduced to one.
- **Sum flattening (lines 57–63):** Whenever either operand is already a `Sum`,
  its `terms` list is spliced directly into a new flat `Sum` — no nesting.
- **Default promotion (line 66):** Any two non-Sum, non-zero exprs produce `Sum([a, b])`
  instead of `BinOp("+", a, b)`, priming the next step for the flatten branch.

The resulting `Sum([t1, t2, …, tN])` evaluates in O(N) time and differentiates
cleanly — each term contributes an independent additive branch to $\partial/\partial x$.

> **Trap for the unwary:** Someone reading the accumulator loop might assume `pv = 0.0`
> is a performance hint or a no-op, and refactor it away. In fact it is the intended
> idiomatic pattern — it seeds the type promotion chain that lets Expr arithmetic
> interleave naturally with Python scalars.

### 5.3 `pnl_status` Must Use `@computed`, Not `@computed_expr`

String-valued signals like `"PROFIT"` / `"LOSS"` / `"FLAT"` must **not** use
`@computed_expr`. The `@ticking` schema resolver inspects the decorated property's
return annotation to choose the PyArrow column type. `@computed_expr` wraps the
function inside `_compute_val`, hiding the annotation; the resolver then defaults
to `float`, and PyArrow rejects the string value at write time with:

```
ArrowInvalid: Could not convert 'PROFIT' with type CallableStr: tried to convert to double
```

Use plain `@computed` for any string-valued computed property:

```python
# correct — annotation visible to ticking schema resolver
@computed
def pnl_status(self) -> str:
    val = self.npv          # CallableFloat, supports > / < directly
    if val > 0: return "PROFIT"
    if val < 0: return "LOSS"
    return "FLAT"
```


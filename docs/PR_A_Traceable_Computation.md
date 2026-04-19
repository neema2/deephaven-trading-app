# PR-A: Traceable Computation

This PR is the first of 3 stacked PRs covering the features of PR-3 (Curve fitters with analytic differentiation) and PR-4 (IR scenario risk).

PR-A: sensitive changes to core reactive framework
PR-B: majority of new code: example pricers, fitters, scenarios
PR-C: streaming dashboards & static notebooks

---

## What is in this PR?
PR-A isolates the reactive engine change. It addresses the PR-3 review concern of architectural change by unifying two approaches: the current AST parser `@computed` ensures code is easy to debug, operating on simple floats; while PR-3 proposed `@computed_expr` to enable richer expression tree generation, but required the users code to operate on Expr objects. This PR-A proposes a unified `@traceable` instead, that does both using Python type flexibility to extend the Float type. This enables code to operate identically to the original, or be traced.

**👉 [docs/architecture/reactive_expr_dag.md](docs/architecture/reactive_expr_dag.md)**

The code encourages the tagging of some inputs using `@traceable(is_variable=True)` (such as market data) while leaving others as more static data (such as trade definitions).  This enables analytic differentiation to calculate risk to only the variables.

A minimal test is provided, to compare `@computed` and `@traceable` reactive behaviour and their Expr generation.   A second minimal test is provided to demonstrate the differentiation to variables.   More will be added, but for this submission choosing to keep the PR slim.

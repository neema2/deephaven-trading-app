"""
instruments/portfolio — Named collection of instrument Expr trees.

Provides the Portfolio class, which aggregates multiple IRSwapFixedFloatApprox
Expr trees on a shared curve.  This enables maximum sub-expression sharing
and provides symbolic Jacobian matrices for the fitter.
"""

from __future__ import annotations
from collections import Counter
from typing import Any

from reactive.expr import (
    Const, Expr, diff, eval_cached, 
    Variable, VariableMixin, Field,
    BinOp, UnaryOp, Func, If, Coalesce, IsNull, StrOp,
    _cast_numeric_sql
)
from instruments.ir_swap_fixed_floatapprox import IRSwapFixedFloatApprox


class Portfolio:
    """A collection of named swap Expr trees on a shared curve.

    Because all swaps call the same curve's df(t), and df
    caches Expr objects, the portfolio's expression graph has maximum
    sub-expression sharing.  For example, df(5.0) is the same
    Python object in the 5Y swap's tree and the 10Y swap's tree.

    Provides:
      .npv_exprs         → {name: Expr}           named NPV expressions
      .residual_exprs    → {name: Expr}           NPV/notional (for fitter)
      .risk_exprs        → {name: {pillar: Expr}} per-swap Jacobian rows
      .jacobian_exprs    → {name: {pillar: Expr}} ∂residual/∂pillar (for fitter)
      .total_npv_expr    → Expr                   Σ NPV across portfolio

    All return Expr trees — eval(ctx) for Python, to_sql() for SQL.
    """

    def __init__(self):
        self._instruments: dict[str, Any] = {}
        self._all_pillar_names: set[str] = set()

    def add_instrument(self, name: str, instrument: Any):
        """Add any pre-constructed instrument (swap, etc.) to the portfolio.
        Must support .npv() and optionally .notional.
        """
        self._instruments[name] = instrument
        # Update pillar names union
        if hasattr(instrument, "pillar_names"):
            self._all_pillar_names.update(instrument.pillar_names)
        return instrument

    @property
    def names(self) -> list[str]:
        return list(self._instruments.keys())

    @property
    def pillar_names(self) -> list[str]:
        return sorted(list(self._all_pillar_names))

    # ── Named dictionaries of Expr trees ───────────────────────────────

    @property
    def npv_exprs(self) -> dict[str, Expr]:
        """Named NPV expressions: {name: npv_expr}."""
        return {name: inst.npv() for name, inst in self._instruments.items()}

    @property
    def residual_exprs(self) -> dict[str, Expr]:
        """NPV / notional for each instrument (what the fitter minimizes)."""
        res = {}
        for name, inst in self._instruments.items():
            notional = getattr(inst, "notional", 
                       getattr(inst, "leg1_notional", 1.0))
            res[name] = inst.npv() * Const(1.0 / notional)
        return res

    @property
    def total_npv_expr(self) -> Expr:
        """Sum of all NPVs — a single Expr tree."""
        from reactive.expr import Sum
        return Sum([inst.npv() for inst in self._instruments.values()])

    @property
    def risk_exprs(self) -> dict[str, dict[str, Expr]]:
        """Per-instrument risk: {name: {pillar: ∂npv/∂pillar Expr}}."""
        memo: dict = {}
        pillars = self.pillar_names
        return {
            name: {
                pillar_name: diff(inst.npv(), pillar_name, _memo=memo)
                for pillar_name in pillars
            }
            for name, inst in self._instruments.items()
        }

    @property
    def jacobian_exprs(self) -> dict[str, dict[str, Expr]]:
        """Fitter Jacobian: ∂residual_i / ∂pillar_j as Expr trees.
        Each row is an instrument, each column is a pillar.
        """
        result = {}
        memo: dict = {}
        pillars = self.pillar_names
        for name, inst in self._instruments.items():
            notional = getattr(inst, "notional", 
                       getattr(inst, "leg1_notional", 1.0))
            scale = Const(1.0 / notional)
            result[name] = {
                pillar_name: diff(inst.npv(), pillar_name, _memo=memo) * scale
                for pillar_name in pillars
            }
        return result

    # ── Convenience evaluators ─────────────────────────────────────────

    def pillar_context(self) -> dict[str, float]:
        """Current pillar rates aggregated from all instruments' curves."""
        ctx = {}
        for inst in self._instruments.values():
            if hasattr(inst, "pillar_context"):
                ctx.update(inst.pillar_context())
        return ctx

    def eval_npvs(self, ctx: dict) -> dict[str, float]:
        """Evaluate all NPVs."""
        return {name: eval_cached(expr, ctx) for name, expr in self.npv_exprs.items()}

    def eval_residuals(self, ctx: dict) -> dict[str, float]:
        """Evaluate residual vector (what the fitter minimizes)."""
        return {name: eval_cached(expr, ctx) for name, expr in self.residual_exprs.items()}

    def eval_instrument_risk(self, ctx: dict) -> dict[str, dict[str, float]]:
        """Evaluate the risk per instrument (full Jacobian matrix)."""
        cache: dict = {}
        return {
            name: {label: eval_cached(expr, ctx, _cache=cache) for label, expr in row.items()}
            for name, row in self.jacobian_exprs.items()
        }

    def eval_total_risk(self, ctx: dict) -> dict[str, float]:
        """Aggregate ∂(Σnpv)/∂pillar across all instruments."""
        total_expr = self.total_npv_expr
        cache: dict = {}
        return {
            pillar_name: eval_cached(diff(total_expr, pillar_name), ctx, _cache=cache)
            for pillar_name in self.pillar_names
        }

    # ── Sub-expression sharing stats ───────────────────────────────────

    def shared_nodes(self) -> dict[str, int]:
        """Count how many instruments reference each cached node."""
        node_ids: Counter[int] = Counter()
        for inst in self._instruments.values():
            seen = set()
            _walk_ids(inst.npv(), seen)
            for nid in seen:
                node_ids[nid] += 1
        return {f"node_{nid}": count for nid, count in node_ids.items() if count > 1}

    def to_sql_optimized(self, ctx: dict[str, float]) -> str:
        """Generate a single optimized SQL query for the entire Portfolio Jacobian.

        Uses Common Table Expressions (CTEs) to preserve the Expr DAG structure,
        ensuring shared sub-expressions (like discount factors) are computed
        exactly once.

        Implements pruning: zero derivatives (Const 0.0) are omitted from the
        output columns.

        Returns a single SQL string with:
          - A 'pillars' CTE for the input rates.
          - Multiple 'node_NNN' CTEs for shared sub-expressions.
          - A final SELECT with all NPV and Jacobian columns.
        """
        # 1. Identify all target expressions and their sub-expression counts
        npv_targets = self.npv_exprs
        jac_targets = self.jacobian_exprs

        all_exprs: list[Expr] = list(npv_targets.values())
        relevant_jac: dict[str, dict[str, Expr]] = {}

        for swap_name, row in jac_targets.items():
            relevant_jac[swap_name] = {}
            for pillar, expr in row.items():
                if isinstance(expr, Const) and expr.value == 0.0:
                    continue # Prune zero sensitivities
                relevant_jac[swap_name][pillar] = expr
                all_exprs.append(expr)

        # 2. Find shared nodes (in-degree > 1 across the whole portfolio)
        node_counts: Counter[int] = Counter()
        def _collect(e: Expr, visited: set[int]):
            node_counts[id(e)] += 1
            if id(e) in visited: return
            visited.add(id(e))
            for child in _get_children(e):
                _collect(child, visited)

        global_visited: set[int] = set()
        for expr in all_exprs:
            _collect(expr, global_visited)

        shared_node_ids = {nid for nid, count in node_counts.items() if count > 1}

        # 3. Topological sort of shared nodes for CTE generation
        # (We need to define a node's CTE after its children's CTEs)
        ordered_shared: list[Expr] = []
        visited_shared = set()
        def _topo(e: Expr):
            if id(e) in visited_shared: return
            for child in _get_children(e):
                _topo(child)
            if id(e) in shared_node_ids:
                visited_shared.add(id(e))
                ordered_shared.append(e)

        for expr in all_exprs:
            _topo(expr)

        # 4. Generate SQL fragments
        cte_defs = []
        # Input pillars
        pillar_cols = ", ".join(f'{rate} AS "{name}"' for name, rate in ctx.items())
        cte_defs.append(f'pillars AS (SELECT {pillar_cols})')

        # Use a substitution map: if id(e) is in here, use the CTE name
        subst: dict[int, str] = {}

        def _get_sql(e: Expr) -> str:
            return _to_sql_dag(e, subst, "pillars")

        # Shared node CTEs
        for i, node in enumerate(ordered_shared):
            # Leaves don't need their own CTEs if they are just pillars or constants
            if isinstance(node, (Variable, VariableMixin, Const)):
                subst[id(node)] = _get_sql(node)
                continue

            name = f"node_{i}"
            sql = _get_sql(node)
            
            # Identify which previous nodes this CTE depends on
            dependencies = ["pillars"]
            for prev_idx in range(i):
                if f"node_{prev_idx}.val" in sql:
                    dependencies.append(f"node_{prev_idx}")
            
            from_clause = ", ".join(dependencies)
            cte_defs.append(f'{name} AS (SELECT ({sql}) AS val FROM {from_clause})')
            subst[id(node)] = f"{name}.val"

        # 5. Final SELECT columns
        select_cols = []
        for name, expr in npv_targets.items():
            select_cols.append(f'{_get_sql(expr)} AS "{name}_NPV"')
        
        for swap_name, row in relevant_jac.items():
            for pillar, expr in row.items():
                col_name = f"{swap_name}_dNPV_d{pillar}"
                select_cols.append(f'{_get_sql(expr)} AS "{col_name}"')

        with_clause = "WITH " + ",\n  ".join(cte_defs)
        
        # Identify all final dependencies for the final SELECT
        final_deps = ["pillars"]
        final_sql_all = " ".join(select_cols)
        for i in range(len(ordered_shared)):
             if f"node_{i}.val" in final_sql_all:
                 final_deps.append(f"node_{i}")

        from_final = ", ".join(final_deps)
        final_select = "SELECT\n  " + ",\n  ".join(select_cols) + f"\nFROM {from_final}"

        return f"{with_clause}\n{final_select}"

    def to_skinny_components(self, extractor=None, per_swap=True) -> list[dict]:
        """Flatten the Portfolio NPVs and Jacobians into a list of basis function components.
        
        If per_swap=False, identical components across all trades are collapsed (agg weights),
        saving both data movement and redundant math calculations.
        """
        if extractor is None:
            from reactive.basis_extractor import BasisExtractor
            extractor = BasisExtractor()
            
        # Use aggregation to collapse identical atoms if per_swap is False
        # key = (Component_Class, Component_Type, X1, X2..., p1, p2...)
        agg_map: dict[tuple, float] = {}

        def _add_to_agg(trade_name, comp_class, expr):
            trade_components = extractor.extract_components(expr)
            for bf, vars, params, weight in trade_components:
                # Key ignores Trade_Id if per_swap is False
                key_prefix = (trade_name if per_swap else None, comp_class, bf.component_type)
                full_key = key_prefix + tuple(vars) + tuple(params)
                agg_map[full_key] = agg_map.get(full_key, 0.0) + weight

        # 1. NPV Components
        for name, expr in self.npv_exprs.items():
            _add_to_agg(name, "NPV", expr)
                
        # 2. Jacobian Components
        for name, row_exprs in self.jacobian_exprs.items():
            for pillar, expr in row_exprs.items():
                if isinstance(expr, Const) and expr.value == 0.0:
                    continue # Pruning zero sensitivities
                _add_to_agg(name, f"dNPV_d{pillar}", expr)

        # 3. Convert map back to list of dicts
        components_list = []
        for key, total_weight in agg_map.items():
            trade_name, comp_class, comp_type = key[:3]
            # Find the bf and get its param/var counts from the registry
            bf = next((b for b in extractor.registry.values() if b.component_type == comp_type), None)
            if bf is None:
                raise ValueError(f"Unknown component type: {comp_type}")

            vars = key[3:3 + bf.num_vars]
            params = key[3 + bf.num_vars:]
            
            row = {
                "Component_Class": comp_class,
                "Component_Type": comp_type,
                "Weight": total_weight
            }
            if per_swap:
                row["Swap_Id"] = trade_name
            for j, v in enumerate(vars):
                row[f"X{j+1}"] = v
            for j, p in enumerate(params):
                row[f"p{j+1}"] = p
            components_list.append(row)
                    
        return components_list

    def to_skinny_sql_query(self, extractor, per_swap=True) -> str:
        """Generate static duckdb vectorized SQL query to evaluate the skinny components."""
        max_vars = max((bf.num_vars for bf in extractor.registry.values()), default=1)
        
        select_cols = []
        if per_swap:
            select_cols.append("c.Swap_Id")
        
        math_case_lines = []
        for bf in extractor.registry.values():
            s_expr = bf.sql_template.replace("^", "**")
            for j in range(1, bf.num_vars + 1):
                s_expr = s_expr.replace(f"X{j}", f"s{j}.Knot_Value")
            math_case_lines.append(f"                WHEN {bf.component_type} THEN {s_expr}")
        
        math_case_stmt = "CASE c.Component_Type\n" + "\n".join(math_case_lines) + "\n            END"
        
        select_cols.append(f"SUM(CASE WHEN c.Component_Class = 'NPV' THEN c.Weight * ({math_case_stmt}) ELSE 0.0 END) as NPV")
        
        # Add a column for every pillar risk
        for pillar in self.pillar_names:
            col_name = f"dNPV_d{pillar}"
            select_cols.append(f'SUM(CASE WHEN c.Component_Class = \'{col_name}\' THEN c.Weight * ({math_case_stmt}) ELSE 0.0 END) as "{col_name}"')
            
        select_clause = ",\n        ".join(select_cols)
        
        # Use CROSS JOIN for scenario set + LEFT JOINs for knot lookups
        joins = [
            "FROM t_components c",
            "    CROSS JOIN (SELECT DISTINCT Scenario_Id FROM t_scenarios) sc",
        ]
        for j in range(1, max_vars + 1):
            joins.append(f"    LEFT JOIN t_scenarios s{j} ON c.X{j} = s{j}.Knot_Id AND sc.Scenario_Id = s{j}.Scenario_Id")
            
        joins_clause = "\n".join(joins)
        
        group_by = "sc.Scenario_Id"
        if per_swap:
            group_by += ", c.Swap_Id"
            
        return f"SELECT\n    {group_by},\n    {select_clause}\n{joins_clause}\nGROUP BY {group_by}"

def _to_sql_dag(expr: Expr, subst: dict[int, str], col: str) -> str:
    """Efficiently build SQL fragment for a DAG node using substitution map."""
    if id(expr) in subst:
        return subst[id(expr)]

    # Handle leaves
    if isinstance(expr, (Variable, VariableMixin)):
        return expr.expr_to_sql(col)
    if isinstance(expr, Const):
        return expr.to_sql(col)
    if isinstance(expr, Field):
        return expr.to_sql(col)

    # Handle composite nodes
    from reactive.expr import Sum as SumExpr
    if isinstance(expr, SumExpr):
        parts = [_to_sql_dag(t, subst, col) for t in expr.terms]
        if not parts:
            return "0"
        return "(" + " + ".join(parts) + ")"

    if isinstance(expr, BinOp):
        l_sql = _to_sql_dag(expr.left, subst, col)
        r_sql = _to_sql_dag(expr.right, subst, col)
        
        # Numeric casting logic from expr.py
        if expr.op in ("+", "-", "*", "/", "%", "**", ">", "<", ">=", "<="):
            if id(expr.left) not in subst and isinstance(expr.left, Field):
                l_sql = f"({col}->>'{expr.left.name}')::float"
            if id(expr.right) not in subst and isinstance(expr.right, Field):
                r_sql = f"({col}->>'{expr.right.name}')::float"
        
        from reactive.expr import _SQL_OPS
        op = _SQL_OPS[expr.op]
        return f"({l_sql} {op} {r_sql})"

    if isinstance(expr, UnaryOp):
        s = _to_sql_dag(expr.operand, subst, col)
        if id(expr.operand) not in subst and isinstance(expr.operand, Field):
            s = f"({col}->>'{expr.operand.name}')::float"
            
        if expr.op == "neg": return f"(-{s})"
        if expr.op == "abs": return f"ABS({s})"
        if expr.op == "not": return f"NOT ({s})"
        return s

    if isinstance(expr, Func):
        from reactive.expr import Func as PREFunc
        sql_name = PREFunc._SQL_FUNCS.get(expr.name, expr.name.upper())
        args = []
        for a in expr.args:
            asql = _to_sql_dag(a, subst, col)
            if id(a) not in subst and isinstance(a, Field):
                asql = f"({col}->>'{a.name}')::float"
            args.append(asql)
        return f"{sql_name}({', '.join(args)})"

    if isinstance(expr, If):
        cond = _to_sql_dag(expr.condition, subst, col)
        t = _to_sql_dag(expr.then_, subst, col)
        e = _to_sql_dag(expr.else_, subst, col)
        return f"CASE WHEN {cond} THEN {t} ELSE {e} END"

    # Fallback to naive if unknown (should not happen for basic IRS)
    return expr.to_sql(col)


def _get_children(expr: Expr) -> list[Expr]:
    """Helper to extract child expressions from any node type."""
    from reactive.expr import Sum
    if isinstance(expr, Sum):
        return list(expr.terms)
    if isinstance(expr, BinOp):
        return [expr.left, expr.right]
    if isinstance(expr, UnaryOp):
        return [expr.operand]
    if isinstance(expr, (Func, Coalesce)):
        return list(getattr(expr, 'args', [])) if isinstance(expr, Func) else list(getattr(expr, 'exprs', []))
    if isinstance(expr, If):
        return [expr.condition, expr.then_, expr.else_]
    if isinstance(expr, (IsNull, StrOp)):
        res = [expr.operand]
        if hasattr(expr, 'arg') and expr.arg:
            res.append(expr.arg)
        return res
    return []



def _walk_ids(expr: Expr, seen: set[int]) -> None:
    """Walk an Expr tree collecting node ids."""
    nid = id(expr)
    if nid in seen:
        return
    seen.add(nid)
    for child in _get_children(expr):
        _walk_ids(child, seen)


def expr_to_executable_sql(expr: Expr, ctx: dict[str, float]) -> str:
    """Wrap an Expr's to_sql() fragment in a complete executable query."""
    cols = ", ".join(f'{rate} AS "{name}"' for name, rate in ctx.items())
    fragment = expr.to_sql()
    return f'WITH pillars AS (SELECT {cols})\nSELECT ({fragment}) AS result FROM pillars'

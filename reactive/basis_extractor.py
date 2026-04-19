import hashlib
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field

from reactive.expr import Expr, Const, VariableMixin, Variable, BinOp, UnaryOp, Func, If, Coalesce, IsNull, StrOp, Field, Sum

@dataclass
class BasisFunction:
    component_type: int
    signature: str
    sql_template: str
    dh_template: str
    num_vars: int
    num_params: int

class BasisExtractor:
    """
    Dynamically extracts basis functions from Expr DAGs by fingerprinting
    the mathematical structure, replacing variables with X1..Xn and
    constants with p1..pn.
    """
    def __init__(self):
        self.registry: Dict[str, BasisFunction] = {} # signature -> BasisFunction
        self.registry_by_type: Dict[int, BasisFunction] = {} # type_id -> BasisFunction
        self.next_type_id = 1
        
    def extract_components(self, expr: Expr) -> List[Tuple[BasisFunction, List[str], List[float], float]]:
        """
        Takes an overarching Expr (like a Swap NPV) and recursively breaks down
        any top-level additions or subtractions into individual components.
        Returns a list of (BasisFunction, vars, params, weight_multiplier)
        where weight_multiplier handles subtractions (e.g. -1.0 for right side of a subtraction).
        """
        components = []
        
        def _breakdown(node: Expr, multiplier: float):
            if isinstance(node, Sum):
                for term in node.terms:
                    _breakdown(term, multiplier)
            elif isinstance(node, BinOp) and node.op == "+":
                _breakdown(node.left, multiplier)
                _breakdown(node.right, multiplier)
            elif isinstance(node, BinOp) and node.op == "-":
                _breakdown(node.left, multiplier)
                _breakdown(node.right, -multiplier)
            elif isinstance(node, BinOp) and node.op == "*":
                # Distribute scalar multiplication into the weight
                if isinstance(node.left, Const):
                    _breakdown(node.right, multiplier * node.left.value)
                elif isinstance(node.right, Const):
                    _breakdown(node.left, multiplier * node.right.value)
                else:
                    # Non-constant multiplier — extract as single basis
                    bf, extracted_vars, extracted_params = self.extract(node)
                    components.append((bf, extracted_vars, extracted_params, multiplier))
            elif isinstance(node, UnaryOp) and node.op == "neg":
                _breakdown(node.operand, -multiplier)
            elif isinstance(node, Const) and node.value == 0:
                pass # ignore zero constants added
            else:
                # We reached a non-additive element, extract it as a basis!
                bf, extracted_vars, extracted_params = self.extract(node)
                components.append((bf, extracted_vars, extracted_params, multiplier))
                
        _breakdown(expr, 1.0)
        return components
        
    def extract(self, expr: Expr) -> Tuple[BasisFunction, List[str], List[float]]:
        """
        Extracts the basis function and the specific parameters for a given Expr.
        Returns (BasisFunction, list_of_variable_names, list_of_constant_values)
        """
        # State for the current traversal
        self._current_vars = []
        self._current_params = []
        self._var_map = {} # variable_name -> index (e.g., 'X1')
        
        signature, sql_tmpl, dh_tmpl = self._visit(expr)
        
        if signature not in self.registry:
            bf = BasisFunction(
                component_type=self.next_type_id,
                signature=signature,
                sql_template=sql_tmpl,
                dh_template=dh_tmpl,
                num_vars=len(self._current_vars),
                num_params=len(self._current_params)
            )
            self.registry[signature] = bf
            self.registry_by_type[bf.component_type] = bf
            self.next_type_id += 1
            
        # We MUST return the expected number of variables for the registered bf signature,
        # not the current lengths which might be different if it clashed (which it shouldn't unless hash collision,
        # but let's just make sure the registry works correctly). 
        # Actually, if signature is identical, num_vars must be identical.
        return self.registry[signature], list(self._current_vars), list(self._current_params)

    def _visit(self, expr: Expr) -> Tuple[str, str, str]:
        """
        Recursively visit nodes.
        Returns (canonical_signature, sql_template, dh_template)
        """
        if isinstance(expr, Const):
            # For simplicity, parameterize all constants for now.
            val = expr.value
            self._current_params.append(val)
            pid = f"p{len(self._current_params)}"
            # Signature must include the parameter index to differentiate p1..p1 vs p1..p2
            return f"C_{pid}", pid, pid

        if isinstance(expr, (Variable, VariableMixin)):
            name = expr.name
            if name not in self._var_map:
                self._current_vars.append(name)
                self._var_map[name] = f"X{len(self._current_vars)}"
            vid = self._var_map[name]
            return f"V_{vid}", vid, vid
            
        if isinstance(expr, BinOp):
            l_sig, l_sql, l_dh = self._visit(expr.left)
            r_sig, r_sql, r_dh = self._visit(expr.right)
            
            # Note: For commutative ops (+, *), we could sort the signatures to find more commonalities.
            # But naive traversal is fine for V1.
            op = expr.op
            
            # Canonical signature: OP(left, right)
            sig = f"{op.upper()}({l_sig},{r_sig})"
            
            # SQL / DH mapping
            from reactive.expr import _SQL_OPS, _PURE_OPS
            sql_op = _SQL_OPS.get(op, op)
            dh_op = _PURE_OPS.get(op, op)
            
            sql_tmpl = f"({l_sql} {sql_op} {r_sql})"
            
            if op == "**":
                dh_tmpl = f"Math.pow({l_dh}, {r_dh})"
            else:
                dh_tmpl = f"({l_dh} {dh_op} {r_dh})"
                
            return sig, sql_tmpl, dh_tmpl

        if isinstance(expr, Sum):
            term_sigs = []
            term_sqls = []
            term_dhs = []
            for term in expr.terms:
                s, sq, dh = self._visit(term)
                term_sigs.append(s)
                term_sqls.append(sq)
                term_dhs.append(dh)
            sig = "SUM(" + ",".join(term_sigs) + ")"
            sql_tmpl = "(" + " + ".join(term_sqls) + ")"
            dh_tmpl = "(" + " + ".join(term_dhs) + ")"
            return sig, sql_tmpl, dh_tmpl

        if isinstance(expr, Func):
            arg_sigs = []
            arg_sqls = []
            arg_dhs = []
            for arg in expr.args:
                s, sq, dh = self._visit(arg)
                arg_sigs.append(s)
                arg_sqls.append(sq)
                arg_dhs.append(dh)
                
            sig = f"FUNC_{expr.name.upper()}(" + ",".join(arg_sigs) + ")"
            
            from reactive.expr import Func as ExprFunc
            sql_name = ExprFunc._SQL_FUNCS.get(expr.name, expr.name.upper())
            dh_name = ExprFunc._DH_FUNCS.get(expr.name, expr.name)
            
            sql_tmpl = f"{sql_name}(" + ", ".join(arg_sqls) + ")"
            dh_tmpl = f"{dh_name}(" + ", ".join(arg_dhs) + ")"
            
            return sig, sql_tmpl, dh_tmpl

        if isinstance(expr, UnaryOp):
            s, sq, dh = self._visit(expr.operand)
            sig = f"UNARY_{expr.op.upper()}({s})"
            
            if expr.op == "neg":
                return sig, f"(-{sq})", f"(-{dh})"
            if expr.op == "abs":
                return sig, f"ABS({sq})", f"Math.abs({dh})"
            if expr.op == "not":
                return sig, f"NOT ({sq})", f"!({dh})"

        # Fallback for unsupported nodes in basis extraction (e.g. Strings, Conditionals)
        raise NotImplementedError(f"Node type {type(expr)} not supported in dynamic basis extraction yet.")


    def evaluate_component(self, component_type: int, params: List[float], knot_values: List[float]) -> float:
        """
        Evaluate a specific basis function component given its parameters and knot values.
        Provides a fast, purely Python-native execution path.
        """
        bf = None
        for registered_bf in self.registry.values():
            if registered_bf.component_type == component_type:
                bf = registered_bf
                break
                
        if bf is None:
            raise ValueError(f"Unknown component type: {component_type}")
            
        if len(params) != bf.num_params:
            raise ValueError(f"Expected {bf.num_params} params, got {len(params)} for type {component_type}")
            
        if len(knot_values) != bf.num_vars:
            raise ValueError(f"Expected {bf.num_vars} knots, got {len(knot_values)} for type {component_type}")
            
        # The fastest way to execute this natively in Python without re-parsing 
        # is to take the generated sql/dh template or a generic python template,
        # compile it once to a Python code object, and evaluate it.
        if not hasattr(bf, "_compiled_py"):
            py_code = bf.dh_template.replace("Math.pow", "pow").replace("Math.", "")
            bf._compiled_py = compile(py_code, f"<basis_{component_type}>", "eval")
            
        # 2. Build local namespace map
        locals_map = {}
        for i, p_val in enumerate(params):
            locals_map[f"p{i+1}"] = p_val
            
        for i, x_val in enumerate(knot_values):
            locals_map[f"X{i+1}"] = x_val
            
        # 3. Fast built-in execution
        return eval(bf._compiled_py, {"pow": pow, "abs": abs}, locals_map)


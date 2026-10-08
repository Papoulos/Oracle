import ast
import math
import random
import operator
import functools
from typing import Any, Annotated
from pydantic import AfterValidator, Field

class ExprError(ValueError):
    """Raised for any validation or evaluation error in the expression."""
    pass

class CompiledExpr:
    def __init__(self, tree: ast.Expression, source: str):
        self.tree = tree
        self.source = source

ALLOWED_FUNCTIONS = {'min', 'max', 'floor', 'ceil', 'abs', 'clamp', 'd'}

MAX_LEN = 300
MAX_NODES = 200
MAX_DEPTH = 20
MAX_VALUE = 1_000_000
MAX_INTERMEDIATE = 10**12

def _validate_node(node: ast.AST, depth: int, node_count: list[int]):
    if depth > MAX_DEPTH:
        raise ExprError("Expression too deep")
    node_count[0] += 1
    if node_count[0] > MAX_NODES:
        raise ExprError("Expression too complex")

    if isinstance(node, ast.Expression):
        _validate_node(node.body, depth + 1, node_count)
    elif isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            if abs(node.value) > MAX_VALUE:
                raise ExprError(f"Constant out of bounds: {node.value}")
        elif isinstance(node.value, bool):
            pass
        else:
            raise ExprError(f"Invalid constant type: {type(node.value)}")
    elif isinstance(node, ast.Name):
        pass # We check bare names in a separate pass
    elif isinstance(node, ast.Attribute):
        if node.attr.startswith("_"):
            raise ExprError("Attributes starting with _ are forbidden")
        _validate_node(node.value, depth + 1, node_count)
    elif isinstance(node, ast.BinOp):
        if not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod)):
            raise ExprError(f"Unsupported binary operator: {type(node.op)}")
        _validate_node(node.left, depth + 1, node_count)
        _validate_node(node.right, depth + 1, node_count)
    elif isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, (ast.UAdd, ast.USub, ast.Not)):
            raise ExprError(f"Unsupported unary operator: {type(node.op)}")
        _validate_node(node.operand, depth + 1, node_count)
    elif isinstance(node, ast.BoolOp):
        if not isinstance(node.op, (ast.And, ast.Or)):
            raise ExprError(f"Unsupported boolean operator: {type(node.op)}")
        for val in node.values:
            _validate_node(val, depth + 1, node_count)
    elif isinstance(node, ast.Compare):
        for op in node.ops:
            if not isinstance(op, (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)):
                raise ExprError(f"Unsupported comparison operator: {type(op)}")
        _validate_node(node.left, depth + 1, node_count)
        for comp in node.comparators:
            _validate_node(comp, depth + 1, node_count)
    elif isinstance(node, ast.IfExp):
        _validate_node(node.test, depth + 1, node_count)
        _validate_node(node.body, depth + 1, node_count)
        _validate_node(node.orelse, depth + 1, node_count)
    elif isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in ALLOWED_FUNCTIONS:
            raise ExprError("Invalid function call")
        if node.keywords:
            raise ExprError("Keyword arguments not allowed")
        for arg in node.args:
            _validate_node(arg, depth + 1, node_count)
    elif isinstance(node, ast.Load):
        pass
    else:
        raise ExprError(f"Unsupported node type: {type(node)}")


class CheckBareNames(ast.NodeVisitor):
    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id in ALLOWED_FUNCTIONS:
            for arg in node.args:
                self.visit(arg)
            for kw in node.keywords:
                self.visit(kw)
        else:
            self.generic_visit(node)

    def visit_Attribute(self, node):
        pass

    def visit_Name(self, node):
        if node.id in ALLOWED_FUNCTIONS:
            raise ExprError(f"Function {node.id} used as a bare name")
        if node.id.startswith("_"):
            raise ExprError("Variables starting with _ are forbidden")
        self.generic_visit(node)


@functools.lru_cache(maxsize=1024)
def compile_expr(expr: str) -> CompiledExpr:
    if len(expr) > MAX_LEN:
        raise ExprError("Expression too long")

    max_paren_depth = 0
    curr_paren_depth = 0
    for char in expr:
        if char == '(':
            curr_paren_depth += 1
            max_paren_depth = max(max_paren_depth, curr_paren_depth)
        elif char == ')':
            curr_paren_depth -= 1

    if max_paren_depth > MAX_DEPTH:
        raise ExprError("Expression too deep (parentheses)")

    try:
        tree = ast.parse(expr, mode='eval')
    except (SyntaxError, RecursionError, MemoryError) as e:
        raise ExprError(f"Syntax error: {e}")

    _validate_node(tree, 0, [0])

    visitor = CheckBareNames()
    visitor.visit(tree)

    return CompiledExpr(tree, expr)

def _get_path(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Attribute):
        return f"{_get_path(node.value)}.{node.attr}"
    else:
        raise ExprError("Invalid path")

def referenced_variables(expr: str) -> set[str]:
    compiled = compile_expr(expr)
    vars = set()

    def walk_vars(n):
        if isinstance(n, ast.Name):
            if n.id not in ALLOWED_FUNCTIONS:
                vars.add(n.id)
        elif isinstance(n, ast.Attribute):
            try:
                vars.add(_get_path(n))
            except ExprError:
                pass
        elif isinstance(n, ast.Call):
            for arg in n.args:
                walk_vars(arg)
        elif isinstance(n, ast.BinOp):
            walk_vars(n.left)
            walk_vars(n.right)
        elif isinstance(n, ast.UnaryOp):
            walk_vars(n.operand)
        elif isinstance(n, ast.BoolOp):
            for v in n.values:
                walk_vars(v)
        elif isinstance(n, ast.Compare):
            walk_vars(n.left)
            for c in n.comparators:
                walk_vars(c)
        elif isinstance(n, ast.IfExp):
            walk_vars(n.test)
            walk_vars(n.body)
            walk_vars(n.orelse)
        elif isinstance(n, ast.Expression):
            walk_vars(n.body)

    walk_vars(compiled.tree)
    return vars

def _evaluate_node(node: ast.AST, ctx: dict, rng: random.Random) -> Any:
    if isinstance(node, ast.Expression):
        return _evaluate_node(node.body, ctx, rng)
    elif isinstance(node, ast.Constant):
        return node.value
    elif isinstance(node, (ast.Name, ast.Attribute)):
        path = _get_path(node)
        parts = path.split('.')
        curr = ctx
        for part in parts:
            if not isinstance(curr, dict) or part not in curr:
                raise ExprError(f"variable inconnue : {path}")
            curr = curr[part]
        if not isinstance(curr, (int, float, bool)):
            raise ExprError(f"variable non numérique : {path}")
        return curr
    elif isinstance(node, ast.BinOp):
        left = _evaluate_node(node.left, ctx, rng)
        right = _evaluate_node(node.right, ctx, rng)

        if isinstance(node.op, ast.Add):
            res = left + right
        elif isinstance(node.op, ast.Sub):
            res = left - right
        elif isinstance(node.op, ast.Mult):
            res = left * right
        elif isinstance(node.op, ast.Div):
            if right == 0:
                raise ExprError("Division par zéro")
            res = left / right
        elif isinstance(node.op, ast.FloorDiv):
            if right == 0:
                raise ExprError("Division par zéro")
            res = left // right
        elif isinstance(node.op, ast.Mod):
            if right == 0:
                raise ExprError("Division par zéro")
            res = left % right
        else:
            raise ExprError("Unsupported operator")

        if isinstance(res, (int, float)) and abs(res) > MAX_INTERMEDIATE:
            raise ExprError("valeur hors limites")
        return res
    elif isinstance(node, ast.UnaryOp):
        operand = _evaluate_node(node.operand, ctx, rng)
        if isinstance(node.op, ast.UAdd):
            return +operand
        elif isinstance(node.op, ast.USub):
            return -operand
        elif isinstance(node.op, ast.Not):
            return not operand
        else:
            raise ExprError("Unsupported operator")
    elif isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            for val in node.values:
                res = _evaluate_node(val, ctx, rng)
                if not res:
                    return False
            return True
        elif isinstance(node.op, ast.Or):
            for val in node.values:
                res = _evaluate_node(val, ctx, rng)
                if res:
                    return True
            return False
        else:
            raise ExprError("Unsupported operator")
    elif isinstance(node, ast.Compare):
        left = _evaluate_node(node.left, ctx, rng)
        for op, comp in zip(node.ops, node.comparators):
            right = _evaluate_node(comp, ctx, rng)
            if isinstance(op, ast.Eq):
                res = left == right
            elif isinstance(op, ast.NotEq):
                res = left != right
            elif isinstance(op, ast.Lt):
                res = left < right
            elif isinstance(op, ast.LtE):
                res = left <= right
            elif isinstance(op, ast.Gt):
                res = left > right
            elif isinstance(op, ast.GtE):
                res = left >= right
            else:
                raise ExprError("Unsupported operator")
            if not res:
                return False
            left = right
        return True
    elif isinstance(node, ast.IfExp):
        test = _evaluate_node(node.test, ctx, rng)
        if test:
            return _evaluate_node(node.body, ctx, rng)
        else:
            return _evaluate_node(node.orelse, ctx, rng)
    elif isinstance(node, ast.Call):
        func_name = node.func.id
        args = [_evaluate_node(arg, ctx, rng) for arg in node.args]

        if func_name == 'min':
            if not args:
                raise ExprError("min() requires at least one argument")
            return min(args)
        elif func_name == 'max':
            if not args:
                raise ExprError("max() requires at least one argument")
            return max(args)
        elif func_name == 'floor':
            if len(args) != 1:
                raise ExprError("floor() requires exactly one argument")
            return math.floor(args[0])
        elif func_name == 'ceil':
            if len(args) != 1:
                raise ExprError("ceil() requires exactly one argument")
            return math.ceil(args[0])
        elif func_name == 'abs':
            if len(args) != 1:
                raise ExprError("abs() requires exactly one argument")
            return abs(args[0])
        elif func_name == 'clamp':
            if len(args) != 3:
                raise ExprError("clamp() requires exactly three arguments")
            x, lo, hi = args
            return max(lo, min(x, hi))
        elif func_name == 'd':
            if len(args) != 2:
                raise ExprError("d() requires exactly two arguments")
            n, faces = args
            if type(n) is bool or type(faces) is bool:
                 raise ExprError("d() arguments must be integers")
            if not (isinstance(n, int) or (isinstance(n, float) and n.is_integer())):
                raise ExprError("d() arguments must be integers")
            if not (isinstance(faces, int) or (isinstance(faces, float) and faces.is_integer())):
                raise ExprError("d() arguments must be integers")
            n = int(n)
            faces = int(faces)
            if not (1 <= n <= 100):
                raise ExprError("d() n out of bounds (1-100)")
            if not (2 <= faces <= 1000):
                raise ExprError("d() faces out of bounds (2-1000)")

            return sum(rng.randint(1, faces) for _ in range(n))
        else:
            raise ExprError(f"Unknown function: {func_name}")
    else:
        raise ExprError(f"Unsupported node: {type(node)}")

def evaluate(expr_or_compiled: str | CompiledExpr, ctx: dict, rng: random.Random | None = None) -> int | float | bool:
    if isinstance(expr_or_compiled, str):
        compiled = compile_expr(expr_or_compiled)
    else:
        compiled = expr_or_compiled

    if rng is None:
        rng = random.Random()

    try:
        return _evaluate_node(compiled.tree, ctx, rng)
    except ExprError:
        raise
    except Exception as e:
        raise ExprError(f"Error evaluating {compiled.source}: {e}")

def evaluate_int(expr_or_compiled: str | CompiledExpr, ctx: dict, rng: random.Random | None = None) -> int:
    res = evaluate(expr_or_compiled, ctx, rng)
    if isinstance(res, bool):
        raise ExprError("Result is boolean, expected int")
    if not isinstance(res, (int, float)):
        raise ExprError("Result is not numeric")
    if isinstance(res, float) and not res.is_integer():
        raise ExprError("Result is not an exact integer")
    return int(res)

def _check_formula(v: str) -> str:
    compile_expr(v)
    return v

Formula = Annotated[str, Field(description="Formule arithmétique (langage fermé de mechanics.expr)"), AfterValidator(_check_formula)]

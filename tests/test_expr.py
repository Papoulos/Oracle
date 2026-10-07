import pytest
import random
import ast

from mechanics.expr import (
    compile_expr, evaluate, evaluate_int, referenced_variables,
    ExprError, Formula, MAX_VALUE, MAX_INTERMEDIATE
)
from pydantic import BaseModel, ValidationError

def test_cypher_cost():
    expr = "max(0, 3 + 2*(levels-1) + (levels if impaired else 0) - edge) if levels > 0 else 0"

    def eval_cypher(levels, impaired, edge):
        ctx = {"levels": levels, "impaired": impaired, "edge": edge}
        return evaluate_int(expr, ctx)

    assert eval_cypher(1, False, 0) == 3
    assert eval_cypher(2, False, 0) == 5
    assert eval_cypher(3, False, 0) == 7

    assert eval_cypher(1, False, 1) == 2
    assert eval_cypher(2, False, 1) == 4
    assert eval_cypher(3, False, 1) == 6

    assert eval_cypher(1, False, 3) == 0
    assert eval_cypher(2, False, 3) == 2
    assert eval_cypher(3, False, 3) == 4

    assert eval_cypher(1, True, 0) == 4
    assert eval_cypher(2, True, 0) == 7
    assert eval_cypher(3, True, 0) == 10

    assert eval_cypher(0, False, 0) == 0

def test_recovery_rules():
    expr = "d(1, 6) + character.tier"
    ctx = {"character": {"tier": 2}}
    rng = random.Random(42)

    results = [evaluate_int(expr, ctx, rng) for _ in range(20000)]
    assert min(results) == 3
    assert max(results) == 8
    avg = sum(results) / len(results)
    assert 5.4 <= avg <= 5.6

def test_dnd_logic():
    expr = "max(1, floor(hit_dice.max / 2))"
    assert evaluate_int(expr, {"hit_dice": {"max": 7}}) == 3
    assert evaluate_int(expr, {"hit_dice": {"max": 1}}) == 1

def test_dangerous_expressions():
    dangerous = [
        "d(1, 6) if 1 else exit()",
        "__import__('os')",
        "().__class__",
        "[x for x in range(9)]",
        "2 ** 99",
        "levels.__class__",
        "lambda: 1",
        "'abc'",
        "levels[0]",
        "x != 0 and 10 / x > 1" + " " * 400, # > 300 chars
        "10 / 0", # division by zero during evaluation
        "unknown_var",
        "max", # bare name
        "eval('1')",
        "exec('1')",
        "compile('1', '', 'eval')",
        "(((((((((((((((((((((1)))))))))))))))))))))" # deep
    ]

    ctx = {"levels": 1, "x": 0}
    for expr in dangerous:
        with pytest.raises(ExprError):
            evaluate(expr, ctx)

def test_referenced_variables():
    assert referenced_variables("a + b.c + max(d.e, 1)") == {"a", "b.c", "d.e"}
    assert "max" not in referenced_variables("max(1, 2)")
    assert referenced_variables("x != 0 and 10 / x > 1") == {"x"}

def test_evaluate_int_precision():
    assert evaluate_int("3.0", {}) == 3
    with pytest.raises(ExprError):
        evaluate_int("3.5", {})
    with pytest.raises(ExprError):
        evaluate_int("True", {})

def test_determinism():
    expr = "d(10, 6)"
    res1 = evaluate_int(expr, {}, random.Random(42))
    res2 = evaluate_int(expr, {}, random.Random(42))
    assert res1 == res2

def test_short_circuit():
    assert evaluate("x != 0 and 10 / x > 1", {"x": 0}) == False
    assert evaluate("x == 0 or 10 / x > 1", {"x": 0}) == True
    assert evaluate("1 if True else 10 / 0", {}) == 1
    assert evaluate("10 / 0 if False else 1", {}) == 1

def test_pydantic_formula():
    class Model(BaseModel):
        f: Formula

    assert Model(f="1+1").f == "1+1"

    with pytest.raises(ValidationError):
        Model(f="1 + 'a'")

def test_no_forbidden_functions_in_file():
    with open('mechanics/expr.py', 'r') as f:
        content = f.read()

    tree = ast.parse(content)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in ['eval', 'exec', 'compile', '__import__']

def test_non_numeric_vars():
    with pytest.raises(ExprError, match="variable non numérique : a"):
        evaluate("a + 1", {"a": "string"})

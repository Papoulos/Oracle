from mechanics.expr import evaluate, ExprError

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
    try:
        evaluate(expr, ctx)
        print(f"FAILED to raise: {expr}")
    except Exception as e:
        pass

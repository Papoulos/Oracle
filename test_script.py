from mechanics.expr import compile_expr, ExprError

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

for expr in dangerous:
    try:
        compile_expr(expr)
        print(f"FAILED to raise: {expr}")
    except Exception as e:
        pass

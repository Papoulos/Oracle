from mechanics.expr import compile_expr, ExprError
print("Testing 10/0")
compile_expr("10/0") # Should succeed compilation, fail evaluation. Wait, the test checks evaluate()

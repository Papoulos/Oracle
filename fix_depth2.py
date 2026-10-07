import ast

expr = "(((((((((((((((((((((1)))))))))))))))))))))"
tree = ast.parse(expr, mode='eval')
print(ast.dump(tree))

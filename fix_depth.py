import ast

def _get_depth(node):
    if hasattr(node, '_depth'):
         return node._depth

    if isinstance(node, ast.Expression):
        node._depth = _get_depth(node.body) + 1
        return node._depth

    max_d = 0
    for child in ast.iter_child_nodes(node):
        max_d = max(max_d, _get_depth(child))
    node._depth = max_d + 1
    return node._depth

expr = "(((((((((((((((((((((1)))))))))))))))))))))"
tree = ast.parse(expr, mode='eval')
print("Depth:", _get_depth(tree))

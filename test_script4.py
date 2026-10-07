import ast

class CheckBareNames(ast.NodeVisitor):
    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id in {"max", "min", "d"}:
            for arg in node.args:
                self.visit(arg)
            for kw in node.keywords:
                self.visit(kw)
        else:
            self.generic_visit(node)

    def visit_Attribute(self, node):
        # Stop generic visit if node.value is a Name, because if it is 'd', it's 'd.e', which means 'd' is not a function call, but a root.
        # So we skip checking Name if it is a value in Attribute, because we know it's a variable path.
        pass

    def visit_Name(self, node):
        if node.id in {"max", "min", "d"}:
            raise Exception(f"Function {node.id} used as a bare name")
        self.generic_visit(node)

tree = ast.parse("a + b.c + max(d.e, 1)")
CheckBareNames().visit(tree)
print("SUCCESS")

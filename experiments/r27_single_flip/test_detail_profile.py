"""Verify instrumentation preserves all prediction expressions, without torch."""
import ast
from pathlib import Path
import unittest


class StripInstrumentation(ast.NodeTransformer):
    def visit_FunctionDef(self,node):
        if node.name=='mark':return None
        node.name='predict_view'
        return self.generic_visit(node)
    def visit_Expr(self,node):
        if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='mark':return None
        return self.generic_visit(node)
    def visit_Assign(self,node):
        target=node.targets[0]
        if isinstance(target,ast.Name):
            if target.id in ('segments','last'):return None
            if target.id=='result':return ast.copy_location(ast.Return(value=node.value),node)
        if isinstance(target,ast.Subscript) and isinstance(target.value,ast.Name) and target.value.id=='result':return None
        return self.generic_visit(node)
    def visit_Return(self,node):
        if isinstance(node.value,ast.Name) and node.value.id=='result':return None
        return self.generic_visit(node)


class ProfileTests(unittest.TestCase):
    def test_prediction_ast_is_unchanged_after_removing_marks(self):
        root=Path(__file__).resolve().parent
        def function(name):
            tree=ast.parse((root/name).read_text())
            return next(n for n in tree.body if isinstance(n,ast.FunctionDef))
        original=function('detail_adapter.py')
        profiled=StripInstrumentation().visit(function('profile_detail_adapter.py'))
        self.assertEqual(ast.dump(original),ast.dump(profiled))


if __name__=='__main__':unittest.main()

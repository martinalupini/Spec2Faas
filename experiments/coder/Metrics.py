from radon.visitors import ComplexityVisitor
import ast
from cognitive_complexity.api import get_cognitive_complexity


"""
To compute Cyclomatic Complexity
"""
def compute_CC(code):
    try:
        visitor = ComplexityVisitor.from_code(code)
        return visitor.total_complexity
    except IndentationError:
        return 0
    except SyntaxError:
        return 0


"""
To compute Cognitive Complexity
"""
def compute_CoG(code: str):
    try:
        tree = ast.parse(code)

        func_node = None
        # Search for the first function definition in the tree
        for node in tree.body:
            # Check if the node is a function
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_node = node
                break

        if func_node:
            return get_cognitive_complexity(func_node)
    except IndentationError:
        return 0
    except SyntaxError:
        return 0
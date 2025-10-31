from radon.visitors import ComplexityVisitor
import ast
from cognitive_complexity.api import get_cognitive_complexity




def compute_CC(code: str):

    visitor = ComplexityVisitor.from_code(code)

    results = visitor.functions[0]

    return results.complexity


def compute_CoG(code: str):
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
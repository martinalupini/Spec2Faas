from radon.visitors import ComplexityVisitor


def compute_CC(code: str):

    visitor = ComplexityVisitor.from_code(code)

    results = visitor.functions[0]

    return results.complexity
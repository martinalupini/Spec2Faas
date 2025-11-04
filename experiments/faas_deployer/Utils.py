import re
import ast
import json
from typing import Tuple, List, Any
import pandas as pd

def extract_param_names(testo: str, entry_point: str) ->List[str]:
    pattern = re.compile(r"def\s+" + re.escape(entry_point) + r"\s*\((.*?)\):")

    match = pattern.search(testo)

    if not match:
        return None

    param_strings = match.group(1)

    if not param_strings.strip():
        return []

    param_list = [p.strip() for p in param_strings.split(',')]

    param_names = []
    for param in param_list:
        name = param.split(':')[0].strip()
        param_names.append(name)

    return param_names


def extract_param_values(code: str) -> Any:

    tree = ast.parse(code)

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            # Check if the node is called 'input'
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id == 'inputs':

                # Get the variable value
                param_values = ast.literal_eval(node.value)

                # Return the first element of the list
                if isinstance(param_values, list) and len(param_values) > 0:
                    return param_values[0]
                else:
                    raise ValueError("The 'inputs' is empty or it's not a list")


def create_json(names: list, values: list, filename: str):

    if len(names) != len(values):
        print(names, values)
        raise ValueError("The number of values and number of names are not equal")

    data = dict(zip(names, values))

    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)


func= "from typing import List def has_close_elements(numbers: List[float], threshold: float) -> bool:"

#extract_param_names(func)

from pathlib import Path
import logging
import os
import re
from autogen_core import TRACE_LOGGER_NAME
import yaml
import json
import uuid
from typing import List
from autogen_core import FunctionCall



def load_env_variables():

    try:
        # load environment variables from .env file (requires `pip install python-dotenv`)
        from dotenv import load_dotenv

        env_path = Path(__file__).resolve().parents[1] / ".env"

        load_dotenv(dotenv_path=env_path)

    except ImportError:
        pass


def set_logging_config(log_path):

    logging.basicConfig(level=logging.WARNING)
    logger = logging.getLogger(TRACE_LOGGER_NAME)
    logger.setLevel(logging.DEBUG)

    # File's handler
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s")
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    # Forcing reconfiguration of logging
    logging.basicConfig(level=logging.DEBUG, handlers=[file_handler], force=True)

    logging.getLogger("autogen_core").setLevel(logging.DEBUG)
    logging.getLogger(TRACE_LOGGER_NAME).setLevel(logging.DEBUG)

    logging.getLogger(TRACE_LOGGER_NAME).propagate = True


def get_config_data(path):

    with open(path, 'r') as file:
        config = yaml.safe_load(file)
        return config['llm']


def get_func_call_from_json(json_string) -> List[FunctionCall]:
    # Removing data before and after {}
    cleaned_data = clean_string(json_string)
    # Putting escape characters before special characters
    fixed_string = fix_json_string_with_code(cleaned_data)
    # Transforming string into json
    data = json.loads(fixed_string)

    function_name = data["name"]
    arguments_dict = data["arguments"]

    # Transform into string again according to the format of FunctionCall
    arguments_as_string = json.dumps(arguments_dict)

    function_id = str(uuid.uuid4())

    function_call_instance = FunctionCall(
        id=function_id,
        name=function_name,
        arguments=arguments_as_string
    )

    return [function_call_instance]


def clean_string(input_string):

    start_index = input_string.find('{')
    end_index = input_string.rfind('}')

    return input_string[start_index: end_index + 1]


def fix_json_string_with_code(malformed_json: str) -> str:
    try:
        # First, we try to load it directly. If it works, there's nothing to fix.
        json.loads(malformed_json)
        return malformed_json
    except json.JSONDecodeError:
        # If it fails, we proceed with the manual fix.
        pass

    # Find the start of the "code" field's value
    start_marker = '"code": "'
    start_index = malformed_json.find(start_marker)

    if start_index == -1:
        start_marker = '"code":"'
        start_index = malformed_json.find(start_marker)

    if start_index == -1:
        # If there's no "code" field, the error lies elsewhere.
        raise ValueError("The JSON string is malformed, but not due to the 'code' field.")

    content_start_index = start_index + len(start_marker)

    # Now, we intelligently find the end of the code's value string.
    current_pos = content_start_index
    content_end_index = -1

    while True:
        # Find the next double quote
        end_quote_index = malformed_json.find('"', current_pos)

        if end_quote_index == -1:
            # If no closing quote is found, the JSON is severely corrupted.
            raise ValueError("Corrupted JSON: no closing quote found for 'code' value.")

        # Check the first non-whitespace character after this quote
        next_char_index = end_quote_index + 1
        while next_char_index < len(malformed_json) and malformed_json[next_char_index].isspace():
            next_char_index += 1

        # If we're at the end of the string, or if the next character is a comma or a closing brace,
        # we have found the TRUE closing quote.
        if next_char_index == len(malformed_json) or malformed_json[next_char_index] in [',', '}']:
            content_end_index = end_quote_index
            break
        else:
            # Otherwise, this was a quote inside the code string. We continue our search.
            current_pos = end_quote_index + 1

    # Now we have the three parts of the string
    part_before = malformed_json[:content_start_index]
    code_string = malformed_json[content_start_index:content_end_index]
    part_after = malformed_json[content_end_index:]

    # Perform escaping only on the code part
    escaped_code = code_string.replace('\\', '\\\\')
    escaped_code = escaped_code.replace('"', '\\"')
    escaped_code = escaped_code.replace('\n', '\\n')
    escaped_code = escaped_code.replace('\r', '\\r')
    escaped_code = escaped_code.replace('\t', '\\t')

    # Reassemble the corrected JSON string
    return f"{part_before}{escaped_code}{part_after}"

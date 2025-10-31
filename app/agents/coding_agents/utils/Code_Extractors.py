import re
from typing import List
from autogen_core.code_executor import CodeBlock
import autopep8

def extract_markdown_code_blocks(markdown_text: str) -> List[CodeBlock]:
    pattern = re.compile(r"```(?:\s*([\w\+\-]+))?\n([\s\S]*?)```")
    matches = pattern.findall(markdown_text)
    code_blocks: List[CodeBlock] = []
    for match in matches:
        language = match[0].strip() if match[0] else ""
        code_content = match[1]
        code_blocks.append(CodeBlock(code=code_content, language=language))
    return code_blocks


def extract_signature(description: str) -> str:
    match = re.search(r"^\s*def.*:$", description, re.MULTILINE)

    if match:
        signature = match.group(0).strip()
    else:
        signature = ""
    return signature



def fix_indent(text: str) -> str:
    return autopep8.fix_code(text)

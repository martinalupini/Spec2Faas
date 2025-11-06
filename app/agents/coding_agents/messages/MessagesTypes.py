from dataclasses import dataclass

@dataclass
class Message:
    content: str
    type: str


@dataclass
class CodeMessage:
    specification: str
    function_signature: str
    code: str
    tests: str
    sender: str

@dataclass
class DeployMessage:
    code:str


@dataclass
class DebugMessage:
    specification: str
    code:str
    error_message: str

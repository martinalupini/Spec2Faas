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
    test: str

@dataclass
class DeployMessage:
    """TODO"""
    code:str

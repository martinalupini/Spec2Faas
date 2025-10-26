from dataclasses import dataclass


@dataclass
class Message:
    content: str
    type: str

@dataclass
class CodeWritingRequest:
    content: str

@dataclass
class CodeWritingMessage:
    specification: str

@dataclass
class FinalCodeWritingResult:
    code: str
    type: str

@dataclass
class DeployMessage:
    code:str

@dataclass
class CodeExecutorFinalResult:
    code: str
    type: str


@dataclass
class CodeMessage:
    specification: str
    function_signature: str


@dataclass
class DebugMessage:
    specification: str
    code:str
    tests: str
    error_message: str

@dataclass
class ExecuteCodeRequest:
    specification: str
    code: str
    tests: str
    sender: str


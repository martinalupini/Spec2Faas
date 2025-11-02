from dataclasses import dataclass
from autogen_core import CancellationToken

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

@dataclass
class TestCodeMessage:
    specification: str
    function_signature: str
    prompt: bool

@dataclass
class TestCodeResult:
    content: str
    time: float
    tokens: float
    ctx: CancellationToken

@dataclass
class TestDeployMessage:
    code:str

@dataclass
class TestDeployResult:
    result: str
    time: float
    tokens: float
    ctx: CancellationToken
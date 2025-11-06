from dataclasses import dataclass
from autogen_core import CancellationToken

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

@dataclass
class TestExecCodeMessage:
    specification: str
    function_signature: str
    code: str
    tests: str

@dataclass
class TestExecCodeResult:
    final_function: str
    passed: bool
    time: float
    tokens: float
    attempts: int

@dataclass
class TestDebugMessage:
    specification: str
    code: str
    error_message: str


@dataclass
class TestDebugResult:
    code: str
    tokens: float
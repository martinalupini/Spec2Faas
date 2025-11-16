from dataclasses import dataclass
from autogen_core import CancellationToken, AgentId

@dataclass
class TestCodeMessage:
    specification: str
    function_signature: str
    prompt: bool
    system: bool = False
    time: dict = None
    tokens: dict = None

@dataclass
class TestCodeResult:
    content: str
    time: float | dict
    tokens: float | dict
    ctx: CancellationToken

@dataclass
class TestDeployMessage:
    code:str
    system: bool = False
    time: dict = None
    tokens: dict = None

@dataclass
class TestDeployResult:
    result: str
    time: float | dict
    tokens: float | dict
    ctx: CancellationToken

@dataclass
class TestExecCodeMessage:
    specification: str
    function_signature: str
    code: str
    tests: str

@dataclass
class TestExecCodeSystemMessage:
    specification: str
    function_signature: str
    code: str
    tests: str
    id: str = ""
    time: dict = None
    tokens: dict = None

@dataclass
class TestExecCodeResult:
    final_function: str
    passed: bool
    time: float | dict
    tokens: float | dict
    attempts: int

@dataclass
class TestDebugMessage:
    specification: str
    code: str
    error_message: str
    new_chat: bool
    system: bool = False
    time: dict = None
    tokens: dict = None


@dataclass
class TestDebugResult:
    code: str
    tokens: float | dict


@dataclass
class FinalTestResult:
    time: dict
    tokens: dict
    generated_function: str
    generated_test: str
    corrected_function: str
    generated: bool
    deployed: bool
    ctx: CancellationToken

@dataclass
class TestMessage:
    content: str
    time: dict = None
    tokens: dict = None
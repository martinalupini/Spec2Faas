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
    system: bool = False

@dataclass
class TestExecCodeResult:
    final_function: str
    passed: bool
    time: float
    tokens: float
    attempts: int
    tokens_debugger:float = 0
    time_debugger: float = 0

@dataclass
class TestDebugMessage:
    specification: str
    code: str
    error_message: str
    new_chat: bool


@dataclass
class TestDebugResult:
    code: str
    tokens: float

@dataclass
class TestMessage:
    content:str

@dataclass
class TestMessageResult:
    content: str
    time : float
    tokens: float


@dataclass
class TestSystemMessage:
    tokens: dict
    time: dict
    # From Assistant
    prompt: str = ""
    # From Entry Point
    signature: str = ""
    # From Coder
    original_func: str = ""
    code: str = ""
    new_chat: bool = False
    # From test designer
    tests: str = ""
    tests_str: str = ""
    sender: str = ""
    # From debugger and test executor
    final_func: str = ""
    code_final_func: str = ""
    attempts: int = 0
    generated: bool = False
    # From deployer
    result_deployment: str = ""
    deployed: bool = False
    type: str = ""
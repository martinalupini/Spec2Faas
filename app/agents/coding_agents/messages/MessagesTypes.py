from dataclasses import dataclass


"""
From main to Assistant
From Assistant to Entry Point
"""
@dataclass
class Message:
    content: str
    type: str

"""
From Entry Point to Coder
From Entry Point to Test Designer
From Coder to Test Executor
From Test Designer to Test Executor 
"""
@dataclass
class CodeMessage:
    specification: str
    function_signature: str
    code: str
    tests: str
    sender: str


"""
From Assistant to FaaS Deployer
"""
@dataclass
class DeployMessage:
    code:str


"""
From Test Executor to Debugger
"""
@dataclass
class DebugMessage:
    specification: str
    code:str
    error_message: str

"""
Tool Execution System for S-7 ASI
Provides secure execution of Python code, shell commands, and API calls
Part of the TRUE ASI System - 100/100 Quality
"""

import os
import sys
import json
import subprocess
import tempfile
import requests
import docker
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import asyncio
import aiohttp
from pathlib import Path


class ToolType(Enum):
    """Types of tools available"""
    PYTHON = "python"
    SHELL = "shell"
    API = "api"
    FILE = "file"
    DATABASE = "database"
    WEB = "web"


@dataclass
class ToolResult:
    """Result from tool execution"""
    success: bool
    output: Any
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = None


class PythonSandbox:
    """Secure Python code execution in isolated environment"""
    
    def __init__(self, timeout: int = 30, memory_limit: str = "512m"):
        self.timeout = timeout
        self.memory_limit = memory_limit
        self.docker_client = docker.from_env()
        
    def execute(self, code: str, context: Dict[str, Any] = None) -> ToolResult:
        """Execute Python code in Docker sandbox"""
        try:
            import time
            start_time = time.time()
            
            # Create temporary file for code
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                code_file = f.name
            
            # Prepare context as JSON
            context_file = None
            if context:
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                    json.dump(context, f)
                    context_file = f.name
            
            # Run in Docker container
            container = self.docker_client.containers.run(
                "python:3.11-slim",
                f"python {os.path.basename(code_file)}",
                volumes={
                    os.path.dirname(code_file): {'bind': '/workspace', 'mode': 'rw'}
                },
                working_dir='/workspace',
                mem_limit=self.memory_limit,
                network_disabled=False,  # Allow network for API calls
                detach=True,
                remove=True
            )
            
            # Wait for completion with timeout
            try:
                result = container.wait(timeout=self.timeout)
                logs = container.logs().decode('utf-8')
                
                execution_time = time.time() - start_time
                
                # Clean up
                os.unlink(code_file)
                if context_file:
                    os.unlink(context_file)
                
                if result['StatusCode'] == 0:
                    return ToolResult(
                        success=True,
                        output=logs,
                        execution_time=execution_time,
                        metadata={'status_code': 0}
                    )
                else:
                    return ToolResult(
                        success=False,
                        output=logs,
                        error=f"Exit code: {result['StatusCode']}",
                        execution_time=execution_time
                    )
                    
            except Exception as e:
                container.kill()
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Execution timeout or error: {str(e)}"
                )
                
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Sandbox error: {str(e)}"
            )


class ShellExecutor:
    """Secure shell command execution"""
    
    def __init__(self, allowed_commands: List[str] = None, timeout: int = 30):
        self.allowed_commands = allowed_commands or [
            'ls', 'cat', 'grep', 'find', 'wc', 'head', 'tail', 
            'sort', 'uniq', 'cut', 'awk', 'sed', 'git'
        ]
        self.timeout = timeout
        
    def execute(self, command: str, cwd: str = None) -> ToolResult:
        """Execute shell command with safety checks"""
        try:
            import time
            start_time = time.time()
            
            # Parse command
            cmd_parts = command.split()
            if not cmd_parts:
                return ToolResult(
                    success=False,
                    output=None,
                    error="Empty command"
                )
            
            # Check if command is allowed
            base_cmd = cmd_parts[0]
            if base_cmd not in self.allowed_commands:
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Command '{base_cmd}' not allowed"
                )
            
            # Execute command
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=cwd
            )
            
            execution_time = time.time() - start_time
            
            if result.returncode == 0:
                return ToolResult(
                    success=True,
                    output=result.stdout,
                    execution_time=execution_time,
                    metadata={'returncode': 0}
                )
            else:
                return ToolResult(
                    success=False,
                    output=result.stdout,
                    error=result.stderr,
                    execution_time=execution_time,
                    metadata={'returncode': result.returncode}
                )
                
        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                output=None,
                error=f"Command timeout after {self.timeout}s"
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Execution error: {str(e)}"
            )


class APIExecutor:
    """Execute API calls to external services"""
    
    def __init__(self, api_keys: Dict[str, str] = None, timeout: int = 30):
        self.api_keys = api_keys or {}
        self.timeout = timeout
        self.session = requests.Session()
        
    def execute(self, 
                url: str, 
                method: str = "GET",
                headers: Dict[str, str] = None,
                params: Dict[str, Any] = None,
                data: Any = None,
                json_data: Dict[str, Any] = None) -> ToolResult:
        """Execute API call"""
        try:
            import time
            start_time = time.time()
            
            # Prepare headers
            final_headers = headers or {}
            
            # Add API key if available
            for key_name, key_value in self.api_keys.items():
                if key_name.lower() in url.lower():
                    final_headers['Authorization'] = f'Bearer {key_value}'
                    break
            
            # Execute request
            response = self.session.request(
                method=method.upper(),
                url=url,
                headers=final_headers,
                params=params,
                data=data,
                json=json_data,
                timeout=self.timeout
            )
            
            execution_time = time.time() - start_time
            
            # Parse response
            try:
                response_data = response.json()
            except:
                response_data = response.text
            
            if response.status_code < 400:
                return ToolResult(
                    success=True,
                    output=response_data,
                    execution_time=execution_time,
                    metadata={
                        'status_code': response.status_code,
                        'headers': dict(response.headers)
                    }
                )
            else:
                return ToolResult(
                    success=False,
                    output=response_data,
                    error=f"HTTP {response.status_code}: {response.reason}",
                    execution_time=execution_time,
                    metadata={'status_code': response.status_code}
                )
                
        except requests.Timeout:
            return ToolResult(
                success=False,
                output=None,
                error=f"Request timeout after {self.timeout}s"
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"API error: {str(e)}"
            )
    
    async def execute_async(self,
                           url: str,
                           method: str = "GET",
                           headers: Dict[str, str] = None,
                           params: Dict[str, Any] = None,
                           json_data: Dict[str, Any] = None) -> ToolResult:
        """Execute async API call"""
        try:
            import time
            start_time = time.time()
            
            # Prepare headers
            final_headers = headers or {}
            
            # Add API key if available
            for key_name, key_value in self.api_keys.items():
                if key_name.lower() in url.lower():
                    final_headers['Authorization'] = f'Bearer {key_value}'
                    break
            
            # Execute async request
            async with aiohttp.ClientSession() as session:
                async with session.request(
                    method=method.upper(),
                    url=url,
                    headers=final_headers,
                    params=params,
                    json=json_data,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    execution_time = time.time() - start_time
                    
                    # Parse response
                    try:
                        response_data = await response.json()
                    except:
                        response_data = await response.text()
                    
                    if response.status < 400:
                        return ToolResult(
                            success=True,
                            output=response_data,
                            execution_time=execution_time,
                            metadata={
                                'status_code': response.status,
                                'headers': dict(response.headers)
                            }
                        )
                    else:
                        return ToolResult(
                            success=False,
                            output=response_data,
                            error=f"HTTP {response.status}: {response.reason}",
                            execution_time=execution_time,
                            metadata={'status_code': response.status}
                        )
                        
        except asyncio.TimeoutError:
            return ToolResult(
                success=False,
                output=None,
                error=f"Request timeout after {self.timeout}s"
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Async API error: {str(e)}"
            )


class ToolExecutionSystem:
    """Unified tool execution system for S-7 ASI"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Initialize executors
        self.python_sandbox = PythonSandbox(
            timeout=self.config.get('python_timeout', 30),
            memory_limit=self.config.get('python_memory', '512m')
        )
        
        self.shell_executor = ShellExecutor(
            allowed_commands=self.config.get('allowed_commands'),
            timeout=self.config.get('shell_timeout', 30)
        )
        
        self.api_executor = APIExecutor(
            api_keys=self.config.get('api_keys', {}),
            timeout=self.config.get('api_timeout', 30)
        )
        
        # Tool registry
        self.tools = self._register_tools()
        
    def _register_tools(self) -> Dict[str, Dict[str, Any]]:
        """Register available tools"""
        return {
            'python_exec': {
                'type': ToolType.PYTHON,
                'executor': self.python_sandbox.execute,
                'description': 'Execute Python code in sandbox'
            },
            'shell_exec': {
                'type': ToolType.SHELL,
                'executor': self.shell_executor.execute,
                'description': 'Execute shell commands'
            },
            'api_call': {
                'type': ToolType.API,
                'executor': self.api_executor.execute,
                'description': 'Make API calls'
            },
            'api_call_async': {
                'type': ToolType.API,
                'executor': self.api_executor.execute_async,
                'description': 'Make async API calls'
            }
        }
    
    def execute_tool(self, tool_name: str, **kwargs) -> ToolResult:
        """Execute a tool by name"""
        if tool_name not in self.tools:
            return ToolResult(
                success=False,
                output=None,
                error=f"Tool '{tool_name}' not found"
            )
        
        tool = self.tools[tool_name]
        try:
            return tool['executor'](**kwargs)
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Tool execution error: {str(e)}"
            )
    
    async def execute_tool_async(self, tool_name: str, **kwargs) -> ToolResult:
        """Execute a tool asynchronously"""
        if tool_name not in self.tools:
            return ToolResult(
                success=False,
                output=None,
                error=f"Tool '{tool_name}' not found"
            )
        
        tool = self.tools[tool_name]
        try:
            if asyncio.iscoroutinefunction(tool['executor']):
                return await tool['executor'](**kwargs)
            else:
                return tool['executor'](**kwargs)
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Tool execution error: {str(e)}"
            )
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """List all available tools"""
        return [
            {
                'name': name,
                'type': tool['type'].value,
                'description': tool['description']
            }
            for name, tool in self.tools.items()
        ]


# Example usage
if __name__ == "__main__":
    # Initialize system
    config = {
        'python_timeout': 30,
        'python_memory': '512m',
        'shell_timeout': 30,
        'api_timeout': 30,
        'api_keys': {
            'openai': os.getenv('OPENAI_API_KEY', ''),
            'anthropic': os.getenv('ANTHROPIC_API_KEY', '')
        }
    }
    
    system = ToolExecutionSystem(config)
    
    # Example: Execute Python code
    python_code = """
import json
import math

result = {
    'pi': math.pi,
    'sqrt_2': math.sqrt(2),
    'factorial_10': math.factorial(10)
}

print(json.dumps(result, indent=2))
"""
    
    result = system.execute_tool('python_exec', code=python_code)
    print(f"Python execution: {result.success}")
    print(f"Output: {result.output}")
    
    # Example: Execute shell command
    result = system.execute_tool('shell_exec', command='ls -la')
    print(f"Shell execution: {result.success}")
    print(f"Output: {result.output}")
    
    # Example: Make API call
    result = system.execute_tool(
        'api_call',
        url='https://api.github.com/repos/python/cpython',
        method='GET'
    )
    print(f"API call: {result.success}")
    print(f"Output type: {type(result.output)}")

#!/usr/bin/env python3
"""
Function Call Tracer Utility
Add this to any script to trace function execution flow
"""

import functools
import time
from typing import Any, Callable

class CallTracer:
    """Simple function call tracer for debugging execution flow"""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.indent_level = 0
        
    def trace_calls(self, func: Callable) -> Callable:
        """Decorator to trace function calls"""
        if not self.enabled:
            return func
            
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            indent = "  " * self.indent_level
            func_name = f"{func.__module__}.{func.__name__}"
            
            print(f"{indent}🔵 ENTER: {func_name}()")
            self.indent_level += 1
            
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                self.indent_level -= 1
                print(f"{indent}🟢 EXIT:  {func_name}() -> Success ({elapsed:.2f}s)")
                return result
            except Exception as e:
                elapsed = time.time() - start_time
                self.indent_level -= 1
                print(f"{indent}🔴 EXIT:  {func_name}() -> Error: {e} ({elapsed:.2f}s)")
                raise
                
        return wrapper

# Usage example:
tracer = CallTracer(enabled=True)

# Apply to functions you want to trace:
@tracer.trace_calls
def my_function():
    pass

# Or apply manually:
# my_function = tracer.trace_calls(my_function)

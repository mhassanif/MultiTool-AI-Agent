from langchain_core.tools import Tool
from langchain_experimental.utilities import PythonREPL

# init
python_repl = PythonREPL()

# tool to pass 
repl_tool = Tool(
    name="python_repl",
    description="A Python shell. Use this to execute Python commands. Input should be a valid Python command. If you want to see the output of a value, you should print it out with `print(...)`.",
    func=python_repl.run,
)

# test
result = repl_tool.func("import math; print(math.factorial(7))")
print("Python REPL Output:", result)

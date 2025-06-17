from langchain.agents import initialize_agent, Tool
from langchain_ollama.llms import OllamaLLM
from langchain_experimental.utilities import PythonREPL

# initialize llm
llm = OllamaLLM(model="phi3")

# intiailize tools
python_repl = PythonREPL()
repl_tool = Tool(
    name="Python REPL",
    description="A Python shell for executing Python commands. Use this to execute valid Python code. Ensure to print() results.",
    func=python_repl.run,
)

# tools List (more tools later)
tools = [repl_tool]

# agent intialization
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True,
)


query = """
Calculate the factorial of 17 please
"""
response = agent.run(query)
print("Agent Response:\n", response)

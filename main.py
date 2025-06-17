import os
import sys
from typing import List, Dict, Any
from datetime import datetime

# LangChain imports
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import BaseMessage
from langchain_ollama import OllamaLLM
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_experimental.utilities import PythonREPL
from langchain.tools import Tool
from langchain.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub



class TaskOrientedAgent:
    def __init__(self, model_name: str = "phi3", temperature: float = 0.1):
        self.model_name = model_name
        self.temperature = temperature
        
        # Initialize components
        self.llm = self._setup_llm()
        self.tools = self._setup_tools()
        self.memory = self._setup_memory()
        self.agent = self._setup_agent()
        
        print(f"✅ Agent initialized with {model_name} model")
        print(f"🛠️  Available tools: {[tool.name for tool in self.tools]}")
    
    def _setup_llm(self) -> OllamaLLM:
        """Setup the OllamaLLM LLM with phi3 model."""
        try:
            llm = OllamaLLM(
                model=self.model_name,
                temperature=self.temperature,
                verbose=True
            )
            # Test the connection
            test_response = llm.invoke("Hello, respond with 'OK' if you're working.")
            print(f"📡 LLM Connection Test: {test_response.strip()}")
            return llm
        except Exception as e:
            print(f"❌ Error setting up LLM: {e}")
            print("💡 Make sure OllamaLLM is running and phi3 model is installed:")
            print("   OllamaLLM pull phi3")
            print("   OllamaLLM serve")
            sys.exit(1)
    
    def _setup_tools(self) -> List[Tool]:
        """Setup the tools for the agent."""
        tools = []
        
        # DuckDuckGo Search Tool 
        try:
            ddg_search = DuckDuckGoSearchRun()
            search_tool = Tool(
                name="DuckDuckGoSearch",
                description="Useful for searching real-time information on the internet. "
                          "Input should be a search query string. "
                          "Use this when you need current information, facts, or data.",
                func=ddg_search.run
            )
            tools.append(search_tool)
        except Exception as e:
            print(f"⚠️  Warning: Could not setup DuckDuckGo search: {e}")
        
        # Python REPL Tool (using langchain_experimental)
        try:
            python_repl = PythonREPL()
            python_tool = Tool(
                name="PythonREPL",
                description="A Python shell. Use this to execute Python commands. Input should be a valid Python command. If you want to see the output of a value, you should print it out with `print(...)`.",
                func=python_repl.run
            )
            tools.append(python_tool)
            print("✅ Python REPL tool setup successfully")
        except Exception as e:
            print(f"⚠️  Warning: Could not setup Python REPL: {e}")
        
        # Memory retrieval tool
        memory_tool = Tool(
            name="ConversationMemory",
            description="Access previous conversation history and context. "
                      "Use this to recall information from earlier in the conversation.",
            func=self._get_memory_context
        )
        tools.append(memory_tool)
        
        return tools
    
    def _setup_memory(self) -> ConversationBufferWindowMemory:
        """Setup conversational memory."""
        return ConversationBufferWindowMemory(
            memory_key="chat_history",
            k=10,  # Keep last 10 exchanges
            return_messages=True
        )
    
    def _get_memory_context(self, query: str = "") -> str:
        """Retrieve context from conversation memory."""
        if hasattr(self.memory, 'chat_memory') and self.memory.chat_memory.messages:
            context = "Previous conversation context:\n"
            for msg in self.memory.chat_memory.messages[-6:]:  # Last 3 exchanges
                role = "Human" if msg.type == "human" else "Assistant"
                context += f"{role}: {msg.content}\n"
            return context
        return "No previous conversation context available."
    
    def _setup_agent(self) -> AgentExecutor:
        """Setup the ReAct agent with custom prompt."""
        
        # Custom prompt template for better reasoning
        prompt_template = """
You are a helpful AI assistant that can use tools to solve complex tasks.
You have access to the following tools:

{tools}

Here's what has been discussed so far:
{chat_history}

Use the following format for your responses:

Question: the input question you must answer
Thought: think about what you need to do step by step
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}
"""

        

        prompt = PromptTemplate(
            input_variables=["tools", "tool_names", "input", "agent_scratchpad"],
            template=prompt_template
        )
        print("📋 Using custom ReAct prompt template")
        
        # ReAct agent
        agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        
        # agent executor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=10,
            early_stopping_method="generate"
        )
        
        return agent_executor
    
    def run_task(self, task: str) -> str:
        print(f"\nTask: {task}")
        print("=" * 50)
        
        try:
            start_time = datetime.now()
            response = self.agent.invoke({"input": task})
            end_time = datetime.now()
            
            duration = (end_time - start_time).total_seconds()
            
            print(f"\Task completed in {duration:.2f} seconds")
            
            return response.get("output", "No output generated")
            
        except Exception as e:
            error_msg = f"Error executing task: {str(e)}"
            print(error_msg)
            return error_msg
    
    def interactive_mode(self):
        print("\nTask-Oriented AI Agent - Interactive Mode")
        print("Type 'quit', 'exit', or 'bye' to stop")
        print("=" * 50)
        
        while True:
            try:
                task = input("\nEnter your task: ").strip()
                
                if task.lower() in ['quit', 'exit', 'bye', '']:
                    print("Goodbye!")
                    break
                
                response = self.run_task(task)
                print(f"\nResponse: {response}")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")


def main():
    print("Task-Oriented AI Agent Using LangChain and Tool Integration!")
    agent = TaskOrientedAgent()
    agent.interactive_mode()

if __name__ == "__main__":
    main()
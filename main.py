import os
import sys
import json
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime

# LangChain imports
from langchain.agents import AgentExecutor, create_react_agent
from langchain.schema import BaseMessage
from langchain_ollama import OllamaLLM
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_experimental.utilities import PythonREPL
from langchain.tools import Tool
from langchain.prompts import PromptTemplate

# ChromaDB for persistent memory
import chromadb
from chromadb.config import Settings


class ChromaMemoryManager:
    """Handles persistent memory using ChromaDB."""
    
    def __init__(self, persist_directory: str = "./chroma_memory"):
        """Initialize ChromaDB client and collection."""
        self.persist_directory = persist_directory
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                is_persistent=True
            )
        )
        
        # Get or create collection for conversation memory
        try:
            self.collection = self.client.get_collection("conversation_memory")
            print("✅ Connected to existing ChromaDB memory collection")
        except:
            self.collection = self.client.create_collection(
                name="conversation_memory",
                metadata={"description": "Stores conversation history and context"}
            )
            print("✅ Created new ChromaDB memory collection")
    
    def add_interaction(self, user_input: str, agent_response: str, tools_used: List[str] = None):
        """Store a conversation interaction in ChromaDB."""
        interaction_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        # Create searchable text combining input and response
        searchable_text = f"User: {user_input}\nAgent: {agent_response}"
        
        metadata = {
            "timestamp": timestamp,
            "user_input": user_input,
            "agent_response": agent_response,
            "tools_used": json.dumps(tools_used or []),
            "interaction_id": interaction_id
        }
        
        self.collection.add(
            documents=[searchable_text],
            metadatas=[metadata],
            ids=[interaction_id]
        )
    
    def search_memory(self, query: str, n_results: int = 3) -> str:
        """Search through conversation history for relevant context."""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            if not results['documents'] or not results['documents'][0]:
                return "No relevant conversation history found."
            
            # Format the context from search results (more concise)
            context = "Relevant context:\n"
            for i, metadata in enumerate(results['metadatas'][0]):
                context += f"{i+1}. User: {metadata.get('user_input', 'N/A')[:100]}...\n"
                context += f"   Response: {metadata.get('agent_response', 'N/A')[:100]}...\n"
            
            return context
            
        except Exception as e:
            return f"Error: {str(e)}"
    
    def get_recent_context(self, limit: int = 2) -> str:
        """Get recent conversation context."""
        try:
            # Get all documents and sort by timestamp
            all_results = self.collection.get()
            
            if not all_results['metadatas']:
                return "No conversation history."
            
            # Sort by timestamp (most recent first)
            sorted_interactions = sorted(
                zip(all_results['metadatas'], all_results['documents']),
                key=lambda x: x[0].get('timestamp', ''),
                reverse=True
            )[:limit]
            
            if not sorted_interactions:
                return "No recent conversation history."
            
            context = "Recent context:\n"
            for metadata, doc in sorted_interactions:
                context += f"User: {metadata.get('user_input', 'N/A')[:80]}...\n"
                context += f"Agent: {metadata.get('agent_response', 'N/A')[:80]}...\n"
            
            return context
            
        except Exception as e:
            return f"Error: {str(e)}"


class EnhancedTaskOrientedAgent:
    """Enhanced task-oriented agent with ChromaDB memory and improved workflow."""
    
    def __init__(self, model_name: str = "phi3", temperature: float = 0.1):
        self.model_name = model_name
        self.temperature = temperature
        
        # Initialize components
        self.llm = self._setup_llm()
        self.memory_manager = ChromaMemoryManager()
        self.tools = self._setup_tools()
        self.agent = self._setup_agent()
        
        print(f"✅ Enhanced Agent initialized with {model_name} model")
        print(f"🛠️  Available tools: {[tool.name for tool in self.tools]}")
        print(f"🧠 ChromaDB memory system ready")
    
    def _setup_llm(self) -> OllamaLLM:
        """Setup the Ollama LLM."""
        try:
            llm = OllamaLLM(
                model=self.model_name,
                temperature=self.temperature,
                verbose=False  # Reduced verbosity for cleaner output
            )
            # Test the connection
            test_response = llm.invoke("Respond with 'OK'")
            print(f"📡 LLM Connection: {test_response.strip()}")
            return llm
        except Exception as e:
            print(f"❌ Error setting up LLM: {e}")
            print("💡 Make sure Ollama is running and phi3 model is installed")
            sys.exit(1)
    
    def _setup_tools(self) -> List[Tool]:
        """Setup tools for the agent."""
        tools = []
        
        # Search Tool
        try:
            ddg_search = DuckDuckGoSearchRun()
            search_tool = Tool(
                name="web_search",
                description="Search the internet for current information, facts, news, or data. "
                          "Use when you need up-to-date information not in your knowledge base.",
                func=ddg_search.run
            )
            tools.append(search_tool)
        except Exception as e:
            print(f"⚠️  Warning: Could not setup web search: {e}")
        
        # Python REPL Tool with better instructions
        try:
            python_repl = PythonREPL()
            python_tool = Tool(
                name="python_execute",
                description="Execute Python code for calculations, data analysis, or programming tasks. "
                          "CRITICAL: Always use print() to see results. Example: print(2+2) not just 2+2. "
                          "For variables: x=5; print(x). For lists: data=[1,2,3]; print(data). "
                          "Always print the final result you want to return.",
                func=python_repl.run
            )
            tools.append(python_tool)
        except Exception as e:
            print(f"⚠️  Warning: Could not setup Python REPL: {e}")
        
        # Memory Search Tool
        memory_search_tool = Tool(
            name="memory_search",
            description="Search through previous conversations for relevant context or information. "
                      "Use when you need to recall something from past interactions.",
            func=self.memory_manager.search_memory
        )
        tools.append(memory_search_tool)
        
        # Recent Context Tool
        recent_context_tool = Tool(
            name="recent_context",
            description="Get recent conversation context to understand ongoing discussion.",
            func=lambda x: self.memory_manager.get_recent_context()
        )
        tools.append(recent_context_tool)
        
        return tools
    
    def _setup_agent(self) -> AgentExecutor:
        """Setup the ReAct agent with improved prompt."""
        
        # Enhanced prompt template for better planning and execution
        prompt_template = """You are an intelligent task-oriented AI assistant. You think step by step and use tools efficiently.

CONVERSATION CONTEXT:
{memory_context}

AVAILABLE TOOLS:
{tools}

CRITICAL PYTHON USAGE RULES:
- When using python_execute, ALWAYS use print() to see results
- Example: print(2+2) NOT just 2+2
- For variables: x=5; print(x)  
- For calculations: result = 10*5; print(result)
- For lists/data: data=[1,2,3]; print(sum(data))

WORKFLOW GUIDELINES:
1. PLAN: Understand the task and plan your approach
2. EXECUTE: Use tools strategically - don't repeat the same action
3. RESPOND: Give direct answers based on findings

FORMAT:
Question: the input question you must answer
Thought: I need to [plan your approach - what tools will you use and why]
Action: the action to take, should be one of [{tool_names}]
Action Input: [specific input for the tool]
Observation: [result from the tool]
... (repeat Action/Action Input/Observation only if you need different information)
Thought: I now have enough information to answer
Final Answer: [clear, direct answer based on your findings]

Question: {input}
Thought: {agent_scratchpad}"""

        prompt = PromptTemplate(
            input_variables=["memory_context", "tools", "tool_names", "input", "agent_scratchpad"],
            template=prompt_template
        )
        
        # Create ReAct agent
        agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        
        # Create agent executor with improved settings
        agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=6,  # Reduced to prevent excessive loops
            early_stopping_method="generate",
            return_intermediate_steps=True
        )
        
        return agent_executor
    
    def run_task(self, task: str) -> Dict[str, Any]:
        """Execute a task and return detailed results."""
        print(f"\n📋 Task: {task}")
        print("=" * 60)
        
        try:
            start_time = datetime.now()
            
            # Get memory context for this task
            memory_context = self.memory_manager.search_memory(task)
            
            # Execute the task with memory context
            result = self.agent.invoke({
                "input": task,
                "memory_context": memory_context
            })
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Extract tools used from intermediate steps
            tools_used = []
            if 'intermediate_steps' in result and result['intermediate_steps']:
                tools_used = [step[0].tool for step in result['intermediate_steps']]
            
            # Get the final output
            output = result.get("output", "No output generated")
            
            # Store interaction in ChromaDB (async to avoid blocking)
            try:
                self.memory_manager.add_interaction(
                    user_input=task,
                    agent_response=output,
                    tools_used=tools_used
                )
            except Exception as e:
                print(f"⚠️  Memory storage warning: {e}")
            
            print(f"\n✅ Task completed in {duration:.2f} seconds")
            print(f"🛠️  Tools used: {', '.join(tools_used) if tools_used else 'None'}")
            
            return {
                "output": output,
                "duration": duration,
                "tools_used": tools_used,
                "success": True
            }
            
        except Exception as e:
            error_msg = f"❌ Error executing task: {str(e)}"
            print(error_msg)
            
            return {
                "output": error_msg,
                "duration": 0,
                "tools_used": [],
                "success": False
            }
    
    def interactive_mode(self):
        """Run the agent in interactive mode."""
        print("\n🤖 Enhanced Task-Oriented AI Agent")
        print("💭 Powered by ChromaDB memory system")
        print("Type 'quit', 'exit', or 'bye' to stop")
        print("Type 'memory' to search conversation history")
        print("=" * 60)
        
        while True:
            try:
                task = input("\n🎯 Enter your task: ").strip()
                
                if task.lower() in ['quit', 'exit', 'bye', '']:
                    print("👋 Goodbye!")
                    break
                
                if task.lower() == 'memory':
                    query = input("🔍 Search memory for: ").strip()
                    if query:
                        context = self.memory_manager.search_memory(query)
                        print(f"\n🧠 Memory Search Results:\n{context}")
                    continue
                
                # Execute the task
                result = self.run_task(task)
                
                # Display the response
                print(f"\n🤖 Response: {result['output']}")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")


def main():
    """Main function to run the enhanced agent."""
    print("🚀 Enhanced Task-Oriented AI Agent with ChromaDB Memory")
    print("=" * 60)
    
    try:
        agent = EnhancedTaskOrientedAgent()
        agent.interactive_mode()
    except Exception as e:
        print(f"❌ Failed to initialize agent: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
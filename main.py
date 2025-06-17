import streamlit as st
import asyncio
import json
import uuid
from typing import List, Dict, Any
from datetime import datetime
import chromadb
from chromadb.config import Settings
from langchain.agents import AgentExecutor, create_react_agent
from langchain.schema import BaseMessage
from langchain_ollama import OllamaLLM
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_experimental.utilities import PythonREPL
from langchain.tools import Tool
from langchain.prompts import PromptTemplate
from langchain.callbacks.base import AsyncCallbackHandler

# Custom callback handler to stream intermediate steps
class StreamlitCallbackHandler(AsyncCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.step_counter = 0

    async def on_agent_action(self, action, **kwargs):
        self.step_counter += 1
        with self.container:
            st.markdown(f"**Step {self.step_counter}: Thought**")
            st.write(action.log.strip())
            st.markdown(f"**Action**: {action.tool}")
            st.markdown(f"**Action Input**: {action.tool_input}")
            st.markdown("---")

    async def on_tool_end(self, output, **kwargs):
        with self.container:
            st.markdown(f"**Observation**: {output}")
            st.markdown("---")

class ChromaMemoryManager:
    def __init__(self):
        self.client = chromadb.EphemeralClient(settings=Settings(anonymized_telemetry=False))
        try:
            self.collection = self.client.get_collection("conversation_memory")
        except:
            self.collection = self.client.create_collection(
                name="conversation_memory",
                metadata={"description": "Stores conversation history and context"}
            )
    
    def add_interaction(self, user_input: str, agent_response: str, tools_used: List[str] = None):
        interaction_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
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
        try:
            results = self.collection.query(query_texts=[query], n_results=n_results)
            if not results['documents'] or not results['documents'][0]:
                return "No relevant conversation history found."
            context = "Relevant context:\n"
            for i, metadata in enumerate(results['metadatas'][0]):
                context += f"{i+1}. User: {metadata.get('user_input', 'N/A')[:100]}...\n"
                context += f"   Response: {metadata.get('agent_response', 'N/A')[:100]}...\n"
            return context
        except Exception as e:
            return f"Error: {str(e)}"
    
    def get_recent_context(self, limit: int = 2) -> str:
        try:
            all_results = self.collection.get()
            if not all_results['metadatas']:
                return "No conversation history."
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
    def __init__(self, model_name: str = "phi3", temperature: float = 0.1):
        self.model_name = model_name
        self.temperature = temperature
        self.llm = self._setup_llm()
        self.memory_manager = ChromaMemoryManager()
        self.tools = self._setup_tools()
    
    def _setup_llm(self) -> OllamaLLM:
        try:
            llm = OllamaLLM(model=self.model_name, temperature=self.temperature, verbose=False)
            return llm
        except Exception as e:
            st.error(f"Error setting up LLM: {e}")
            raise
    
    def _setup_tools(self) -> List[Tool]:
        tools = []
        try:
            ddg_search = DuckDuckGoSearchRun()
            search_tool = Tool(
                name="web_search",
                description="Search the internet for current information, facts, news, or data.",
                func=ddg_search.run
            )
            tools.append(search_tool)
        except Exception as e:
            st.warning(f"Could not setup web search: {e}")
        
        try:
            python_repl = PythonREPL()
            python_tool = Tool(
                name="python_execute",
                description="Execute Python code for calculations or data analysis. Always import libraries and use print().",
                func=python_repl.run
            )
            tools.append(python_tool)
        except Exception as e:
            st.warning(f"Could not setup Python REPL: {e}")
        
        return tools
    
    def _setup_agent(self, container) -> AgentExecutor:
        prompt_template = """You are an intelligent task-oriented AI assistant. You think step by step and use tools efficiently.

CONVERSATION CONTEXT (from previous interactions):
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
1. PLAN: Consider the conversation context and plan your approach
2. EXECUTE: Use tools strategically
3. RESPOND: Give direct answers based on context and findings

FORMAT:
Question: the input question you must answer
Thought: I need to [plan your approach considering the context]
Action: [{tool_names}]
Action Input: [specific input for the tool]
Observation: [result from the tool]
Thought: I now have enough information to answer
Final Answer: [clear, direct answer]

Question: {input}
Thought: {agent_scratchpad}"""
        
        prompt = PromptTemplate(
            input_variables=["memory_context", "tools", "tool_names", "input", "agent_scratchpad"],
            template=prompt_template
        )
        
        agent = create_react_agent(llm=self.llm, tools=self.tools, prompt=prompt)
        
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=6,
            early_stopping_method="generate",
            return_intermediate_steps=True,
            callbacks=[StreamlitCallbackHandler(container)]
        )
    
    async def run_task(self, task: str, container) -> Dict[str, Any]:
        start_time = datetime.now()
        memory_context = self.memory_manager.search_memory(task)
        agent_executor = self._setup_agent(container)
        
        result = await agent_executor.ainvoke({
            "input": task,
            "memory_context": memory_context
        })
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        tools_used = [step[0].tool for step in result['intermediate_steps']] if 'intermediate_steps' in result else []
        output = result.get("output", "No output generated")
        
        try:
            self.memory_manager.add_interaction(
                user_input=task,
                agent_response=output,
                tools_used=tools_used
            )
        except Exception as e:
            st.warning(f"Memory storage warning: {e}")
        
        return {
            "output": output,
            "duration": duration,
            "tools_used": tools_used,
            "success": True
        }

async def main():
    st.title("🤖 Enhanced Task-Oriented AI Agent")
    st.markdown("💭 Powered by ChromaDB memory and LangChain")
    st.markdown("Enter a task below to see the agent reason step-by-step in real-time.")

    # Initialize session state
    if 'agent' not in st.session_state:
        try:
            st.session_state.agent = EnhancedTaskOrientedAgent()
            st.success("Agent initialized successfully!")
        except Exception as e:
            st.error(f"Failed to initialize agent: {e}")
            return
    
    # Input form
    with st.form(key="task_form"):
        task = st.text_input("🎯 Enter your task:", placeholder="e.g., Calculate the square root of 25")
        submit_button = st.form_submit_button("Run Task")
    
    # Container for streaming steps
    reasoning_container = st.container()
    
    # Container for final output
    result_container = st.container()
    
    if submit_button and task:
        with reasoning_container:
            st.markdown("### Agent Reasoning Process")
            st.markdown("---")
        
        try:
            result = await st.session_state.agent.run_task(task, reasoning_container)
            with result_container:
                st.markdown("### Final Answer")
                st.write(result['output'])
                st.markdown(f"**Duration**: {result['duration']:.2f} seconds")
                st.markdown(f"**Tools Used**: {', '.join(result['tools_used']) if result['tools_used'] else 'None'}")
        except Exception as e:
            with result_container:
                st.error(f"Error executing task: {e}")

if __name__ == "__main__":
    asyncio.run(main())
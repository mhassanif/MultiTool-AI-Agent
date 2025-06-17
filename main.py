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
import nest_asyncio

# Enable nested asyncio for Streamlit compatibility
nest_asyncio.apply()

def run_async_task(agent, task, container):
    """Wrapper function to run async tasks in Streamlit"""
    try:
        # Try to get the current event loop
        loop = asyncio.get_event_loop()
    except RuntimeError:
        # If no event loop exists, create a new one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    # Run the async task
    return loop.run_until_complete(agent.run_task(task, container))

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

IMPORTANT: ALWAYS reference and use the conversation context provided below when relevant to the current question.

CONVERSATION CONTEXT (from previous interactions):
{memory_context}

AVAILABLE TOOLS:
{tools}

CRITICAL INSTRUCTIONS:
1. ALWAYS read and consider the conversation context above before answering
2. Reference previous interactions when they are relevant to the current question
3. Build upon previous answers and maintain conversation continuity
4. If the context contains relevant information, mention it in your response

CRITICAL PYTHON USAGE RULES:
- When using python_execute, ALWAYS use print() to see results
- Example: print(2+2) NOT just 2+2
- For variables: x=5; print(x)
- For calculations: result = 10*5; print(result)
- For lists/data: data=[1,2,3]; print(sum(data))

WORKFLOW GUIDELINES:
1. PLAN: Review conversation context and plan your approach
2. EXECUTE: Use tools strategically, referencing context when relevant
3. RESPOND: Give direct answers that build on previous interactions

FORMAT:
Question: the input question you must answer
Thought: I need to [plan your approach, considering the conversation context]
Action: [{tool_names}]
Action Input: [specific input for the tool]
Observation: [result from the tool]
Thought: I now have enough information to answer (considering previous context)
Final Answer: [clear, direct answer that references previous context when relevant]

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
        
        # Get both relevant memory and recent context
        memory_context = self.memory_manager.search_memory(task, n_results=5)
        recent_context = self.memory_manager.get_recent_context(limit=3)
        
        # Combine both contexts
        full_context = f"{recent_context}\n\n{memory_context}"
        
        # Debug: Show what memory context is being used
        with container:
            with st.expander("🧠 Memory Context Being Used", expanded=False):
                st.text(full_context)
        
        agent_executor = self._setup_agent(container)
        
        result = await agent_executor.ainvoke({
            "input": task,
            "memory_context": full_context
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
            st.success(f"✅ Interaction saved to memory (Total interactions: {len(self.memory_manager.collection.get()['ids'])})")
        except Exception as e:
            st.warning(f"Memory storage warning: {e}")
        
        return {
            "output": output,
            "duration": duration,
            "tools_used": tools_used,
            "success": True,
            "memory_context": full_context
        }

def main():
    st.title("🤖 Enhanced Task-Oriented AI Agent")
    st.markdown("💭 Powered by ChromaDB memory and LangChain")
    st.markdown("Enter tasks below to see the agent reason step-by-step. Ask multiple questions!")

    # Initialize session state
    if 'agent' not in st.session_state:
        try:
            st.session_state.agent = EnhancedTaskOrientedAgent()
            st.success("Agent initialized successfully!")
        except Exception as e:
            st.error(f"Failed to initialize agent: {e}")
            return
    
    # Initialize chat history in session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Clear chat button and memory status
    col1, col2, col3 = st.columns([1, 2, 2])
    with col1:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()
    
    with col2:
        if st.session_state.agent:
            try:
                memory_count = len(st.session_state.agent.memory_manager.collection.get()['ids'])
                st.info(f"🧠 Memory: {memory_count} interactions stored")
            except:
                st.info("🧠 Memory: Ready to store interactions")
    
    with col3:
        if st.button("🔄 Clear Memory"):
            if st.session_state.agent:
                try:
                    st.session_state.agent.memory_manager.collection.delete(
                        ids=st.session_state.agent.memory_manager.collection.get()['ids']
                    )
                    st.success("Memory cleared!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error clearing memory: {e}")
    
    # Input form at the top
    with st.form(key="task_form", clear_on_submit=True):
        task = st.text_input("🎯 Enter your task:", placeholder="e.g., Calculate the square root of 25")
        submit_button = st.form_submit_button("Ask Question")
    
    # Process new question
    if submit_button and task:
        # Add question to history immediately
        question_entry = {
            'type': 'question',
            'content': task,
            'timestamp': datetime.now()
        }
        st.session_state.chat_history.append(question_entry)
        
        # Show processing message
        with st.spinner('🤔 Agent is thinking...'):
            try:
                # Create a container for this specific question's reasoning
                reasoning_container = st.container()
                
                # Run the task using a wrapper function
                result = run_async_task(st.session_state.agent, task, reasoning_container)
                
                # Add reasoning and answer to history
                reasoning_entry = {
                    'type': 'reasoning',
                    'content': 'Reasoning steps completed',
                    'timestamp': datetime.now()
                }
                
                answer_entry = {
                    'type': 'answer',
                    'content': result['output'],
                    'duration': result['duration'],
                    'tools_used': result['tools_used'],
                    'timestamp': datetime.now()
                }
                
                st.session_state.chat_history.extend([reasoning_entry, answer_entry])
                
            except Exception as e:
                error_entry = {
                    'type': 'error',
                    'content': f"Error executing task: {e}",
                    'timestamp': datetime.now()
                }
                st.session_state.chat_history.append(error_entry)
        
        # Force rerun to update the display and reset the form
        st.rerun()
    
    # Display chat history
    if st.session_state.chat_history:
        st.markdown("## Chat History")
        
        # Group history items by question-reasoning-answer sets
        i = 0
        question_counter = 1
        
        while i < len(st.session_state.chat_history):
            entry = st.session_state.chat_history[i]
            
            if entry['type'] == 'question':
                # Display question
                st.markdown(f"### Question {question_counter}: {entry['content']}")
                
                # Look for corresponding reasoning and answer
                reasoning_content = None
                answer_content = None
                
                # Check next items for reasoning and answer
                if i + 1 < len(st.session_state.chat_history):
                    next_entry = st.session_state.chat_history[i + 1]
                    if next_entry['type'] == 'reasoning':
                        reasoning_content = next_entry
                        i += 1
                
                if i + 1 < len(st.session_state.chat_history):
                    next_entry = st.session_state.chat_history[i + 1]
                    if next_entry['type'] == 'answer':
                        answer_content = next_entry
                        i += 1
                    elif next_entry['type'] == 'error':
                        answer_content = next_entry
                        i += 1
                
                # Display collapsible reasoning section
                if reasoning_content:
                    with st.expander(f"🔍 View Reasoning Process for Question {question_counter}", expanded=False):
                        st.markdown("*Agent reasoning steps were executed above - expand to see this was processed*")
                        st.info("The detailed reasoning steps were shown in real-time when the question was processed.")
                
                # Display answer
                if answer_content:
                    if answer_content['type'] == 'answer':
                        st.markdown("**Answer:**")
                        st.write(answer_content['content'])
                        
                        # Show metadata
                        col1, col2 = st.columns(2)
                        with col1:
                            st.caption(f"⏱️ Duration: {answer_content['duration']:.2f} seconds")
                        with col2:
                            tools_used = answer_content.get('tools_used', [])
                            st.caption(f"🛠️ Tools: {', '.join(tools_used) if tools_used else 'None'}")
                    
                    elif answer_content['type'] == 'error':
                        st.error(answer_content['content'])
                
                st.markdown("---")
                question_counter += 1
            
            i += 1
    
    # Show helpful information at the bottom
    if not st.session_state.chat_history:
        st.info("👆 Ask your first question above to get started!")
    
    # Footer
    st.markdown("---")
    st.markdown("💡 **Tips:** You can ask multiple questions, and the agent will remember previous conversations!")

if __name__ == "__main__":
    main()
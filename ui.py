import streamlit as st
from streamlit_chat import message
from main import EnhancedTaskOrientedAgent
import json
from datetime import datetime

class StreamlitUI:
    """Handles the Streamlit user interface for the AI Agent."""
    
    def __init__(self):
        """Initialize the UI components and session state."""
        self.initialize_session_state()
        self.agent = self.initialize_agent()  # Initialize agent before sidebar
        self.setup_sidebar()
        
    def initialize_session_state(self):
        """Initialize session state variables."""
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        if 'debug_info' not in st.session_state:
            st.session_state.debug_info = []
        if 'reasoning_log' not in st.session_state:
            st.session_state.reasoning_log = []
        if 'current_reasoning' not in st.session_state:
            st.session_state.current_reasoning = None
    
    def initialize_agent(self):
        """Initialize the AI agent."""
        with st.spinner('🤖 Initializing AI Agent...'):
            return EnhancedTaskOrientedAgent()
    
    def setup_sidebar(self):
        """Setup the sidebar with options and memory search."""
        st.sidebar.title("🔧 Tools & Options")
        
        # Agent Reasoning Section (at the top for visibility)
        st.sidebar.markdown("### 🧠 Agent Reasoning")
        if st.session_state.current_reasoning:
            with st.sidebar.expander("Current Thought Process", expanded=True):
                st.code(st.session_state.current_reasoning)
        
        # Memory Search
        st.sidebar.markdown("### 🔍 Memory Search")
        memory_query = st.sidebar.text_input("Search conversation history:")
        if memory_query and hasattr(self, 'agent'):
            with st.sidebar:
                with st.spinner('Searching memory...'):
                    context = self.agent.memory_manager.search_memory(memory_query)
                    st.markdown(f"### Search Results\n{context}")
        
        # Statistics
        st.sidebar.markdown("### 📊 Statistics")
        if st.session_state.messages:
            total_interactions = len(st.session_state.messages) // 2
            col1, col2 = st.sidebar.columns(2)
            col1.metric("Total Interactions", total_interactions)
            col2.metric("Total Steps", len(st.session_state.debug_info))
            
        # Reasoning History
        if st.session_state.reasoning_log:
            with st.sidebar.expander("Previous Reasoning Steps", expanded=False):
                for i, log in enumerate(reversed(st.session_state.reasoning_log[-5:])):
                    st.markdown(f"**Step {len(st.session_state.reasoning_log) - i}**")
                    st.code(log)
                    st.markdown("---")
    
    def display_chat_history(self):
        """Display the chat history using streamlit-chat."""
        for i, msg in enumerate(st.session_state.messages):
            if i % 2 == 0:  # User message
                message(msg, is_user=True, key=f"user_{i}")
            else:  # Agent message
                message(msg, is_user=False, key=f"agent_{i}")
    
    def display_debug_info(self):
        """Display debug information in an easily visible area."""
        if st.session_state.debug_info:
            latest_debug = st.session_state.debug_info[-1]
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("### 🔍 Latest Action")
                st.markdown(f"""
                **Time:** {latest_debug['timestamp']}  
                **Duration:** {latest_debug['duration']:.2f} seconds  
                **Tools Used:** {', '.join(latest_debug['tools_used']) if latest_debug['tools_used'] else 'None'}
                """)
            
            with col2:
                if latest_debug['reasoning']:
                    with st.expander("View Reasoning", expanded=True):
                        st.code(latest_debug['reasoning'])
            
            with st.expander("View Previous Actions", expanded=False):
                for debug in reversed(st.session_state.debug_info[:-1]):
                    st.markdown(f"""
                    **Time:** {debug['timestamp']}  
                    **Duration:** {debug['duration']:.2f} seconds  
                    **Tools Used:** {', '.join(debug['tools_used']) if debug['tools_used'] else 'None'}
                    """)
                    if debug['reasoning']:
                        st.code(debug['reasoning'])
                    st.markdown("---")
    
    def run(self):
        """Run the Streamlit UI."""
        st.title("🤖 MultiTool AI Agent")
        st.markdown("""
        This AI agent can help you with various tasks using multiple tools including:
        - 🌐 Web search
        - 🐍 Python code execution
        - 🧠 Conversation memory
        """)
        
        # Display the latest reasoning and debug info at the top
        if st.session_state.current_reasoning:
            with st.expander("🧠 Current Reasoning Process", expanded=True):
                st.code(st.session_state.current_reasoning)
        
        # Chat interface
        self.display_chat_history()
        self.display_debug_info()
        
        # Input area
        user_input = st.chat_input("What can I help you with?")
        
        if user_input:
            # Add user message to chat
            st.session_state.messages.append(user_input)
            
            # Get agent response
            with st.spinner('🤔 Thinking...'):
                result = self.agent.run_task(user_input)
                
            # Update reasoning log
            if hasattr(result, 'get'):  # Check if result is a dict
                reasoning = result.get('reasoning', '')
                if reasoning:
                    st.session_state.current_reasoning = reasoning
                    st.session_state.reasoning_log.append(reasoning)
            
            # Add agent response to chat
            response = result['output'] if isinstance(result, dict) else str(result)
            st.session_state.messages.append(response)
            
            # Add debug info
            debug_info = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'duration': result.get('duration', 0) if isinstance(result, dict) else 0,
                'tools_used': result.get('tools_used', []) if isinstance(result, dict) else [],
                'reasoning': result.get('reasoning', '') if isinstance(result, dict) else ''
            }
            st.session_state.debug_info.append(debug_info)
            
            # Rerun to update the display
            st.rerun()

def main():
    """Main function to run the Streamlit UI."""
    ui = StreamlitUI()
    ui.run()

if __name__ == "__main__":
    main()

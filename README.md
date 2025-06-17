# Task-Oriented AI Chatbot

This project implements a task-oriented AI chatbot using Streamlit for a clean web interface, LangChain for agent-based reasoning, and ChromaDB for conversation memory. The chatbot processes user tasks (e.g., calculations, information retrieval) by reasoning step-by-step, utilizing tools like web search and Python code execution, and maintains context from previous interactions for coherent responses.

## Features
- **Interactive Chat Interface**: Built with Streamlit, featuring `st.chat_message` for displaying user and assistant messages with avatars and `st.chat_input` for user input.
- **Reasoning Display**: Shows real-time reasoning steps (thoughts, actions, observations) in a sidebar for transparency.
- **Memory Management**: Stores and retrieves conversation history using ChromaDB, enabling context-aware responses.
- **Tool Integration**: Supports two tools:
  - **DuckDuckGo Search**: Retrieves current information from the web for tasks requiring external data.
  - **Python REPL**: Executes Python code for calculations or data analysis, ensuring results are printed.
- **ReAct Agent**: Uses LangChain's ReAct (Reasoning + Acting) agent for multi-step reasoning, chosen for its ability to dynamically select tools based on task requirements.

## Project Components
- **Streamlit UI**: Provides a user-friendly interface with a main chat area and a sidebar. The chat area displays user inputs and assistant responses, while the sidebar shows reasoning steps, memory context, and controls (Clear Chat, Clear Memory).
- **LangChain ReAct Agent**: The core reasoning engine, built with `langchain` and `langchain_ollama`. It uses a prompt template to guide the agent through planning, tool execution, and response generation, referencing conversation context when relevant.
- **ChromaDB Memory**: A vector store (`chromadb`) that saves user inputs, assistant responses, and tool usage as searchable documents, allowing the agent to recall recent and relevant interactions.
- **Tools**:
  - **DuckDuckGo Search** (`langchain_community`): Used for tasks requiring up-to-date information, such as news or facts. The agent avoids unnecessary searches for non-information-retrieval tasks.
  - **Python REPL** (`langchain_experimental`): Executes Python code for tasks like calculations (e.g., "Calculate the square root of 25"), requiring `print()` statements for output visibility.
- **Ollama LLM**: The chatbot uses the `phi3` model by default, hosted via Ollama for local inference. Users can switch to a more powerful model (e.g., LLaMA, Mistral) based on their PC's capabilities for better performance.

## Why These Components?
- **Streamlit**: Chosen for its simplicity in creating interactive web UIs with minimal code, ideal for rapid prototyping and displaying chat elements.
- **ReAct Agent**: Selected over Zero-shot or Conversational agents because it supports multi-step reasoning and tool selection, making it suitable for complex, task-oriented queries.
- **ChromaDB**: Used for lightweight, local vector storage to maintain conversation history, enabling context-aware responses without external dependencies.
- **DuckDuckGo Search**: Provides a free, accessible search tool for real-time information retrieval, avoiding API costs.
- **Python REPL**: Enables dynamic code execution for computational tasks, enhancing the agent's versatility.
- **Ollama**: Allows local LLM deployment, ensuring privacy and flexibility to use different models based on hardware.

## Workflow
1. **User Input**: The user enters a task via the chat input (e.g., "Calculate the square root of 25").
2. **Context Retrieval**: The agent retrieves recent and relevant conversation history from ChromaDB to inform its response.
3. **Reasoning**: The ReAct agent follows a structured process:
   - **Plan**: Analyzes the task and context to decide the approach.
   - **Execute**: Selects and uses tools (e.g., Python REPL for calculations, DuckDuckGo for searches) if needed.
   - **Respond**: Generates a clear, context-aware answer.
4. **Display**: User input and assistant response appear in the main chat area, with reasoning steps (thought, action, observation) streamed to the sidebar.
5. **Memory Storage**: The interaction (input, response, tools used) is saved to ChromaDB for future reference.

## Setup Instructions
1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```
2. **Install Python**:
   - Ensure Python 3.9+ is installed.
3. **Install Ollama**:
   - Download and install Ollama from [ollama.ai](https://ollama.ai).
   - Pull the `phi3` model:
     ```bash
     ollama pull phi3
     ```
   - For better performance, consider pulling a more powerful model (e.g., `llama3`, `mistral`) based on your PC's capabilities (CPU/GPU, RAM).
4. **Install Dependencies**:
   - Create a virtual environment (optional but recommended):
     ```bash
     python -m venv venv
     source venv/bin/activate  # On Windows: venv\Scripts\activate
     ```
   - Install required libraries:
     ```bash
     pip install streamlit langchain langchain-ollama langchain-community langchain-experimental chromadb nest_asyncio
     ```
5. **Run the Application**:
   - Start the Streamlit app:
     ```bash
     streamlit run app.py
     ```
   - Open your browser to `http://localhost:8501` to interact with the chatbot.

## Usage
- **Chat Interface**: Type a task in the input box (e.g., "Calculate the square root of 25" or "Find recent AI news").
- **Sidebar**: View real-time reasoning steps and memory context in the sidebar. Use the "Clear Chat" button to reset the chat history or "Clear Memory" to delete stored interactions.
- **Model Selection**: The default model is `phi3`. To use a different model, update the `model_name` parameter in the `EnhancedTaskOrientedAgent` class (e.g., `model_name="llama3"`). Choose a model compatible with your hardware for optimal results.
- **Tools**: The agent automatically selects tools based on the task:
  - Python REPL for calculations or data analysis.
  - DuckDuckGo Search for information retrieval.
- **Memory**: The agent recalls up to 3 recent and 5 relevant past interactions to maintain conversation continuity.

## Known Issues
- **Ollama Dependency**: Requires a running Ollama server with the specified model. Ensure the server is active before running the app.
- **Memory Storage**: If ChromaDB fails to save interactions, a warning is displayed but does not affect the response.
- **Tool Limitations**: The Python REPL requires `print()` statements for output, and DuckDuckGo Search may return limited results for niche queries.

## Next Steps
- Enhance the UI with custom avatars or themes for `st.chat_message`.
- Add streaming response support for the LLM output (requires Ollama streaming compatibility).
- Integrate additional tools (e.g., API triggers, file processing) with proper safeguards.
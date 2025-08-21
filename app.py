import os
import re
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.agents import initialize_agent, Tool, AgentType
from langchain_core.messages import HumanMessage, AIMessage
from langchain.callbacks.base import BaseCallbackHandler
from langchain.agents.agent import AgentExecutor
from langchain.schema import AgentFinish
from langchain.memory import ConversationBufferMemory
from langchain.docstore.document import Document
from typing import List, Dict, Any, Tuple
import hashlib
from datetime import datetime

# =================== Load environment ===================
load_dotenv()

# API Keys from .env
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

# =================== Streamlit UI Configuration ===================
st.set_page_config(page_title="Gemini RAG Agent", page_icon="📈", layout="wide")
st.title("Agentic RAG with Websearch")

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "uploaded_files" not in st.session_state:
    st.session_state["uploaded_files"] = {}
if "vectorstore" not in st.session_state:
    st.session_state["vectorstore"] = None
if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# =================== Sidebar UI ===================
st.sidebar.header("Document Management")

# File uploader for multiple PDFs
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF files", 
    type=["pdf"], 
    accept_multiple_files=True,
    help="You can upload multiple PDF files for analysis"
)

# Button to reset all uploaded files
if st.sidebar.button("🗑️ Reset All Documents", help="Remove all uploaded documents"):
    st.session_state["uploaded_files"] = {}
    st.session_state["vectorstore"] = None
    st.sidebar.success("All documents have been removed")

# Display uploaded files with option to remove individual ones
if st.session_state["uploaded_files"]:
    st.sidebar.subheader("Uploaded Documents")
    files_to_remove = []
    


# Button to clear chat history
if st.sidebar.button("🗑️ Clear Chat History", help="Clear all conversation history"):
    st.session_state["messages"] = []
    st.session_state["memory"].clear()
    st.sidebar.success("Chat history cleared")

# =================== LLM and Embeddings ===================
@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0, streaming=True)

@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

llm = get_llm()
embeddings = get_embeddings()

# =================== PDF Processing with Caching ===================
def get_file_hash(file_content):
    return hashlib.md5(file_content).hexdigest()


# Update the process_pdfs function
def process_pdfs(uploaded_files):
    """Process uploaded PDFs and create/update vectorstore"""
    if not uploaded_files:
        return None
    
    all_docs = []
    
    for uploaded_file in uploaded_files:
        file_hash = get_file_hash(uploaded_file.getvalue())
        
        # Check if file is already processed
        if uploaded_file.name in st.session_state["uploaded_files"]:
            if st.session_state["uploaded_files"][uploaded_file.name]["hash"] == file_hash:
                # File hasn't changed, use existing docs
                all_docs.extend(st.session_state["uploaded_files"][uploaded_file.name]["documents"])
                continue
        
        # Process new or changed file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        try:
            with st.spinner(f"Processing {uploaded_file.name}..."):
                pdf_loader = PyPDFLoader(tmp_file_path)
                pages = pdf_loader.load()
                
                # Add metadata to track source and page numbers
                for i, page in enumerate(pages):
                    page.metadata["source"] = uploaded_file.name
                    page.metadata["page"] = i + 1
                
                # Store file info
                st.session_state["uploaded_files"][uploaded_file.name] = {
                    "hash": file_hash,
                    "documents": pages,
                    "processed_at": datetime.now().isoformat()
                }
                
                all_docs.extend(pages)
                
            os.unlink(tmp_file_path)
        except Exception as e:
            st.sidebar.error(f"Error processing {uploaded_file.name}: {str(e)}")
            continue
    
    if not all_docs:
        return None
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        add_start_index=True
    )
    
    split_docs = text_splitter.split_documents(all_docs)
    
    # Create or update vectorstore with proper configuration
    if st.session_state["vectorstore"] is None:
        # Create new vectorstore
        vectorstore = Chroma.from_documents(
            documents=split_docs,
            embedding=embeddings,
            collection_name="rag_documents",
            persist_directory="./chroma_store"
        )
        st.session_state["vectorstore"] = vectorstore
    else:
        # Clear existing vectorstore and create a new one with all documents
        # This ensures we don't have duplicate or stale data
        st.session_state["vectorstore"].delete_collection()
        vectorstore = Chroma.from_documents(
            documents=split_docs,
            embedding=embeddings,
            collection_name="rag_documents",
            persist_directory="./chroma_store"
        )
        st.session_state["vectorstore"] = vectorstore
    
    return st.session_state["vectorstore"]


# Process any uploaded files
if uploaded_files:
    with st.sidebar.status("Processing documents...", expanded=True) as status:
        vectorstore = process_pdfs(uploaded_files)
        if vectorstore:
            status.update(label="Documents processed successfully!", state="complete")
        else:
            status.update(label="Error processing documents", state="error")

# =================== Tools ===================
@st.cache_resource(show_spinner=False)
def get_tools():
    tools = []

    def pdf_retriever_func(query):
        try:
            if st.session_state["vectorstore"] is None:
                return "No documents available for search. Please upload PDF files first."
            
            # First try with a threshold, but if no results, try without threshold
            docs = st.session_state["vectorstore"].similarity_search_with_relevance_scores(
                query, k=5, score_threshold=0.3  # Lowered threshold
            )
            
            # If no results with threshold, try without threshold
            if not docs:
                docs = st.session_state["vectorstore"].similarity_search_with_relevance_scores(
                    query, k=5
                )
            
            if not docs:
                return "No relevant information found in the uploaded documents."
            
            # Format results with sources
            results = []
            for doc, score in docs:
                source = doc.metadata.get("source", "Unknown document")
                page = doc.metadata.get("page", "Unknown page")
                content = doc.page_content
                
                # Highlight query terms in the content
                for term in query.split():
                    if len(term) > 3:
                        pattern = re.compile(re.escape(term), re.IGNORECASE)
                        content = pattern.sub(f"**{term}**", content)
                
                results.append(f"[From {source}, page {page}, score: {score:.2f}]:\n{content}\n")
            
            return "\n\n".join(results)
        except Exception as e:
            return f"Error retrieving documents: {str(e)}"
    
    retriever_tool = Tool(
        name="PDF_Document_Search",
        func=pdf_retriever_func,
        description="Use this tool to search for information in the uploaded PDF documents. Always use this first when asked about content from uploaded files."
    )
    tools.append(retriever_tool)

    # Web Search Tool
    if SERPER_API_KEY:
        def safe_search_func(query):
            try:
                search = GoogleSerperAPIWrapper()
                results = search.results(query)
                
                if not results or "organic" not in results or not results["organic"]:
                    return "No relevant information found in web search."
                
                # Format results with sources
                formatted_results = []
                for i, result in enumerate(results["organic"][:3]):  # Top 3 results
                    title = result.get("title", "No title")
                    link = result.get("link", "No URL")
                    snippet = result.get("snippet", "No content available")
                    
                    formatted_results.append(
                        f"[{i+1}. {title}]({link}):\n{snippet}\n"
                    )
                
                return "\n\n".join(formatted_results)
            except Exception as e:
                return f"Web search failed: {str(e)}. Please try again or rephrase your query."
        
        search_tool = Tool(
            name="Web_Search",
            func=safe_search_func,
            description="Search the web for current information. Use this when the PDF documents don't contain the needed information or when asked for recent/current events."
        )
        tools.append(search_tool)
    
    # Enhanced Summary Tool
    def summary_func(query):
        try:
            # Extract the text to summarize from the query
            if "summarize" in query.lower() or "summary" in query.lower():
                # Try to get context from the conversation history
                chat_history = st.session_state["memory"].chat_memory.messages
                recent_user_input = ""
                
                for msg in reversed(chat_history):
                    if isinstance(msg, HumanMessage):
                        recent_user_input = msg.content
                        break
                
                # If we have a specific query, use the retriever to get relevant content
                if st.session_state["vectorstore"] and recent_user_input:
                    docs = st.session_state["vectorstore"].similarity_search(recent_user_input, k=3)
                    if docs:
                        content_to_summarize = "\n\n".join([doc.page_content for doc in docs])
                        
                        # Use LLM to generate summary
                        summary_prompt = f"""
                        Please provide a concise summary of the following content. 
                        Focus on the key points and main ideas.
                        
                        Content to summarize:
                        {content_to_summarize}
                        
                        Concise summary:
                        """
                        
                        summary = llm.invoke(summary_prompt)
                        return summary.content
                
                return "I need more specific content to summarize. Please provide the text or ask me to summarize specific document content."
            else:
                return "I can help summarize content. Please ask me to summarize something specific."
        except Exception as e:
            return f"Error generating summary: {str(e)}"
    
    summary_tool = Tool(
        name="Text_Summarization",
        func=summary_func,
        description="Use this tool when the user asks for a summary of specific content. It can summarize text from documents or provided content."
    )
    tools.append(summary_tool)

    return tools

# =================== Agent with Memory ===================
def get_agent():
    tools = get_tools()
    
    # Custom prompt to guide the agent
    custom_instructions = """
    You are an AI assistant that helps with document analysis and web research. 
    
    IMPORTANT: 
    1. If the user has uploaded PDF documents, always try to use the PDF_Document_Search tool first before using web search.
    2. Only use web search if the PDF documents don't contain the requested information or if the user asks for current information.
    3. When providing answers, always cite your sources clearly (document name and page for PDFs, URL for web sources).
    4. If the user asks for a summary, use the Text_Summarization tool with appropriate context.
    
    When calling a tool, ALWAYS use this exact format:
    Action: <ToolName>
    Action Input: <string input>
    Do NOT write parentheses or arguments inside the tool name.
    """
    
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,  # Use conversational agent for memory
        verbose=True,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
        max_iterations=4,  # Prevent infinite loops
        early_stopping_method="generate",
        memory=st.session_state["memory"],
        agent_kwargs={
            'prefix': custom_instructions,
            'input_variables': ['input', 'chat_history', 'agent_scratchpad']
        }
    )
    return agent

# =================== Custom Callback Handler ===================
class StreamlitCallbackHandler(BaseCallbackHandler):
    def __init__(self, container, expanded_container=None):
        self.container = container
        self.expanded_container = expanded_container
        self.text = ""
        self.final_answer = ""
        self.intermediate_steps = []
        
    def on_llm_start(self, serialized, prompts, **kwargs):
        self.text += "\n**Thinking...**\n"
        self.container.markdown(self.text)

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.final_answer += token
        self.container.markdown(self.text)

    def on_llm_end(self, response, **kwargs):
        self.text += "\n\n"
        self.container.markdown(self.text)

    def on_tool_start(self, serialized, input_str, **kwargs):
        tool_name = serialized.get('name', 'Unknown Tool')
        self.intermediate_steps.append({
            "type": "tool_start",
            "tool": tool_name,
            "input": input_str
        })
        self.text += f"\n**Executing:** {tool_name}\n"
        self.text += f"**Input:** {input_str}\n"
        self.container.markdown(self.text)

    def on_tool_end(self, output, **kwargs):
        # Truncate very long outputs to avoid flooding the UI
        truncated_output = output[:500] + "..." if len(output) > 500 else output
        self.intermediate_steps.append({
            "type": "tool_end",
            "output": truncated_output
        })
        self.text += f"**Result:** {truncated_output}\n\n"
        self.container.markdown(self.text)

    def on_agent_action(self, action, **kwargs):
        self.intermediate_steps.append({
            "type": "action",
            "tool": action.tool,
            "input": action.tool_input
        })
        self.text += f"\n**Action:** {action.tool}\n"
        self.text += f"**Action Input:** {action.tool_input}\n"
        self.container.markdown(self.text)
        
    def on_agent_finish(self, finish: AgentFinish, **kwargs):
        self.intermediate_steps.append({
            "type": "finish",
            "output": finish.return_values["output"]
        })
        self.text += "\n**Finalizing response...**\n"
        self.container.markdown(self.text)
        # Store the final answer
        self.final_answer = finish.return_values["output"]
        
        # Display intermediate steps in expandable section
        if self.expanded_container:
            with self.expanded_container.expander("View Intermediate Steps"):
                for step in self.intermediate_steps:
                    if step["type"] == "action":
                        st.write(f"**Action:** {step['tool']}")
                        st.write(f"**Input:** {step['input']}")
                    elif step["type"] == "tool_start":
                        st.write(f"**Tool Started:** {step['tool']}")
                        st.write(f"**Input:** {step['input']}")
                    elif step["type"] == "tool_end":
                        st.write(f"**Output:** {step['output']}")
                    elif step["type"] == "finish":
                        st.write("**Final Output Ready**")

# =================== Chat Interface ===================

st.markdown("""
This Streamlit app lets you **upload PDF documents, ask questions, and get answers** using a retrieval-augmented AI agent that searches your PDFs and the web.

**How to use:**
- Upload one or more PDFs via the sidebar.
- Type your question in the chat box. The AI will search documents first, then the web if needed.
- View the AI’s answer along with sources and optional **intermediate reasoning steps**.
""")


# Display chat messages
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        if "sources" in msg and msg["sources"]:
            st.markdown(msg["content"])
            with st.expander("View Sources"):
                for source in msg["sources"]:
                    st.write(f"📄 {source}")
        else:
            st.markdown(msg["content"])

# Chat input
if user_query := st.chat_input("Ask here..."):
    st.session_state["messages"].append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        # Create containers for main response and expanded details
        response_placeholder = st.empty()
        expanded_info = st.empty()
        
        # Initialize callback handler
        cb = StreamlitCallbackHandler(response_placeholder, expanded_info)
        
        try:
            with st.spinner("Processing your query..."):
                agent = get_agent()
                
                # Execute the agent
                response = agent.invoke(
                    {"input": user_query},
                    {"callbacks": [cb]}
                )
                
                # Extract sources from the response
                sources = []
                if "intermediate_steps" in response:
                    for step in response["intermediate_steps"]:
                        if isinstance(step, tuple) and len(step) == 2:
                            action, observation = step
                            # Extract document sources
                            if "From " in observation and "page" in observation:
                                source_matches = re.findall(r"From (.*?), page (\d+)", observation)
                                for match in source_matches:
                                    sources.append(f"Document: {match[0]}, Page: {match[1]}")
                            # Extract web sources
                            elif "http" in observation and "://" in observation:
                                url_matches = re.findall(r"\((https?://[^\s]+)\)", observation)
                                sources.extend(url_matches)
                
                # Add final response to messages with sources
                final_response = cb.final_answer if cb.final_answer else response.get("output", "Sorry, I couldn't generate a response.")
                
                st.session_state["messages"].append({
                    "role": "assistant", 
                    "content": final_response,
                    "sources": list(set(sources))  # Remove duplicates
                })
                
                # Display the final answer with sources
                st.markdown("### Final Answer:")
                st.markdown(final_response)
                
                if sources:
                    with st.expander("View Sources"):
                        for source in list(set(sources)):  # Remove duplicates
                            st.write(f"🔗 {source}")
            
        except Exception as e:
            error_msg = f"Error processing your request: {str(e)}. Please try again."
            st.error(error_msg)
            st.session_state["messages"].append({
                "role": "assistant", 
                "content": error_msg
            })

# Display current document status
if st.session_state["uploaded_files"]:
    st.sidebar.subheader("Document Status")
    for file_name, file_info in st.session_state["uploaded_files"].items():
        st.sidebar.write(f"✅ {file_name} (processed: {file_info['processed_at'][:10]})")
else:
    st.sidebar.info("Upload PDF documents to get started with document analysis.")
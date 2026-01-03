import streamlit as st
import os
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings, ChatHuggingFace
from langchain_classic.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv

# Configuration
DB_FAISS_PATH = r"C:\Projects\AI_medical_chatbot\vectorstore\db_faiss"
HUGGINGFACE_REPO_ID = "meta-llama/Llama-3.1-8B-Instruct"

# Load environment variables
load_dotenv()
TOKEN = os.getenv("HF_TOKEN")

if not TOKEN:
    raise ValueError("HF_TOKEN not found in environment variables")

# ============================================================================
# VECTORSTORE SETUP
# ============================================================================
@st.cache_resource
def load_vectorstore():
    """Load FAISS vectorstore with embeddings"""
    try:
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        db = FAISS.load_local(
            DB_FAISS_PATH, 
            embedding_model, 
            allow_dangerous_deserialization=True
        )
        return db
    except Exception as e:
        st.error(f"Failed to load vectorstore: {e}")
        return None

# ============================================================================
# LLM SETUP
# ============================================================================
@st.cache_resource
def load_llm():
    """Initialize LLM with HuggingFace endpoint"""
    try:
        llm_base = HuggingFaceEndpoint(
            repo_id=HUGGINGFACE_REPO_ID,
            temperature=0.5,
            max_new_tokens=512,
            huggingfacehub_api_token=TOKEN
        )
        return ChatHuggingFace(llm=llm_base)
    except Exception as e:
        st.error(f"Failed to load LLM: {e}")
        return None

# ============================================================================
# RAG CHAIN SETUP
# ============================================================================
@st.cache_resource
def setup_rag_chain():
    """Setup the complete RAG chain"""
    vectorstore = load_vectorstore()
    llm = load_llm()
    
    if vectorstore is None or llm is None:
        return None
    
    system_prompt = (
        "You are a helpful medical information assistant. "
        "Use the provided context to answer health-related questions accurately. "
        "If the context doesn't contain relevant information, say 'I don't have information about that.' "
        "Keep answers concise and avoid unnecessary details.\n\n"
        "Context: {context}"
    )
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    combine_docs_chain = create_stuff_documents_chain(llm, prompt_template)
    rag_chain = create_retrieval_chain(retriever, combine_docs_chain)
    
    return rag_chain

# ============================================================================
# UI & SESSION MANAGEMENT
# ============================================================================
def initialize_session():
    """Initialize session state"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = setup_rag_chain()

def display_chat_history():
    """Display previous messages"""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def get_response(prompt):
    """Get response from RAG chain"""
    try:
        response = st.session_state.rag_chain.invoke({"input": prompt})
        return response
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return None

def format_response(response):
    """Format the response with sources and page numbers"""
    if response is None:
        return None, []
    
    answer = response.get("answer", "")
    sources = []
    
    for doc in response.get("context", []):
        source = doc.metadata.get("source", "Unknown source")
        page = doc.metadata.get("page", None)
        
        if page is not None:
            source_text = f"{source} (Page {page})"
        else:
            source_text = source
        
        sources.append(source_text)
    
    return answer, sources
# ============================================================================
# MAIN APP
# ============================================================================
def main():
    st.set_page_config(page_title="Medical Health Assistant", layout="wide")
    st.title("💊 Ask me anything about your health")
    
    initialize_session()
    
    # Check if RAG chain is ready
    if st.session_state.rag_chain is None:
        st.error("⚠️ System not ready. Please check configuration and restart.")
        return
    
    # Display chat history
    display_chat_history()
    
    # Chat input
    user_input = st.chat_input("Ask me anything about your health...")
    
    if user_input:
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)
        
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })
        
        # Generate response with spinner
        with st.spinner("Thinking..."):
            response = get_response(user_input)
        
        if response:
            answer, sources = format_response(response)
            
            # Format assistant message with sources
            message_content = answer
            if sources and sources[0] != "Unknown source":
                message_content += "\n\n**Sources:**\n"
                for source in sources:
                    message_content += f"- {source}\n"
            
            # Display assistant message
            with st.chat_message("assistant"):
                st.markdown(message_content)
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": message_content
            })

if __name__ == "__main__":
    main()
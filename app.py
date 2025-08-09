import os
from dotenv import load_dotenv
import streamlit as st
import PyPDF2
import asyncio
from datasets import load_dataset
from src.core.db import init_db
from src.core.crud import save_message, load_history, create_new_session, get_all_sessions
from src.core.gemini_api import generate_gemini_response
import uuid
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Asyncio Event Loop Management ---
def get_or_create_eventloop():
    """Gets or creates the event loop for the current thread."""
    try:
        return asyncio.get_running_loop()
    except RuntimeError:  # 'get_running_loop' doesn't work in every context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop

def run_async(coro):
    """Runs an async coroutine in the thread's event loop."""
    return get_or_create_eventloop().run_until_complete(coro)

# Load env
load_dotenv()

def setup_environment():
    """Setup environment variables with error handling"""
    try:
        # Try to get from secrets first, then fallback to env
        if hasattr(st, 'secrets') and st.secrets:
            os.environ.setdefault("CHAT_DB_KEY", st.secrets.get("CHAT_DB_KEY", ""))
            os.environ.setdefault("GEMINI_API_KEY", st.secrets.get("GEMINI_API_KEY", ""))
        
        # Validate required keys
        if not os.environ.get("CHAT_DB_KEY"):
            st.error("❌ CHAT_DB_KEY is required but not found in environment variables or Streamlit secrets")
            st.stop()
        if not os.environ.get("GEMINI_API_KEY"):
            st.error("❌ GEMINI_API_KEY is required but not found in environment variables or Streamlit secrets")
            st.stop()
            
    except Exception as e:
        st.error(f"❌ Error setting up environment")
        st.stop()

setup_environment()

# Preload dataset with error handling
@st.cache_data(show_spinner="Loading disaster response dataset...")
def load_disaster_dataset():
    """Load disaster dataset with error handling"""
    try:
        logger.info("Loading disaster response dataset...")
        ds = load_dataset("disaster_response_messages", split="train")
        context_data = "\n".join(item["message"] for item in ds if "message" in item)
        logger.info(f"Loaded {len(ds)} disaster response messages")
        return context_data
    except Exception as e:
        logger.error(f"Failed to load disaster dataset: {e}")
        st.warning("⚠️ Could not load disaster response dataset. Using general knowledge only.")
        return ""

# Initialize session state
def init_session_state():
    """Initialize session state variables"""
    if 'current_session_id' not in st.session_state:
        st.session_state.current_session_id = None
    if 'sessions' not in st.session_state:
        st.session_state.sessions = []
    if 'uploaded_context' not in st.session_state:
        st.session_state.uploaded_context = ""

def create_new_chat_session():
    """Create a new chat session"""
    try:
        session_id = str(uuid.uuid4())
        session_name = f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
        run_async(create_new_session(session_id, session_name))
        
        st.session_state.current_session_id = session_id
        st.session_state.sessions.append({'id': session_id, 'name': session_name})
        st.rerun()
    except Exception as e:
        st.error(f"❌ Failed to create new chat session")

async def load_chat_sessions():
    """Load all chat sessions"""
    try:
        sessions = await get_all_sessions()
        st.session_state.sessions = [{'id': s.id, 'name': s.name} for s in sessions]
        if sessions and not st.session_state.current_session_id:
            st.session_state.current_session_id = sessions[0].id
    except Exception as e:
        st.error(f"❌ Failed to load chat sessions")

def process_uploaded_files(uploaded_files):
    """Process uploaded files and extract text"""
    user_context = ""
    if not uploaded_files:
        return user_context
        
    for file in uploaded_files:
        try:
            if file.type == "text/plain":
                content = file.read().decode('utf-8')
                user_context += f"\n--- {file.name} ---\n{content}\n"
            elif file.type == "application/pdf":
                reader = PyPDF2.PdfReader(file)
                pdf_text = ""
                for page_num, page in enumerate(reader.pages):
                    try:
                        text = page.extract_text()
                        if text.strip():
                            pdf_text += text + "\n"
                    except Exception as e:
                        st.warning(f"⚠️ Could not extract text from page {page_num + 1} of {file.name}: {e}")
                
                if pdf_text.strip():
                    user_context += f"\n--- {file.name} ---\n{pdf_text}\n"
                else:
                    st.warning(f"⚠️ No text could be extracted from {file.name}")
                    
        except Exception as e:
            st.error(f"❌ Error processing file {file.name}")
            
    return user_context

async def handle_chat_response(prompt: str, context_data: str, user_context: str, session_id: str):
    """Handle chat response with proper streaming"""
    try:
        # Prepare full context
        full_context = f"""Context from uploaded documents:
{user_context}

Disaster response dataset context:
{context_data}

User question: {prompt}"""

        # Create placeholder for streaming response
        response_placeholder = st.empty()
        full_response = ""
        
        # Stream the response
        async for chunk in generate_gemini_response(full_context):
            if chunk and chunk.strip():
                full_response += chunk
                response_placeholder.markdown(full_response + "▌")
        
        # Final response without cursor
        response_placeholder.markdown(full_response)
        
        # Save the complete response to database
        if full_response.strip():
            await save_message(session_id, "assistant", full_response)
        
        return full_response
        
    except Exception as e:
        st.error(f"❌ Error generating response")
        logger.error(f"Chat response error: {e}")
        return error_msg

# Main app
async def main():
    """Main application function"""
    try:
        # Initialize database
        await init_db()
        
        # Initialize session state
        init_session_state()
        
        # Load existing sessions
        await load_chat_sessions()
        
    except Exception as e:
        st.error(f"❌ Failed to initialize application")
        st.stop()

# Streamlit UI
st.set_page_config(
    page_title="Disaster Relief AI", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Run async initialization
try:
    run_async(main())
except Exception as e:
    st.error(f"❌ Application initialization failed")
    st.stop()

# Load context data once using cache
context_data = load_disaster_dataset()

# Header
st.title("🚨 Disaster Relief AI Assistant")
st.markdown("*Intelligent assistance for emergency responders and disaster management*")

# Sidebar
with st.sidebar:
    st.header("💬 Chat Sessions")
    
    # New chat button
    if st.button("➕ New Chat", use_container_width=True):
        create_new_chat_session()
    
    st.divider()
    
    # Display existing sessions
    if st.session_state.sessions:
        for session in reversed(st.session_state.sessions):  # Show newest first
            is_current = session['id'] == st.session_state.current_session_id
            if st.button(
                f"{'🟢' if is_current else '⚪'} {session['name']}", 
                key=f"session_{session['id']}",
                use_container_width=True
            ):
                st.session_state.current_session_id = session['id']
                st.rerun()
    else:
        st.info("No chat sessions yet. Click 'New Chat' to start!")
    
    st.divider()
    
    # File upload section
    st.header("📄 Upload Documents")
    st.markdown("*Upload PDF or text files for context*")
    
    uploaded_files = st.file_uploader(
        "Choose files",
        accept_multiple_files=True,
        type=["pdf", "txt"],
        help="Upload relevant documents to provide context for your questions"
    )
    
    # Process uploaded files
    if uploaded_files:
        with st.spinner("Processing uploaded files..."):
            st.session_state.uploaded_context = process_uploaded_files(uploaded_files)
        
        st.success(f"✅ Processed {len(uploaded_files)} file(s)")
        
        # Show file info
        for file in uploaded_files:
            st.info(f"📄 {file.name} ({file.size} bytes)")

# Main chat interface
if not st.session_state.current_session_id:
    st.info("👈 Please create a new chat session from the sidebar to get started!")
else:
    # Display current session name
    current_session = next((s for s in st.session_state.sessions if s['id'] == st.session_state.current_session_id), None)
    if current_session:
        st.subheader(f"💬 {current_session['name']}")
    
    # Display chat history for current session
    try:
        history = run_async(load_history(st.session_state.current_session_id))
        
        for role, message in history:
            with st.chat_message(role):
                st.markdown(message)
                
    except Exception as e:
        st.error(f"❌ Failed to load chat history")
        history = []
    
    # Chat input
    if prompt := st.chat_input("Ask a question about disaster response..."):
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Save user message
        try:
            run_async(save_message(st.session_state.current_session_id, "user", prompt))
            
            # Generate and display assistant response
            with st.chat_message("assistant"):
                run_async(handle_chat_response(
                    prompt, 
                    context_data, 
                    st.session_state.uploaded_context,
                    st.session_state.current_session_id
                ))
            
        except Exception as e:
            st.error(f"❌ Failed to process message")
            logger.error(f"Message processing error: {e}")

# Footer
st.divider()
st.markdown(
    "*🤖 Powered by Gemini AI | Built for emergency response teams*",
    help="This AI assistant provides information based on disaster response datasets and uploaded documents."
)


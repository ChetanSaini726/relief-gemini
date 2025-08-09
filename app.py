import streamlit as st
import fitz  # PyMuPDF
import asyncio
from datasets import load_dataset
from src.core.db import init_db
from src.core.crud import (
    save_message,
    load_history,
    create_new_session,
    get_all_sessions,
    delete_session,
)
from src.core.gemini_api import generate_context_aware_response, generate_session_title
import uuid
import logging

logger = logging.getLogger(__name__)

def setup_environment():
    """Setup environment variables with error handling"""
    try:
        # Validate required keys
        if not st.secrets.get("CHAT_DB_KEY"):
            st.error(
                "❌ CHAT_DB_KEY is required but not found in environment variables"
            )
            st.stop()
        if not st.secrets.get("GEMINI_API_KEY"):
            st.error(
                "❌ GEMINI_API_KEY is required but not found in environment variables"
            )
            st.stop()
        if not st.secrets.get("ENVIRONMENT"):
            "ENVIRONMENT", st.secrets.get("ENVIRONMENT", "prod")
            st.warning(
                "⚠️ ENVIRONMENT is required but not found in environment variables, set to production environment"
            )
    except Exception as e:
        st.error("❌ Error setting up environment")
        logger.error(f"Environment setup error: {e}")
        st.stop()

def setup_logging():
    """Setup logging configuration based on the environment"""
    if st.secrets.get("ENVIRONMENT") == "prod":
        logging.basicConfig(level=logging.WARNING)
    elif st.secrets.get("ENVIRONMENT") == "release":
        logging.basicConfig(level=logging.INFO)
    elif st.secrets.get("ENVIRONMENT") == "debug":
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)
    
setup_environment()
setup_logging()

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
        st.warning(
            "⚠️ Could not load disaster response dataset. Using general knowledge only."
        )
        return ""


# Initialize session state
def init_session_state():
    """Initialize session state variables"""
    if "current_session_id" not in st.session_state:
        st.session_state.current_session_id = None
    if "sessions" not in st.session_state:
        st.session_state.sessions = []
    if "uploaded_context" not in st.session_state:
        st.session_state.uploaded_context = ""


def create_new_chat_session():
    """Create a new chat session"""
    # This will set the stage for a new chat.
    # The actual session will be created in the DB after the first message.
    st.session_state.current_session_id = None
    st.rerun()


async def load_chat_sessions():
    """Load all chat sessions"""
    try:
        sessions = await get_all_sessions()
        st.session_state.sessions = [{"id": s.id, "name": s.name} for s in sessions]
        if sessions and not st.session_state.current_session_id:
            st.session_state.current_session_id = sessions[0].id
    except Exception as e:
        st.error("❌ Failed to load chat sessions")
        logger.error(f"Chat sessions loading error: {e}")


def process_uploaded_files(uploaded_files):
    """Process uploaded files and extract text"""
    user_context = ""
    if not uploaded_files:
        return user_context

    for file in uploaded_files:
        try:
            if file.type == "text/plain":
                content = file.read().decode("utf-8")
                user_context += f"\n--- {file.name} ---\n{content}\n"
            elif file.type == "application/pdf":
                pdf_text = ""
                with fitz.open(stream=file.read(), filetype="pdf") as doc:
                    for page_num, page in enumerate(doc):
                        try:
                            text = page.get_text()
                            if text.strip():
                                pdf_text += text + "\n"
                        except Exception as e:
                            st.warning(
                                f"⚠️ Could not extract text from page {page_num + 1} of {file.name}: {e}"
                            )

                if pdf_text.strip():
                    user_context += f"\n--- {file.name} ---\n{pdf_text}\n"
                else:
                    st.warning(f"⚠️ No text could be extracted from {file.name}")

        except Exception as e:
            st.error(f"❌ Error processing file {file.name}")
            logger.error(f"File processing error: {file.name} - {e}")

    return user_context


async def handle_chat_response(
    prompt: str, context_data: str, user_context: str, session_id: str
):
    """Handle chat response with proper streaming"""
    try:
        # Create placeholder for streaming response
        response_placeholder = st.empty()
        full_response = ""

        # Stream the response
        async for chunk in generate_context_aware_response(
            user_prompt=prompt,
            document_context=user_context,
            disaster_context=context_data,
        ):
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
        error_msg = f"❌ Error generating response: {e}"
        st.error("❌ Error generating response")
        logger.error(f"Chat response error: {e}")
        return error_msg


# Streamlit UI
st.set_page_config(
    page_title="Disaster Relief AI", layout="wide", initial_sidebar_state="expanded"
)

# Load context data once using cache
context_data = load_disaster_dataset()

# Header
st.title("🚨 Disaster Relief AI Assistant")
st.markdown("*Intelligent assistance for emergency responders and disaster management*")

# Perform one-time app initialization
init_session_state()
if "app_initialized" not in st.session_state:
    with st.spinner("Connecting to services..."):
        try:
            async def initialize_app():
                await init_db()
                await load_chat_sessions()
            run_async(initialize_app())
            st.session_state.app_initialized = True
        except Exception as e:
            st.error("❌ Application initialization failed.")
            logger.error(f"Fatal initialization error: {e}")
            st.stop()

# Sidebar
with st.sidebar:
    st.header("💬 Chat Sessions")

    # New chat button
    if st.button("➕ New Chat", use_container_width=True):
        create_new_chat_session()

    st.divider()

    # Display existing sessions
    if st.session_state.sessions:
        for session in st.session_state.sessions:
            col1, col2 = st.columns([0.85, 0.15])
            with col1:
                is_current = session["id"] == st.session_state.current_session_id
                if st.button(
                    f"{'🟢' if is_current else '⚪'} {session['name']}",
                    key=f"session_{session['id']}",
                    use_container_width=True,
                ):
                    st.session_state.current_session_id = session["id"]
                    st.rerun()
            with col2:
                if st.button(
                    "🗑️",
                    key=f"delete_{session['id']}",
                    use_container_width=True,
                    help="Delete this session",
                ):
                    run_async(delete_session(session["id"]))
                    st.session_state.sessions = [
                        s for s in st.session_state.sessions if s["id"] != session["id"]
                    ]
                    if st.session_state.current_session_id == session["id"]:
                        st.session_state.current_session_id = (
                            st.session_state.sessions[0]["id"]
                            if st.session_state.sessions
                            else None
                        )
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
        help="Upload relevant documents to provide context for your questions",
    )

    # Process uploaded files
    if uploaded_files:
        with st.spinner("Processing uploaded files..."):
            st.session_state.uploaded_context = process_uploaded_files(uploaded_files)

        st.success(f"✅ Processed {len(uploaded_files)} file(s)")

        # Show file info
        for file in uploaded_files:
            st.info(f"📄 {file.name} ({file.size} bytes)")

# Determine if we are in a new, unsaved chat session
is_new_chat = not st.session_state.current_session_id

# Main chat interface
if is_new_chat:
    st.subheader("💬 New Chat")
    st.info("Ask a question below to start a new conversation!")
else:
    # Display current session name
    current_session = next(
        (
            s
            for s in st.session_state.sessions
            if s["id"] == st.session_state.current_session_id
        ),
        None,
    )
    if current_session:
        st.subheader(f"💬 {current_session['name']}")

    # Display chat history for current session
    try:
        history = run_async(load_history(st.session_state.current_session_id))
        for role, message in history:
            with st.chat_message(role):
                st.markdown(message)
    except Exception as e:
        st.error("❌ Failed to load chat history")
        logger.error(f"Chat history loading error: {e}")

# Chat input
if prompt := st.chat_input("Ask a question about disaster response..."):
    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(prompt)

    if is_new_chat:
        # This is the first message of a new chat.
        async def create_and_process_new_session():
            session_id = str(uuid.uuid4())
            session_name = await generate_session_title(prompt)
            await create_new_session(session_id, session_name)

            # Update session state in-place before doing more async work
            st.session_state.current_session_id = session_id
            st.session_state.sessions.insert(
                0, {"id": session_id, "name": session_name}
            )

            await save_message(session_id, "user", prompt)

            with st.chat_message("assistant"):
                await handle_chat_response(
                    prompt, context_data, st.session_state.uploaded_context, session_id
                )

        run_async(create_and_process_new_session())
        st.rerun()
    else:
        # This is a message in an existing session
        session_id = st.session_state.current_session_id
        try:
            run_async(save_message(session_id, "user", prompt))
            with st.chat_message("assistant"):
                run_async(
                    handle_chat_response(
                        prompt,
                        context_data,
                        st.session_state.uploaded_context,
                        session_id,
                    )
                )
        except Exception as e:
            st.error("❌ Failed to process message")
            logger.error(f"Message processing error: {e}")

# Footer
st.divider()
st.markdown(
    "*🤖 Powered by Gemini AI | Built for emergency response teams*",
    help="This AI assistant provides information based on disaster response datasets and uploaded documents.",
)



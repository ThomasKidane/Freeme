
# app.py
import streamlit as st
import os
import logging
from datetime import datetime

# --- FIX for PyTorch/Streamlit Conflict ---
# (Keep the PyTorch workaround near the top if you still need it)
try:
    import torch
    if hasattr(torch.classes, "__file__") and torch.__path__:
         torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]
    logging.info("Applied PyTorch/Streamlit workaround if necessary.")
except ImportError:
    logging.warning("PyTorch not found, skipping workaround.")
except Exception as e:
    logging.error(f"Error applying PyTorch/Streamlit workaround: {e}")
# --- End Fix ---

# Langchain components
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma # Use the correct import
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_unstructured import UnstructuredLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain.prompts import PromptTemplate # <--- CHANGE: Import PromptTemplate

# --- Configuration ---
NOTES_FOLDER = "documents"
PERSIST_DIRECTORY = "./chroma_db_notes"
OLLAMA_BASE_URL = "http://localhost:11434"
LLM_MODEL = "llama3:8b" # <--- CHANGE: Use the 8 billion parameter model
EMBEDDING_MODEL = "nomic-embed-text" # Or your preferred embedding model

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- Custom Prompt Template Definition --- # <--- CHANGE: Define the new prompt
template = """You are an assistant answering questions based on the provided context from notes and if not just answer the question to the best of your abilities.
Your goal is to be helpful and provide accurate information.

Context from notes:
{context}

Chat History:
{chat_history}

Question: {question}
Answer:"""
CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(template)
# --- End Prompt Definition ---


# --- RAG Pipeline Functions ---

def load_documents_from_folder(folder_path):
    """Loads documents from the specified folder using DirectoryLoader."""
    try:
        loader = DirectoryLoader(
            folder_path,
            glob="**/*.*",
            use_multithreading=True,
            show_progress=True,
            loader_cls=lambda p: UnstructuredLoader(p, strategy="fast", mode="single"),
            silent_errors=True,
        )
        docs = loader.load()
        docs = [doc for doc in docs if doc is not None]
        logger.info(f"Loaded {len(docs)} documents from {folder_path}")
        if not docs:
             st.warning(f"No documents successfully loaded from '{folder_path}'. Check logs and dependencies (pandoc, libmagic, unstructured extras).")
        return docs
    except Exception as e:
        logger.error(f"Critical error loading documents from {folder_path}: {e}", exc_info=True)
        st.error(f"Failed to load documents: {e}.")
        return []


def split_documents(docs):
    """Splits documents into manageable chunks."""
    if not docs:
        return []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=150,
        length_function=len
        )
    split_docs = text_splitter.split_documents(docs)
    logger.info(f"Split {len(docs)} documents into {len(split_docs)} chunks.")
    return split_docs

def create_or_load_vectorstore(split_docs, persist_dir):
    """Creates a new Chroma vector store or loads an existing one."""
    try:
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)

        if os.path.exists(persist_dir) and os.listdir(persist_dir):
            logger.info(f"Loading existing vector store from {persist_dir} using langchain-chroma")
            vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
            return vectorstore
        else: # Create new
            if not split_docs:
                 st.error("No documents were loaded or split. Cannot create vector store.")
                 return None
            logger.info(f"Creating new vector store in {persist_dir} using langchain-chroma")

            logger.info("Filtering complex metadata from document chunks...")
            filtered_docs = filter_complex_metadata(split_docs)
            logger.info(f"Original chunks: {len(split_docs)}, Chunks after filtering: {len(filtered_docs)}")
            if not filtered_docs:
                st.error("All document chunks were filtered out. Check if documents have content and minimal valid metadata.")
                return None

            vectorstore = Chroma.from_documents(
                documents=filtered_docs,
                embedding=embeddings,
                persist_directory=persist_dir
            )
            logger.info("Vector store created and persisted.")
            return vectorstore
    except Exception as e:
        logger.error(f"Error creating/loading vector store: {e}", exc_info=True)
        st.error(f"Failed to initialize vector store: {e}. Is Ollama running and the embedding model '{EMBEDDING_MODEL}' available?")
        return None

def create_conversational_chain(vectorstore):
    """Creates the Conversational Retrieval Chain."""
    if vectorstore is None:
        return None
    try:
        # Use the updated LLM_MODEL
        llm = OllamaLLM(model=LLM_MODEL, base_url=OLLAMA_BASE_URL)

        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            chat_memory=StreamlitChatMessageHistory(key="notes_chat_history"),
            output_key='answer' # Keep this explicit output key
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) # <--- CHANGE: Try retrieving fewer docs (e.g., 3) for the 8B model

        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            # <--- CHANGE: Inject the custom prompt template ---
            combine_docs_chain_kwargs={"prompt": CUSTOM_QUESTION_PROMPT},
            # --- End prompt injection ---
            verbose=True
        )
        logger.info("ConversationalRetrievalChain created.")
        return chain
    except Exception as e:
        logger.error(f"Error creating conversational chain: {e}", exc_info=True)
        st.error(f"Failed to create QA chain: {e}. Is Ollama running and the LLM model '{LLM_MODEL}' available?")
        return None

# --- Streamlit UI ---
# (The Streamlit UI part remains unchanged from your previous working version)
# Make sure the code below this line is identical to your last working version.

st.set_page_config(page_title="Freeme", page_icon="📝")
st.title("📝 Freeme")
st.write("Ask questions about your notes stored in the local 'documents' folder.")

# Initialize chat history
msgs = StreamlitChatMessageHistory(key="notes_chat_history")
if len(msgs.messages) == 0:
    msgs.add_ai_message("How can I help you analyze your notes today?")

# Display chat messages from history
for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

# Load or create vector store and chain
if "conversation_chain" not in st.session_state:
    st.session_state.conversation_chain = None
    st.session_state.vectorstore = None

    with st.spinner("Loading and Indexing Notes... Please wait."):
        if os.path.exists(NOTES_FOLDER) and os.listdir(NOTES_FOLDER):
            docs = load_documents_from_folder(NOTES_FOLDER)
            if docs:
                split_docs = split_documents(docs)
                if split_docs:
                     st.session_state.vectorstore = create_or_load_vectorstore(split_docs, PERSIST_DIRECTORY)
                     if st.session_state.vectorstore:
                         st.session_state.conversation_chain = create_conversational_chain(st.session_state.vectorstore)
                         st.success(f"Notes from '{NOTES_FOLDER}' indexed successfully!")
                     else:
                         st.error("Failed to initialize vector store. Cannot proceed.")
                else:
                     st.warning("Documents were loaded but could not be split into chunks.")
        else:
             st.error(f"Notes folder '{NOTES_FOLDER}' not found or is empty. Please create it and add notes.")


# Handle user input
if prompt := st.chat_input("Ask a question about your notes"):
    st.chat_message("human").write(prompt)

    if st.session_state.conversation_chain is None:
        st.error("The analysis chain is not ready. Please check configuration and logs.")
    else:
        with st.spinner("Thinking..."):
            try:
                result = st.session_state.conversation_chain.invoke({"question": prompt})
                response = result["answer"]
                st.chat_message("ai").write(response)

                with st.expander("Show Sources"):
                    if result.get("source_documents"):
                         for doc in result["source_documents"]:
                              st.info(f"Source: {doc.metadata.get('source', 'Unknown')}")
                              st.text(doc.page_content[:500] + "...")
                    else:
                         st.write("No specific source documents were retrieved for this answer.")

            except Exception as e:
                logger.error(f"Error during chain invocation: {e}", exc_info=True)
                st.error(f"An error occurred: {e}")

# Sidebar
with st.sidebar:
    st.header("Options")
    if st.button("Re-index Notes"):
        # (Keep the re-indexing logic as is)
        with st.spinner("Re-indexing..."):
            # Clear existing session state related to the index/chain
            if "conversation_chain" in st.session_state: del st.session_state.conversation_chain
            if "vectorstore" in st.session_state: del st.session_state.vectorstore
            if os.path.exists(PERSIST_DIRECTORY):
                 import shutil
                 try:
                     shutil.rmtree(PERSIST_DIRECTORY)
                     logger.info(f"Removed old index at {PERSIST_DIRECTORY}")
                 except Exception as e:
                     logger.error(f"Could not remove old index directory: {e}")
                     st.error(f"Could not clear old index: {e}")

            # Reload and re-index
            docs = load_documents_from_folder(NOTES_FOLDER)
            if docs:
                 split_docs = split_documents(docs)
                 if split_docs:
                    st.session_state.vectorstore = create_or_load_vectorstore(split_docs, PERSIST_DIRECTORY)
                    if st.session_state.vectorstore:
                        st.session_state.conversation_chain = create_conversational_chain(st.session_state.vectorstore)
                        st.success("Re-indexing complete!")
                    else:
                        st.error("Failed to re-initialize vector store after clearing.")
                 else:
                    st.warning("No chunks produced during re-indexing.")
            else:
                 st.warning("No documents found during re-indexing attempt.")
        st.rerun()

    st.info(f"Using LLM: {LLM_MODEL}") # Will now show llama3:8b
    st.info(f"Using Embeddings: {EMBEDDING_MODEL}")
    st.info(f"Notes Folder: ./{NOTES_FOLDER}")
    st.info(f"Index Directory: ./{PERSIST_DIRECTORY}")































































































































































































































































































































































































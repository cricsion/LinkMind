import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_huggingface import HuggingFaceEmbeddings
import streamlit as st
import asyncio
import hashlib

# Generate a unique SHA-256 hash for a document using its content and metadata.
def generate_doc_hash(doc):
    """Create SHA-256 hash of document content and metadata."""
    content = doc.page_content + str(doc.metadata)  # Combine text content and metadata
    return hashlib.sha256(content.encode()).hexdigest()

# Asynchronously save document chunks to the FAISS vector store in memory.
async def save_embeddings(chunks):
    if not chunks:
        raise ValueError("Chunks not provided!")
    
    # Check if the FAISS database exists in session; if not, initialize it.
    if "db" not in st.session_state or st.session_state.db is None:
        db = await fetch_model()
    else:
        db = st.session_state.db

    await db.aadd_documents(chunks)  # Add document chunks to FAISS index asynchronously
    print(f"Saved {len(chunks)} chunks to FAISS in memory")
    st.session_state.db = db  # Update session state with the latest database
    return st.session_state.db

# Asynchronously initialize and load the FAISS vector store with Hugging Face embeddings.
async def fetch_model():
    # If no existing FAISS database is found in session, create one.
    if "db" not in st.session_state or st.session_state.db is None:
        # Initialize embeddings using a pre-trained sentence transformer model.
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        # Create a FAISS index, determining dimensions from a sample embedding.
        index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))
        # Construct the FAISS vector store with an in-memory docstore.
        db = FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )
        st.session_state.db = db  # Store the database in the session state
    return st.session_state.db

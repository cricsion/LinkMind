from tools.data import fetch_model
from langchain_core.tools import tool
import hashlib

def generate_content_hash(content: str) -> str:
    """Generate SHA-256 hash for document content"""
    return hashlib.sha256(content.encode()).hexdigest()

@tool("DBSearch",parse_docstring=True)
async def fetch_information(query: str, top_k: int=6): 
    """
    Search the database for relevant documents and return unique results by filtering out duplicates.
    
    Args:
        query (str): The search query text to find relevant documents.
        top_k (int): The maximum number of documents to return.

    Returns:
        str: A concatenated string containing the unique relevant documents, up to the specified limit of `top_k`.
    """
    db = await fetch_model()

    raw_results = await db.asimilarity_search_with_relevance_scores(query, k=top_k*3)

    # Filter by score and deduplicate
    seen_hashes = set()
    unique_results = []

    for doc, score in raw_results:
        content_hash = generate_content_hash(doc.page_content)
        if content_hash not in seen_hashes:
            seen_hashes.add(content_hash)
            unique_results.append(doc)
            
        if len(unique_results) >= top_k:
            break
            
    if not unique_results:
        return "No relevant information found matching the query\n"

    context = "\n\n---\n\n".join([doc.page_content for doc in unique_results])+"\n\n---\n\n"
    return  context

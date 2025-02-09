from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import Iterator
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

class CustomDocumentLoader(BaseLoader):
    """A custom document loader that yields Document objects from various file content formats."""
    
    def __init__(self, file_content) -> None:
        """
        Initialize the loader with file content.

        Args:
            file_content: Can be a dictionary containing keys like 'content', 'snippet', 'title', and 'link'
                          or a simple string representing the file content.
        """
        self.file = file_content

    async def lazy_load(self) -> Iterator[Document]:
        """
        Lazily load and yield Document objects from the provided file content.

        Yields:
            Document: A Document object containing text content and metadata.
                      The generation approach varies based on available keys in the file content.
        """
        file_ = self.file
        
        # If detailed metadata with a snippet is provided, use it to populate the Document.
        if "snippet" in file_.keys():
            yield Document(
                page_content=file_["content"],
                metadata={
                    "snippet": file_["snippet"],
                    "source": file_["title"],
                    "link": file_["link"]
                },
            )
        # If only basic content and link are available, yield a Document with that information.
        elif "content" in file_.keys():
            yield Document(
                page_content=file_["content"],
                metadata={"link": file_["link"]},
            )
        # If the file content is a plain string, wrap it in a Document.
        elif isinstance(file_, str):
            yield Document(
                page_content=file_,
            )
        
        yield Document(page_content="")

async def split_text(documents, chunk_size=1000, chunk_overlap=500, length_function=len, add_start_index=True):
    """
    Split documents into smaller text chunks with potential overlaps for enhanced processing.

    Args:
        documents: A list of Document objects to be split.
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Number of overlapping characters between consecutive chunks to maintain context.
        length_function: Function for computing string length; defaults to Python's built-in len.
        add_start_index: If True, include the start index of each chunk in its metadata.

    Returns:
        List[Document]: A list of Document chunks after splitting the input documents.
    """
    # Create a text splitter instance with custom configuration and multiple separators.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap, 
        length_function=length_function, 
        add_start_index=add_start_index, 
        separators=["\n\n", "\n", "\n\n---\n\n"]
    )
    
    # Split the documents and return the resulting chunks.
    chunks = text_splitter.split_documents(documents)
    return chunks

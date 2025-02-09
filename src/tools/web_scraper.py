from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.tools import tool
import trafilatura
from tools.preprocess import CustomDocumentLoader, split_text
from tools.data import save_embeddings
import asyncio

@tool("Search",parse_docstring=True)
async def fetch_sites(query : str) -> str:
    """
    Performs a web search based on the provided query.

    Args:
        query (str): The search term or question to query.

    Returns:
        str: The result or answer retrieved from the web search, excluding the source.
    """
    search = DuckDuckGoSearchResults(output_format="list",source="text")
    ret = await search.ainvoke(query,backend="text")
    fetched = []
    top_k=3
    content=""
    for row in ret[:top_k]:
        row["content"]=await visit.ainvoke({"query":row})
        content=content+'\n\nTitle:'+row['title']#+'\n\nLink:'+row['link']
        if row["content"]!=None:
            content=content+'\n\nContent:'+row["content"]
        else:
            content=content+'\n\nContent: None'
        fetched.append(row)

    search = DuckDuckGoSearchResults(output_format="list",source="news")
    ret = await search.ainvoke(query,backend="news")
    for row in ret[:top_k]:
        row["content"]=await visit.ainvoke({"query":row})
        content=content+'\n\nTitle:'+row['title']#+'\n\nLink:'+row['link']
        if row["content"]!=None:
            content=content+'\n\nContent:'+row["content"]
        else:
            content=content+'\n\nContent: None'
        fetched.append(row)

    return str(content+'\n\n')

@tool("OpenLink",parse_docstring=True)
async def visit(query : dict) -> str:
    """
    Extracts content from a webpage given a URL.

    Args:
        query (dict): A dictionary containing the key "link" with the URL of the webpage to extract content from.

    Returns:
        str: The extracted content from the webpage, excluding the source or metadata.
    """
    content = trafilatura.fetch_url(query["link"])
    query["content"]=trafilatura.extract(content,favor_precision=True, favor_recall=True)
    if query["content"]==None:
        return query["content"]
    loader = CustomDocumentLoader(query)
    documents=[]
    async for doc in loader.lazy_load():
        documents.append(doc)
    chunks = await split_text(documents, 2048, 512)
    await save_embeddings(chunks)
    return query["content"]

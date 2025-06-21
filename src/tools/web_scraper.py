from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.tools import tool
import trafilatura
from tools.preprocess import CustomDocumentLoader, split_text
from tools.data import save_embeddings
import asyncio
from functools import lru_cache

@lru_cache(maxsize=128)
def cached_fetch_content(url: str) -> str:
    """
    Fetches the content from a given URL, extracts it with trafilatura,
    processes it with a custom document loader, splits it into chunks,
    and saves embeddings. The results are cached for repeated queries.
    """
    i=0
    content = trafilatura.fetch_url(url)
    # content=trafilatura.extract(content,output_format="json",favor_precision=True, favor_recall=True)
    content=trafilatura.extract(content,favor_precision=True, favor_recall=True)
    while content==None and i<5:
        content = trafilatura.fetch_url(url)
        content=trafilatura.extract(content,favor_precision=True, favor_recall=True)
        i+=1

    return content 

@lru_cache(maxsize=128)
def cached_search_content(query: str, source: str) -> list:
    """
    Performs a DuckDuckGo search query for the specified source (text or news)
    and caches the raw result list.
    """
    search = DuckDuckGoSearchResults(output_format="list", source=source)
    ret = search.invoke(query, backend=source)
    return ret

@tool("Search",parse_docstring=True)
async def fetch_sites(query : str) -> str:
    """
    Performs a web search based on the provided query.

    Args:
        query (str): The search term or question to query.

    Returns:
        str: The result or answer retrieved from the web search, excluding the source.
    """
    # Web search
    web_search_task = asyncio.create_task(asyncio.to_thread(lambda: cached_search_content(query, "text")))

    # News search
    news_search_task = asyncio.create_task(asyncio.to_thread(lambda: cached_search_content(query, "news")))

    web_ret, news_ret = await asyncio.gather(web_search_task, news_search_task)

    # Combine results and eliminating duplicate web results
    unique_results = {}
    for item in web_ret + news_ret:
        # Using link as a unique identifier
        if item.get('link') and item['link'] not in unique_results:
            unique_results[item['link']] = item
    
    ret = list(unique_results.values())

    fetched = []
    top_k=8
    content=""
    # Create tasks for text search results
    web_tasks = [visit.ainvoke({"query": row}) for row in ret[:top_k]]
    web_results = await asyncio.gather(*web_tasks)
    
    for i, row in enumerate(ret[:top_k]):
        row["content"] = web_results[i]
        if row["content"] is not None:
            content = content + '\n\nTitle:' + row['title'] + '\n\nLink:'+row['link']
            content = content + '\n\nContent:' + row["content"]
        fetched.append(row)

    # # Create tasks for news search results
    # news_tasks = [visit.ainvoke({"query": row}) for row in ret[:top_k]]
    # news_results = await asyncio.gather(*news_tasks)
    
    # for i, row in enumerate(ret[:top_k]):
    #     row["content"] = news_results[i]
    #     if row["content"] is not None:
    #         content = content + '\n\nTitle:' + row['title'] + '\n\nLink:'+row['link']
    #         content = content + '\n\nContent:' + row["content"]
    #     fetched.append(row)

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
    query["content"]=cached_fetch_content(query["link"])
    if query["content"] is None:
        return "No content could be extracted from the webpage."

    # Process the document and save embeddings
    await process_and_save(query)

    # loader = CustomDocumentLoader(query)
    # documents=[]
    # async for doc in loader.lazy_load():
    #     documents.append(doc)
    # chunks = await split_text(documents, 2048, 512)
    # await save_embeddings(chunks)
    return query["content"]

async def process_and_save(query):
    """
    Background task to process documents and save embeddings.
    """
    try:
        loader = CustomDocumentLoader(query)
        documents = []
        async for doc in loader.lazy_load():
            documents.append(doc)
        chunks = await split_text(documents, 2048, 512)
        batch_size = 10
        for i in range(0,len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            # Process the batch of documents
            await save_embeddings(batch)
    except Exception as e:
        print(f"Background processing error: {str(e)}")
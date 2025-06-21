from dotenv import load_dotenv
import os
from langchain.agents import (Tool, AgentExecutor, create_react_agent,)
from langchain.memory import ConversationBufferMemory
from langchain import hub
from huggingface_hub import login
from tools.web_scraper import fetch_sites, visit
from tools.relevance_search import fetch_information
from tools.data import fetch_model
import asyncio
from langchain.globals import set_llm_cache
from langchain_community.cache import InMemoryCache
import logging
from typing import Dict, Any, Optional, List, Union
import uuid
from langchain_google_genai import ChatGoogleGenerativeAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set up an in-memory cache to optimize LLM calls
set_llm_cache(InMemoryCache())

# Load environment variables from .env file
load_dotenv()

# Retrieve model identifier and Hugging Face API token from env variables and log in
MODEL = os.environ["MODEL"]
GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]

# Initialize conversation memory to maintain chat context
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key='output')

# Pull the base prompt from a hub repo and append additional instructions
agent_prompt = hub.pull("hwchase17/react-chat") + """

Do not finalize the answer until you have received and processed the Observation from any tool you invoke.
It's very important to always include the 'Thought' before any 'Action' or 'Final Answer'. Ensure your output strictly follows the formats above.
You are required to observe the entire output given by a tool.
Action Input: input of the action MUST be followed by a new line.
Ensure that the same tool is NEVER called more than once with identical input—only re-invoke the tool when the input has changed or additional processing is required.

You are developed by Daniyal Anis"""

# Define tools for web search and content extraction
tools = [
    Tool(
        name="Search",
        func=lambda x: asyncio.run(fetch_sites.ainvoke(x)),
        description="A search engine optimized for comprehensive, accurate, and trusted results. Useful for when you need to answer questions about current events. This returns only the answer - not the original source data.",
    ),
    Tool(
        name="OpenLink",
        func=lambda url: asyncio.run(visit.ainvoke({"query": {"link": url}})),
        description="Extracts webpage content from URL. Input: string URL. Returns: extracted text content"
    ),
    Tool(
        name="DBSearch",
        func=lambda query, top_k=6: asyncio.run(fetch_information.ainvoke({"query": query, "top_k": top_k})),
        description="Search database for relevant documents. Contents of websearches are cached in the database, so that they could be accessed through this tool"
    ),
]

# Wrap the LLM into a chat model expecting JSON formatted output
chat_model = ChatGoogleGenerativeAI(
    model=MODEL,
    temperature=0.1,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    google_api_key=GOOGLE_API_KEY,
)

# Create a React-based agent that integrates the chat model, prompt, and defined tools
rag_agent = create_react_agent(
    llm=chat_model,
    prompt=agent_prompt,
    tools=tools,
)

# Instantiate an AgentExecutor with conversation memory and execution constraints
rag_agent_executor = AgentExecutor(
    agent=rag_agent,
    tools=tools,
    return_intermediate_steps=True,
    verbose=True,
    memory=memory,
    handle_parsing_errors="Check your output and make sure it conforms! Do not output an action and a final answer at the same time.",
    agent_executor_kwargs={
        "max_iterations": 10,  
        "max_execution_time": 180,  
        "early_stopping_method": "generate"  
    }
)

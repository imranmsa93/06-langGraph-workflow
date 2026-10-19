from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from langchain_ollama import ChatOllama
from langchain.tools import tool


MODEL= "qwen3:1.7b"
load_dotenv()
@tool
def triple(input:float) -> float:
    """
    input param: take a float paramter
    output: triple the input param and return it
    """
    return float(input) * 3

tools = [TavilySearch(max_results = 1), triple]
llm = ChatOllama(model=MODEL,temperature=0.4).bind_tools(tools)



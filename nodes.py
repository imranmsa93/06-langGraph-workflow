
from react import llm, tools
from dotenv import load_dotenv
from langgraph.prebuilt import ToolNode
from langgraph.graph import MessagesState

load_dotenv()

SYSYEM_MESSAGE="""
You are a helpful assistant that can use tools to answer questions.
"""

def react_reasoning(state:MessagesState) -> MessagesState:
    """
    Run the agent reasoning node.   
""" 
    response = llm.invoke([{"role": "system", "content": SYSYEM_MESSAGE}, *state["messages"]])
    return {"messages": [response]}

tool_node = ToolNode(tools)

    

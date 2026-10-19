
from typing import TypedDict
from langgraph.graph import StateGraph, START, END, MessagesState
from nodes import react_reasoning, tool_node
from langchain.messages import HumanMessage

# class MessagesState(TypedDict):
#     question:str
#     answer:str


builder = StateGraph(MessagesState)
builder.add_node("react_reasoning", react_reasoning)
builder.add_node("tools", tool_node)
builder.add_node("final_answer", react_reasoning)  # same function, second node

builder.add_edge(START, "react_reasoning")
builder.add_edge("react_reasoning", "tools")
builder.add_edge("tools", "final_answer")
builder.add_edge("final_answer", END)

app = builder.compile()
app.get_graph().draw_mermaid_png(output_file_path="flow.png")

def main():
    print("Hello ReAct LangGraph with Function Calling")
    res = app.invoke({"messages": [HumanMessage(content="What is the temperature in Sydney, Australia? List it and then triple it")]})
    for m in res["messages"]:
        m.pretty_print()   # shows every step, not just the last one


if __name__ == "__main__":
    main()

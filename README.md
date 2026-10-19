# ReAct LangGraph with Function Calling

A learning project that builds a **ReAct-style tool-using agent** with
[LangGraph](https://docs.langchain.com/oss/python/langgraph/overview), a local
Ollama model, and two tools (web search and a custom math tool).

The project is built in stages. Stage 1 is a **linear flow** so the mechanics
are easy to see. Stage 2 adds **conditional edges** so the model decides when
to use tools and when to stop.

---

## 1. What this project does

You ask a question in natural language. The graph:

1. Sends the conversation to a local LLM (`qwen3:1.7b` via Ollama).
2. Lets the LLM request a tool (web search or `triple`).
3. Runs the requested tool.
4. Sends the tool result back to the LLM to write the final answer.

Example prompt:

> What is the temperature in Toranto? List it and then triple it.

This needs **two tools in sequence**: search for the temperature, then triple it.

---

## 2. Learning path and where this project fits

| Step | Topic | Status |
|---|---|---|
| 1 | Why graphs: chains vs workflows vs agents | Done |
| 2 | Basic components: State, Node, Edge, START/END, compile, invoke | Done |
| 3 | **Linear ReAct flow with tools (this project, Stage 1)** | In progress |
| 4 | Conditional edges and the tool loop (Stage 2) | Next |
| 5 | Memory and checkpointers (`thread_id`) | Planned |
| 6 | Human-in-the-loop interrupts | Planned |
| 7 | Agentic RAG graph for the Cordova assistant | Planned |

---

## 3. Core concepts used

| Concept | Meaning | Where it appears |
|---|---|---|
| **State** | Shared data every node can read and update | `MessagesState` |
| **Node** | A Python function that does one step | `react_reasoning`, `tool_node` |
| **Edge** | An arrow deciding which node runs next | `add_edge(...)` in `main.py` |
| **START / END** | Special entry and exit markers | `main.py` |
| **compile()** | Locks the design into a runnable graph | `builder.compile()` |
| **invoke()** | Runs the graph on an input | `app.invoke(...)` |
| **Tool** | A function the LLM can ask to be run | `triple`, `TavilySearch` |
| **ToolNode** | Prebuilt node that executes the LLM's tool calls | `nodes.py` |
| **bind_tools** | Tells the LLM which tools exist | `react.py` |

### MessagesState

`MessagesState` holds a single key, `messages`, a list of chat messages.
Each node returns only the **new** message(s). LangGraph **appends** them
instead of overwriting, so the list grows through the run:

```
HumanMessage → AIMessage (tool call) → ToolMessage (result) → AIMessage (answer)
```

---

## 4. Project structure

```
project/
├── main.py        # Builds, compiles and runs the graph
├── nodes.py       # Node functions: LLM reasoning + ToolNode
├── react.py       # LLM setup, tools, bind_tools
├── .env           # API keys (never commit this)
└── pyproject.toml # uv-managed dependencies
```

### Dependency direction (no circular imports)

```
main.py  →  nodes.py  →  react.py
```

Imports must only point **right**. `nodes.py` must never import from `main.py`.

---

## 5. Setup

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) for dependency management
- [Ollama](https://ollama.com/) installed and running
- A [Tavily](https://tavily.com/) API key for web search

### Install dependencies

```bash
uv add langgraph langchain langchain-ollama langchain-tavily python-dotenv
```

> **Note:** use `python-dotenv`, not `dotenv`. They are different packages on
> PyPI. The import `from dotenv import load_dotenv` comes from `python-dotenv`.
> If you added the wrong one: `uv remove dotenv` then `uv add python-dotenv`.

### Pull the model

```bash
ollama pull qwen3:1.7b
```

### Environment variables

Create a `.env` file in the project root:

```
TAVILY_API_KEY=your_key_here
```

### Run

```bash
uv run main.py
```

---

## 6. Code walkthrough (Stage 1: linear flow)

### `react.py`: model and tools

```python
from dotenv import load_dotenv
load_dotenv()   # must run BEFORE TavilySearch is created

from langchain_tavily import TavilySearch
from langchain_ollama import ChatOllama
from langchain.tools import tool

MODEL = "qwen3:1.7b"

@tool
def triple(input: float) -> float:
    """Take a number and return it multiplied by 3."""
    return float(input) * 3

tools = [TavilySearch(max_results=1), triple]
llm = ChatOllama(model=MODEL, temperature=0.4).bind_tools(tools)
```

Key points:

- The `@tool` decorator turns a function into a tool. The **docstring is what
  the LLM reads** to decide when to use it, so write it clearly.
- `bind_tools(tools)` lets the model emit structured tool calls.
- `load_dotenv()` runs first because `TavilySearch` needs `TAVILY_API_KEY`
  at creation time.

### `nodes.py`: the two kinds of node

```python
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode
from react import llm, tools

SYSTEM_MESSAGE = "You are a helpful assistant that can use tools to answer questions."

def react_reasoning(state: MessagesState) -> MessagesState:
    response = llm.invoke(
        [{"role": "system", "content": SYSTEM_MESSAGE}, *state["messages"]]
    )
    return {"messages": [response]}

tool_node = ToolNode(tools)
```

- `react_reasoning` is the **thinking** node: it calls the LLM with the system
  prompt plus the conversation so far.
- `tool_node` is the **acting** node: it reads the last AI message, runs every
  tool call in it, and adds the results as `ToolMessage`s.

### `main.py`: wiring the graph

```python
from dotenv import load_dotenv
from langchain.messages import HumanMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from nodes import react_reasoning, tool_node

load_dotenv()

builder = StateGraph(MessagesState)
builder.add_node("react_reasoning", react_reasoning)
builder.add_node("tools", tool_node)
builder.add_node("final_answer", react_reasoning)  # same function, second node

builder.add_edge(START, "react_reasoning")
builder.add_edge("react_reasoning", "tools")
builder.add_edge("tools", "final_answer")
builder.add_edge("final_answer", END)

app = builder.compile()

def main():
    res = app.invoke(
        {"messages": [HumanMessage(content="What is the temperature in Tokyo?")]}
    )
    for m in res["messages"]:
        m.pretty_print()   # shows every step, not just the last one

if __name__ == "__main__":
    main()
```

---

## 7. How the graph runs

```
START → react_reasoning → tools → final_answer → END
```

| Step | Node | What happens | Messages added |
|---|---|---|---|
| 1 | (input) | User question enters state | `HumanMessage` |
| 2 | `react_reasoning` | LLM decides to call a tool, usually with empty text | `AIMessage` with `tool_calls` |
| 3 | `tools` | `ToolNode` runs the call | `ToolMessage` with the result |
| 4 | `final_answer` | LLM reads the tool result and writes the answer | `AIMessage` with text |
| 5 | END | Graph stops | none |

---

## 8. Problems found while building (and fixes)

| # | Problem | Symptom | Fix |
|---|---|---|---|
| 1 | `nodes.py` imported `State` from `main.py`, while `main.py` imports `nodes.py` | `ImportError` (circular import) | Remove the import. It was unused |
| 2 | `tool_node` imported but never added to the graph | Final message is an `AIMessage` with `tool_calls` and **empty content**, so `print` shows a blank | `builder.add_node("tools", tool_node)` plus an edge to it |
| 3 | `load_dotenv()` never called before `TavilySearch(...)` | Missing `TAVILY_API_KEY` error | Call `load_dotenv()` at the top of `react.py` |
| 4 | Printed only `res["messages"][-1].content` | Hard to see what happened | Loop with `m.pretty_print()` |
| 5 | Typo `SYSYEM_MESSAGE` | None (consistent), but confusing | Renamed to `SYSTEM_MESSAGE` |

---

## 9. Known limitations of the linear flow

The Stage 1 graph is intentionally rigid. Its limits are the reason for Stage 2.

| Limitation | Example |
|---|---|
| Tools **always** run once, even if not needed | Prompt "hi" still goes through `tools` |
| Only **one** tool round | "Temperature in Tokyo, then triple it" needs search first, then `triple`. The linear graph runs only one tool step, so the `triple` call never executes and the answer comes back empty or wrong |
| Model can't skip tools or stop early | Can't answer simple questions directly |
| Model can't retry a tool with different input | One shot only |

---

## 10. Stage 2: conditional edges (next lesson)

Replace the fixed path with a **decision after every LLM call**:

```
START → react_reasoning ──(tool_calls present?)──→ tools ──→ react_reasoning
                         │
                         └──(no tool_calls)──→ END
```

The decision is a **router function** that reads the last message:

- Last message has `tool_calls` → go to `tools`, then **loop back** to the model.
- No `tool_calls` → go to `END`.

This lets one graph handle all of these:

| User asks | Behavior |
|---|---|
| Just the temperature | model → search → model → END |
| Just "triple 14" | model → `triple` → model → END |
| Temperature, then triple it | model → search → model → `triple` → model → END |
| "Hi" | model → END (no tools) |

LangGraph ships a ready-made router for this pattern, `tools_condition`,
which we will use after writing one by hand to understand it.

> Add a loop cap (recursion limit) at that point so a confused model can't
> loop forever.

---

## 11. Troubleshooting

| Symptom | Likely cause | What to try |
|---|---|---|
| `ImportError` on startup | Circular import | Check imports point only `main → nodes → react` |
| Blank final output | Last message is a tool call, not text | Use `pretty_print()` on all messages; ensure tools node exists |
| `TAVILY_API_KEY` error | `.env` not loaded or key missing | Call `load_dotenv()` before creating `TavilySearch` |
| Connection refused to Ollama | Ollama not running | Start Ollama, confirm `ollama list` shows `qwen3:1.7b` |
| Model ignores tools or sends bad arguments | 1.7B models are weak at tool calling | Try a larger model before debugging the graph |
| Edge error about unknown node | Name typo | Names in `add_edge` must exactly match `add_node` |
| Graph won't run | Forgot `compile()` | Always `builder.compile()` before `invoke()` |

---

## 12. Professional notes

- **Compile once, invoke many times.** Define the graph at module level.
- **Print every message while learning.** Reading the message sequence is the
  best way to understand agent behavior.
- **Small local models are unreliable at tools.** Weak results can be a model
  problem, not a graph problem.
- **Each loop turn is another LLM call.** More autonomy means more latency
  and, with hosted models, more cost.
- **Tool docstrings are prompts.** Vague docstrings lead to wrong tool choices.
- **Never commit `.env`.** Add it to `.gitignore`.

---

## 13. Glossary

| Term | Meaning |
|---|---|
| **ReAct** | Reason, then Act: the LLM thinks, calls a tool, reads the result, repeats |
| **Workflow** | Your code decides the path |
| **Agent** | The LLM decides the path at runtime |
| **Flow engineering** | Improving results by designing a multi-step process with checks and retries, rather than relying on one prompt |
| **Super-step** | One round of node execution in LangGraph |
| **Conditional edge** | An edge whose destination is chosen by a router function |
| **Checkpointer** | Saves graph state per conversation so it can resume (later lesson) |

---

## 14. Next actions

1. Run Stage 1 with: "What is the temperature in Tokyo?" and read every message.
2. Run Stage 1 with: "Triple 14." and observe what the linear flow does.
3. Run Stage 1 with: "Temperature in Tokyo, then triple it." and note which tool call never runs.
4. Move on to Stage 2: conditional edges and the tool loop.
from typing import TypedDict, List, Dict, Any
import json
import re

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, BaseMessage, SystemMessage
from langchain_ollama import ChatOllama


class ChatState(TypedDict):
    messages: List[BaseMessage]
    knowledge_graph: Dict[str, Any]
    last_user_input: str
    current_stage: str


llm = ChatOllama(model="qwen2.5:1.5b")

def safe_invoke(messages: List[BaseMessage]):
    try:
        return llm.invoke(messages)
    except Exception as e:
        raise RuntimeError(f"LLM error: {e}") from e


def capture_user_input(state: ChatState) -> ChatState:
    last_user_message = next(
        (msg for msg in reversed(state["messages"]) if isinstance(msg, HumanMessage)),
        None,
    )
    text = last_user_message.content.strip() if last_user_message else ""

    return {
        **state,
        "last_user_input": text,
        "current_stage": "input_captured",
    }


def extract_and_update_graph(state: ChatState) -> ChatState:
    text = state["last_user_input"]

    kg = {
        "entities": dict(state["knowledge_graph"].get("entities", {})),
        "relations": list(state["knowledge_graph"].get("relations", [])),
        "attributes": dict(state["knowledge_graph"].get("attributes", {})),
    }

    kg["entities"]["user"] = {"type": "person"}

    name_match = re.search(r"\bmy name is ([a-zA-Z\s]+?)(?:,| and|\.|$)", text, re.IGNORECASE)
    if name_match:
        name = name_match.group(1).strip().title()
        kg["entities"][name] = {"type": "person"}
        kg["attributes"]["name"] = name

        relation = {
            "source": "user",
            "relation": "name",
            "target": name,
        }
        if relation not in kg["relations"]:
            kg["relations"].append(relation)

    live_match = re.search(r"\bi live in ([a-zA-Z\s]+?)(?:,| and|\.|$)", text, re.IGNORECASE)
    if live_match:
        location = live_match.group(1).strip().title()
        kg["entities"][location] = {"type": "location"}
        kg["attributes"]["location"] = location

        relation = {
            "source": "user",
            "relation": "lives_in",
            "target": location,
        }
        if relation not in kg["relations"]:
            kg["relations"].append(relation)

    brother_match = re.search(
        r"\bi have a brother named ([a-zA-Z\s]+?)(?:,| and|\.|$)",
        text,
        re.IGNORECASE,
    )
    if brother_match:
        brother = brother_match.group(1).strip().title()
        kg["entities"][brother] = {"type": "person"}

        relation = {
            "source": "user",
            "relation": "has_brother",
            "target": brother,
        }
        if relation not in kg["relations"]:
            kg["relations"].append(relation)

    return {
        **state,
        "knowledge_graph": kg,
        "current_stage": "graph_updated",
    }


def generate_response(state: ChatState) -> ChatState:
    kg = state["knowledge_graph"]

    system_prompt = f"""
You are a helpful assistant.

Current state: {state["current_stage"]}

Known entities:
{json.dumps(kg.get("entities", {}), indent=2)}

Known relations:
{json.dumps(kg.get("relations", []), indent=2)}

Known user attributes:
{json.dumps(kg.get("attributes", {}), indent=2)}

Use the stored facts when relevant.
Do not invent facts that are not in memory.
""".strip()

    response = safe_invoke([SystemMessage(content=system_prompt)] + state["messages"])

    return {
        **state,
        "messages": state["messages"] + [response],
        "current_stage": "response_generated",
    }


graph = StateGraph(ChatState)

graph.add_node("capture_user_input", capture_user_input)
graph.add_node("extract_and_update_graph", extract_and_update_graph)
graph.add_node("generate_response", generate_response)

graph.add_edge(START, "capture_user_input")
graph.add_edge("capture_user_input", "extract_and_update_graph")
graph.add_edge("extract_and_update_graph", "generate_response")
graph.add_edge("generate_response", END)

app = graph.compile()


def run_chat():
    state: ChatState = {
        "messages": [],
        "knowledge_graph": {
            "entities": {},
            "relations": [],
        },
        "last_user_input": "",
        "current_stage": "idle",
    }

    print("Chat started. Type 'quit' to exit.")

    while True:
        user_input = input("User: ").strip()

        if user_input.lower() == "quit":
            break

        state["messages"].append(HumanMessage(content=user_input))

        try:
            state = app.invoke(state)
        except Exception as e:
            print(f"Error: {e}")
            break

        print("Chatbot:", state["messages"][-1].content)
        print("Current stage:", state["current_stage"])
        print("Knowledge graph:", json.dumps(state["knowledge_graph"], indent=2))
        print()


if __name__ == "__main__":
    run_chat()
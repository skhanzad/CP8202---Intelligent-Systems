from email import message
from enum import Enum
from operator import ne
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_ollama import ChatOllama

class MemoryType(Enum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"

MEMORY_INFERENCE_PROMPT = """Classify the following user message into episode memory, semantic memory, both, or neither:"""

class ChatbotState(TypedDict):
    knowledge_graph: dict
    messages: list[BaseMessage]

chatbot_llm = ChatOllama(model="gemma2")
inference_llm = ChatOllama(model="gemma2")

def chatbot(state: ChatbotState) -> ChatbotState:
    response = chatbot_llm.invoke(state["messages"])
    return {"messages": state["messages"] + [response]}


def infer_memory_types(state: ChatbotState) -> ChatbotState:
    last_human_message = next((msg for msg in reversed(state["messages"]) if isinstance(msg, HumanMessage)), None)
    
    prompt = MEMORY_INFERENCE_PROMPT.format(message=last_human_message.content)

    response = inference_llm.invoke([HumanMessage(content=prompt)])

    raw = response.content.strip().lower()

    memory_type = MemoryType(raw) if raw in MemoryType._value2member_map_ else None

    return {**state, "memory_type": memory_type}

def add_to_knowledge_graph(state: ChatbotState) -> ChatbotState:
    updated_kg = state["knowledge_graph"]
    return {"messages": state["messages"], "knowledge_graph": updated_kg}


graph = (
    StateGraph(ChatbotState)
    .add_node("node", chatbot)
    .add_node
    .add_edge(START, "node")
    .add_edge("node", END)
)
app = graph.compile()

def run_chat_session():
    state = {"messages": [], "knowledge_graph": {}}
    print("Chat session started. Type 'quit' to exit.")
    while True:
        user_input = input("User: ")
        if user_input.lower() == "quit":
            break
        state["messages"].append(HumanMessage(content=user_input))
        state = app.invoke(state)
        print(f"Chatbot: {state['messages'][-1].content}")

if __name__ == "__main__":
    run_chat_session()
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_ollama import ChatOllama

class AgentState(TypedDict):
    knowledge_graph: dict
    messages: list[BaseMessage]

llm = ChatOllama(model="gemma2")

def chatbot(state: AgentState) -> AgentState:
    response = llm.invoke(state["messages"])
    return {"messages": state["messages"] + [response]}

graph = (
    StateGraph(AgentState)
    .add_node("node", chatbot)
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
import os

from typing import Annotated
from typing_extensions import TypedDict

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.postgres import PostgresSaver


LLM = AzureChatOpenAI(
    azure_endpoint=os.getenv("ENDPOINT_URL"),
    azure_deployment=os.getenv("DEPLOYMENT_NAME"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)


POSTGRES_PASS = os.getenv("POSTGRES_PASS")
POSTGRES_HOST = os.getenv("POSTGRES_HOST")
DB_URI = f"postgresql://langgraph:{POSTGRES_PASS}@{POSTGRES_HOST}:5432/langgraph"


class State(TypedDict):
    messages: Annotated[list, add_messages]


def chatbot(state: State) -> dict:
    return {"messages": [LLM.invoke(state["messages"])]}


def build_graph(checkpointer: PostgresSaver) -> CompiledStateGraph:
    graph_builder = StateGraph(State)
    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_edge("chatbot", END)
    return graph_builder.compile(checkpointer=checkpointer)


def get_ai_response(graph: CompiledStateGraph, config: dict, user_input: str) -> None:
    for event in graph.stream({"messages": [("user", user_input)]}, config, stream_mode="values"):
        for value in event.values():
            if isinstance(value[-1], AIMessage):
                print("Assistant:", value[-1].content)


def get_username() -> str:
    return input("Please enter your username to log in: ")


if __name__ == "__main__":
    with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
        checkpointer.setup()
        graph = build_graph(checkpointer)
        config = {"configurable": {"thread_id": get_username()}}
        print("=== CHATBOT ===")

        while True:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            get_ai_response(graph, config, user_input)

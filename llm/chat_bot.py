from typing import Annotated
import yaml
from typing_extensions import TypedDict
from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
import os
from langchain_openai import ChatOpenAI

with open("../config.yaml", "r") as f:
    config = yaml.safe_load(f)
os.environ["MODEL"]= config["model"]
os.environ["API_KEY"] = config["api_key"]
os.environ["BASE_URL"] = config["base_url"]
llm = ChatOpenAI(model = os.environ["MODEL"],api_key = os.environ["API_KEY"],base_url = os.environ["BASE_URL"],stream_usage=True)



class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)


def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}


graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile()

def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)

try:
    graph.get_graph().draw_mermaid_png(output_file_path="./chatbot_graph.png")
except Exception:
    # This requires some extra dependencies and is optional
    pass



if __name__ == "__main__":
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            stream_graph_updates(user_input)
        except:
            # fallback if input() is not available
            user_input = "What do you know about LangGraph?"
            print("User: " + user_input)
            stream_graph_updates(user_input)
            break
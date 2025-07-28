from langgraph.graph import StateGraph, START, END
from agents.warning import WarningAgent
from agents.prevention import PreventionAgent


def build_prevention_flow():
    builder = StateGraph(dict)

    builder.add_node("warning", WarningAgent().execute)
    builder.add_node("prevention", PreventionAgent().execute)

    builder.add_edge(START, "warning")
    builder.add_edge("warning", "prevention")
    builder.add_edge("prevention", END)

    return builder.compile()
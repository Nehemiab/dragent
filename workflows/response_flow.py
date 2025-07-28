from langgraph.graph import StateGraph, START, END
from agents.reporting import ReportingAgent
from agents.resource import ResourceAgent
from agents.assessment import AssessmentAgent
from agents.reconstruction import ReconstructionAgent

def build_response_flow():
    builder = StateGraph(dict)

    agents = {
        "reporting": ReportingAgent().execute,
        "resource": ResourceAgent().execute,
        "assessment": AssessmentAgent().execute,
        "reconstruction": ReconstructionAgent().execute
    }

    for name, node in agents.items():
        builder.add_node(name, node)

    builder.add_edge(START, "reporting")
    builder.add_edge("reporting", "resource")
    builder.add_edge("reporting", "assessment")
    builder.add_edge("reporting", "reconstruction")

    builder.add_edge("resource", END)
    builder.add_edge("assessment", END)
    builder.add_edge("reconstruction", END)

    return builder.compile()
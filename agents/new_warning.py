from typing import Annotated, Sequence, TypedDict, Literal
import operator
import functools

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode   # 官方预置工具节点

# 假设的 LLM 客户端
import llm.Client
llm = llm.Client.LLMClient()

# -------------------------------------------------
# 1. 状态定义
# -------------------------------------------------
class TyphoonAlertState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    # 用于记录最后一次是谁调用了工具，方便 router 返回
    sender: str


# -------------------------------------------------
# 2. 工具定义
# -------------------------------------------------
@tool
def typhoon_api(location: Annotated[str, "要查询台风信息的地区名称"]):
    """获取指定地区的实时台风数据"""
    return {
        "location": location,
        "typhoon_name": "台风山竹",
        "wind_speed": "35 m/s",
        "predicted_path": "西北方向移动",
        "arrival_time": "预计24小时后影响该地区",
        "rainfall_prediction": "24h累积200-300 mm"
    }


# 把工具统一放到 ToolNode
tool_node = ToolNode([typhoon_api])

# -------------------------------------------------
# 3. 代理提示模板
# -------------------------------------------------

# 台风分析代理（主脑）
analysis_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "你是台风灾害预警分析专家。工作流程：\n"
     "1. 先询问用户需要分析哪个地区。\n"
     "2. 可随时调用 typhoon_api 获取台风数据。\n"
     "3. 可向地形-水体专家（flood_agent）索要地形、山体、水体等信息。\n"
     "4. 当收集到足够信息后，向地形-水体专家说“够了”，并向用户输出完整风险评估、预防方案、疏散建议、物资清单。\n"
     "5. 若数据已充足，在最后一条消息中显式包含 **FINAL ANSWER** 字样结束流程。"),
    MessagesPlaceholder(variable_name="messages")
])
analysis_agent = analysis_prompt | llm.bind_tools([typhoon_api])

# 地形-水体专家（仅做问答，不调用工具）
flood_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "你是地形-水体数据专家。当台风分析专家向你询问某地的地形、山体、水体等信息时，"
     "请基于知识库给出尽可能详细的数据与建议，并继续对话，直到对方说“够了”。"),
    MessagesPlaceholder(variable_name="messages")
])
flood_agent = flood_prompt | llm

# -------------------------------------------------
# 4. 节点函数
# -------------------------------------------------

# ---------- analysis_node ----------
def analysis_node(state: TyphoonAlertState):
    raw = analysis_agent.invoke(state)
    if isinstance(raw, AIMessage):
        msg = raw
    else:
        msg = AIMessage(content=str(raw))
    msg.name = "analyst"          # 新增
    return {"messages": [msg], "sender": "analyst"}

# ---------- flood_node ----------
def flood_node(state: TyphoonAlertState):
    raw = flood_agent.invoke(state)
    if isinstance(raw, AIMessage):
        msg = raw
    else:
        msg = AIMessage(content=str(raw))
    msg.name = "flood"            # 新增
    return {"messages": [msg], "sender": "flood"}

# -------------------------------------------------
# 5. 条件路由函数
# -------------------------------------------------
def router(state: TyphoonAlertState) -> Literal["tool_node", "flood", "__end__", "continue"]:
    """
    决定下一步去哪里：
    - tool_calls -> tool_node
    - FINAL ANSWER -> __end__
    - 否则根据 sender 决定返回哪个节点继续对话
    """
    last_msg = state["messages"][-1]

    # 1. 如果有工具调用 -> 去 tool_node
    if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
        return "tool_node"

    # 2. 如果包含结束信号
    if "FINAL ANSWER" in str(last_msg.content):
        return "__end__"

    # 3. 普通文本对话：按 sender 回到对应节点
    sender = state.get("sender", "analyst")
    if sender == "analyst":
        return "flood"          # 分析节点刚发消息 -> 轮到 flood
    elif sender == "flood":
        return "analyst"        # flood 刚发消息 -> 轮到 analyst
    else:
        return "continue"       # 兜底

# -------------------------------------------------
# 6. 构建图
# -------------------------------------------------
workflow = StateGraph(TyphoonAlertState)

workflow.add_node("analyst", analysis_node)
workflow.add_node("flood", flood_node)
workflow.add_node("tool_node", tool_node)

# START -> analyst
workflow.add_edge(START, "analyst")

# analyst 的条件边
workflow.add_conditional_edges(
    "analyst",
    router,
    {
        "tool_node": "tool_node",
        "flood": "flood",
        "__end__": END,
        "continue": "analyst"  # 理论上不会走到这里
    }
)

# flood 的条件边
workflow.add_conditional_edges(
    "flood",
    router,
    {
        "analyst": "analyst",
        "__end__": END,
        "continue": "flood"
    }
)

# tool_node 的返回边：始终回到调用它的节点
workflow.add_edge("tool_node", "analyst")

# 编译
graph = workflow.compile()

# -------------------------------------------------
# 7. 运行入口
# -------------------------------------------------
def run_typhoon_alert(location: str):
    """运行台风预警系统"""
    initial = {
        "messages": [HumanMessage(content=location)],
        "sender": "analyst"
    }
    final_state = graph.invoke(initial)
    # 找最后一条来自 analyst 的消息作为结果
    for msg in reversed(final_state["messages"]):
        if isinstance(msg, AIMessage) and msg.name == "analyst":
            return msg.content
    return "未生成有效预警方案"


# -------------------------------------------------
# 8. 示例运行
# -------------------------------------------------
if __name__ == "__main__":
    location = "广东省梅州市"
    print(f"正在为 {location} 生成台风预警方案...\n")
    result = run_typhoon_alert(location)
    print("生成的预警方案：\n")
    print(result)
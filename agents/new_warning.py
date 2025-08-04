from typing import Annotated, Sequence, TypedDict, Literal, Tuple
import operator
import asyncio
import functools
import json
from datetime import datetime
from typing_extensions import NotRequired

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode   # 官方预置工具节点
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver #检查点
from openai import api_key, base_url

from dragent_tools.data_reader import typhoon_api as _real_typhoon_api

# 假设的 LLM 客户端
import llm.Client as Client
llm = Client.LLMClient()
Expert_llm=Client.LLMClient(api_key="token-abc123",base_url="http://localhost:8888/v1",model="/root/MiniCPM-o-2_6")
# -------------------------------------------------
# 1. 状态定义
# -------------------------------------------------
class TyphoonAlertState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    # 用于记录最后一次是谁调用了工具，方便 router 返回
    sender: str
    # 卫星图只给 flood 节点用，不进入 messages
    image: NotRequired[bytes]   # 新增：允许缺失


# -------------------------------------------------
# 2. 工具定义
# -------------------------------------------------
@tool
def typhoon_api(
    lat: Annotated[float, "纬度，保留一位小数"],
    lon: Annotated[float, "经度，保留一位小数"]
) -> dict:
    """根据经纬度获取台风实时数据"""
    # 组装成原来函数认识的 JSON 格式
    payload = json.dumps({
        "name": "typhoon_api",
        "arguments": {
            "latitude": lat,
            "longitude": lon,
            "time": datetime.utcnow().strftime('%Y-%m-%d %H:%M')
        }
    })
    return _real_typhoon_api(payload)


# 把工具统一放到 ToolNode
tool_node = ToolNode([typhoon_api])


def _make_image_message(image_bytes: bytes) -> HumanMessage:
    # 这里假设传进来的是 PNG/JPG 原始字节
    import base64
    b64 = base64.b64encode(image_bytes).decode()
    # 如果图片很大，可以改成 `data:image/jpeg;base64,` 等
    url = f"data:image/png;base64,{b64}"
    return HumanMessage(
        content=[
            {"type": "text", "text": "这是当地的卫星图，请据此分析山体、水体、地形等信息。"},
            {"type": "image_url", "image_url": {"url": url}}
        ]
    )

# -------------------------------------------------
# 3. 代理提示模板
# -------------------------------------------------

# 台风分析代理（主脑）
analysis_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "你是台风灾害预警分析专家。工作流程：\n"
     "1. 先询问用户需要分析哪个地区，并请用户给出具体经纬度（或你在对话中主动把地名解析成经纬度）。\n"
     "2. 调用 typhoon_api 时，请一定传入两个参数：lat（纬度）、lon（经度），均保留一位小数。\n"
     "3. 可向地形-水体专家索要地形、水体等信息。\n"
     "4. 收集足够信息后，对专家说“够了”，并向用户输出完整风险评估、预防方案、疏散建议、物资清单。\n"
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
flood_agent = flood_prompt | Expert_llm

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
    # 只在第一次进入 flood_node 时把图片塞进去
    image_msg = _make_image_message(state["image"])
    temp_messages = state["messages"] + [image_msg]

    raw = flood_agent.invoke({"messages": temp_messages})
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


# 7. 运行入口（改为 async + 使用检查点）
# ----------------------------------------------------------
async def run_typhoon_alert(location: str, satellite_image: bytes, thread_id: str = "thread-typhoon-1"):
    """异步运行台风预警系统，带检查点持久化"""
    initial = {
        "messages": [HumanMessage(content=location)],
        "sender": "analyst",
        "image": satellite_image
    }
# 使用 AsyncSqliteSaver，数据库文件 checkpoints.db 会自动创建
    async with AsyncSqliteSaver.from_conn_string("checkpoints.db") as memory:
        graph = workflow.compile(checkpointer=memory)
        try:
            graph.get_graph().draw_mermaid_png(output_file_path="./new_warning.png")
        except Exception:
            # This requires some extra dependencies and is optional
            pass
        final_state = await graph.ainvoke(
            initial,
            {"configurable": {"thread_id": thread_id}}
        )
        # 取出最后一条来自 analyst 的消息
        for msg in reversed(final_state["messages"]):
            if isinstance(msg, AIMessage) and msg.name == "analyst":
                return msg.content
        return "未生成有效预警方案"

# ----------------------------------------------------------
# 8. 示例运行（async main）
# ----------------------------------------------------------
async def main():
    location = "广东省梅州市"
    # 读一张本地卫星图做演示
    with open("satellite.png", "rb") as f:
        img_bytes = f.read()
    print(f"正在为 {location} 生成台风预警方案...\n")
    result = await run_typhoon_alert(location, img_bytes)
    print("生成的预警方案：\n")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
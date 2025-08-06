from __future__ import annotations

import asyncio
import functools
import json
import operator
from datetime import datetime
from typing import Annotated, Literal, Sequence, TypedDict, Tuple
from typing_extensions import NotRequired

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode


#  异步检查点（暂不启用）
# from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

#  LLM 客户端
import llm.Client as Client

llm = Client.LLMClient()
Expert_llm = Client.LLMClient(
    api_key="token-abc123",
    base_url="http://localhost:8888/v1",
    model="lora1",
)


#  1. 状态定义
class TyphoonAlertState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str  # 最后一次是谁调用了工具
    image: NotRequired[bytes]  # 卫星图，仅 flood 节点使用



#  2. 工具定义和工具节点
@tool
def typhoon_api(
    lat: Annotated[float, "纬度，保留一位小数"],
    lon: Annotated[float, "经度，保留一位小数"],
) -> dict:
    """根据经纬度获取台风实时数据"""
    from dragent_tools.data_reader import typhoon_api as _real_typhoon_api

    payload = json.dumps(
        {
            "name": "typhoon_api",
            "arguments": {
                "latitude": lat,
                "longitude": lon,
                "time": datetime.utcnow().strftime("%Y-%m-%d %H:%M"),
            },
        }
    )
    result = _real_typhoon_api(payload)
    print(f"[DEBUG] typhoon_api 返回：{result}")
    return result


# 预置工具节点
tool_node = ToolNode([typhoon_api])



#  3. 图片转为可传递的信息辅助函数
def _make_image_message(image_bytes: bytes) -> HumanMessage:
    """把二进制图片转成 HumanMessage"""
    import base64

    b64 = base64.b64encode(image_bytes).decode()
    url = f"data:image/png;base64,{b64}"
    return HumanMessage(
        content=[
            {"type": "text", "text": "这是当地的卫星图，请据此分析山体、水体、地形等信息。"},
            {"type": "image_url", "image_url": {"url": url}},
        ]
    )



#  4. 提示词的模板
# 台风分析代理（主脑）
analysis_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "你是台风灾害预警分析专家.你能使用以下工具：typhoon_api。\n"
            "你的助手flood是多模态大模型，能够识别是否存在水体以及周边情况的。\n"
            "在所以回答结束后，以FINAL ANSWER结尾，以便团队知道停止。\n"
            "请按照以下格式输出你的分析和对flood的询问：\n"
            "```query：向flood询问的问题\n"
            "analyses：你的分析```\n"
            "请你查询台风，向flood询问是否存在水体,请你在得到肯定回复后水体细节。\n"
            "之后，根据所提供的信息作出该地区风险评估、预防方案、疏散建议、所需物资。"
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
analysis_agent = analysis_prompt | llm.bind_tools([typhoon_api])

# 地形-水体专家
flood_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是water_analyst。一个负责分析水体情况的多模态大模型.你的任务是分析一张卫星图卫星图上面的水体以及周边的环境.\n"
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
flood_agent = flood_prompt | Expert_llm




#  5. 消息裁剪函数（防止爆上下文自己加的
def _extract_last_question(messages: Sequence[BaseMessage]) -> Sequence[BaseMessage]:
    """
    仅保留：
      - 最后一条 HumanMessage（用户问题）
      - 紧跟其后的 AIMessage（分析师追问，如果有）
    其余全部丢弃，节省 token
    """
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            if i + 1 < len(messages) and isinstance(messages[i + 1], AIMessage):
                return [messages[i], messages[i + 1]]
            return [messages[i]]
    # fallback
    return [HumanMessage(content="请基于卫星图描述当地地形、水体特征。")]



#  6. 两个智能体的节点函数
def analysis_node(state: TyphoonAlertState):
    raw = analysis_agent.invoke(state)
    msg = raw if isinstance(raw, AIMessage) else AIMessage(content=str(raw))
    msg.name = "analyst"
    print(f"[DEBUG] analysis_node 生成的消息：{msg.content}")
    return {"messages": [msg], "sender": "analyst"}


def flood_node(state: TyphoonAlertState):
    concise_msgs = _extract_last_question(state["messages"])
    print("↓↓↓↓ flood_node 收到的消息 ↓↓↓↓")
    for m in concise_msgs:
        print(f"[{type(m).__name__}] {m.content}")
    print("↑↑↑↑ 以上为 flood_node 收到的消息 ↑↑↑↑")

    # 仅在第一次进入 flood_node 时插入图片
    image_msg = _make_image_message(state["image"])
    temp_messages = concise_msgs + [image_msg]

    raw = flood_agent.invoke({"messages": temp_messages})
    msg = raw if isinstance(raw, AIMessage) else AIMessage(content=str(raw))
    msg.name = "flood"
    print(f"[DEBUG] flood_node 返回：{msg.content}")
    return {"messages": [msg], "sender": "flood"}



#  7. 条件路由
def router(state: TyphoonAlertState) -> Literal["tool_node", "flood", "__end__", "continue"]:
    last_msg = state["messages"][-1]

    if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
        next_node = "tool_node"
    elif "FINAL ANSWER" in str(last_msg.content):
        next_node = "__end__"
    else:
        sender = state.get("sender", "analyst")
        next_node = "flood" if sender == "analyst" else "analyst"
    print(f"[ROUTER] {state.get('sender')} → {next_node}")
    return next_node



#  8. 构建图
workflow = StateGraph(TyphoonAlertState)

workflow.add_node("analyst", analysis_node)
workflow.add_node("flood", flood_node)
workflow.add_node("tool_node", tool_node)

workflow.add_edge(START, "analyst")

workflow.add_conditional_edges(
    "analyst",
    router,
    {"tool_node": "tool_node", "flood": "flood", "__end__": END, "continue": "analyst"},
)

workflow.add_conditional_edges(
    "flood",
    router,
    {"analyst": "analyst", "__end__": END, "continue": "flood"},
)

workflow.add_edge("tool_node", "analyst")

# 编译图
new_warning = workflow.compile(name="new_warning")



# 9. 异步运行入口以及检查点文件输出（暂不启用）
"""
async def run_typhoon_alert(location: str, satellite_image: bytes, thread_id: str = "thread-typhoon-1"):
    initial = {
        "messages": [HumanMessage(content=location)],
        "sender": "analyst",
        "image": satellite_image,
    }

    async with AsyncSqliteSaver.from_conn_string("checkpoints.db") as memory:
        graph = workflow.compile(checkpointer=memory)
        try:
            graph.get_graph().draw_mermaid_png(output_file_path="./new_warning.png")
        except Exception:
            pass

        final_state = await graph.ainvoke(
            initial,
            {"configurable": {"thread_id": thread_id}},
        )
        for msg in reversed(final_state["messages"]):
            if isinstance(msg, AIMessage) and msg.name == "analyst":
                return msg.content
        return "未生成有效预警方案"


async def main():
    location_name = "广东省梅州市"
    lat, lon = 24.3, 116.1
    location = f"{location_name}（纬度 {lat:.1f}，经度 {lon:.1f}）"
    with open("demo_picture.png", "rb") as f:
        img_bytes = f.read()
    result = await run_typhoon_alert(location, img_bytes)
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
"""

graph=workflow.compile()
events=graph.stream(
    {
        "messages": [HumanMessage(content="广东省梅州市（纬度 24.3，经度 116.1）")],
        "sender": "analyst",
        "image": open("demo_picture.png", "rb").read()
    },
    {"recursion_limit": 10},
)
for event in events:
    print(event)
    print("-----")
#if __name__ == "__main__":
#    asyncio.run(main())
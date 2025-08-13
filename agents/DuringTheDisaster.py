from __future__ import annotations

import asyncio
import functools
import json
import operator
import re
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
    model="lora3",
)


#  1. 状态定义
class TyphoonAlertState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str  # 最后一次是谁调用了工具
    image: NotRequired[bytes]  # 卫星图，仅 road 节点使用



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
                "time":"2024-11-10 20:00",
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
            "system",
            "你是台风灾中资源调度与强险方案专家，可按以下步骤工作：\n"
            "1. 调用 typhoon_api 获取台风实时数据；\n"
            "2. 将结果发给 damage 助手，让其评估道路损毁；\n"
            "3. 收到 damage 返回后，整合所有信息，一次性输出：\n"
            "   - 地区风险评估\n"
            "   - 强险方案\n"
            "   - 所需物资清单\n"
            "   - 资源调度细节\n"
            "【输出规范】\n"
            "- 正文结束后，必须另起一行，仅写：\n"
            "FINAL ANSWER\n"
            "- 正文任何地方都不得提前出现这四个字。"
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
analysis_agent = analysis_prompt | llm.bind_tools([typhoon_api])

# 道路专家
road_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是road_analyst。一个负责分析道路情况的多模态大模型.你的任务是分析一张卫星图上面的道路以及周边的环境.\n"
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
road_agent = road_prompt | Expert_llm




#  5. 消息裁剪函数（防止爆上下文自己加的
def _extract_last_question(messages: Sequence[BaseMessage]) -> Sequence[BaseMessage]:
    last_content = messages[-1].content or ""
    match = re.search(r"query：'(.*?)'\\n```", last_content, re.IGNORECASE | re.DOTALL)
    return [HumanMessage(content=match.group(1).strip() if match else "")]


#  6. 两个智能体的节点函数
def analysis_node(state: TyphoonAlertState):
    raw = analysis_agent.invoke(state)
    msg = raw if isinstance(raw, AIMessage) else AIMessage(content=str(raw))
    msg.name = "analyst"
    print(f"[DEBUG] analysis_node 生成的消息：{msg.content}")
    return {"messages": [msg], "sender": "analyst"}


def road_node(state: TyphoonAlertState):
    concise_msgs = _extract_last_question(state["messages"])
    print("↓↓↓↓ road_node 收到的消息 ↓↓↓↓")
    for m in concise_msgs:
        print(f"[{type(m).__name__}] {m.content}")
    print("↑↑↑↑ 以上为 road_node 收到的消息 ↑↑↑↑")

    # 仅在第一次进入 road_node 时插入图片
    image_msg = _make_image_message(state["image"])
    temp_messages = concise_msgs + [image_msg]

    raw = road_agent.invoke({"messages": temp_messages})
    msg = raw if isinstance(raw, AIMessage) else AIMessage(content=str(raw))
    msg.name = "road"
    print(f"[DEBUG] road_node 返回：{msg.content}")
    return {"messages": [msg], "sender": "road"}


#  7. 条件路由
def router(state: TyphoonAlertState) -> str:
    last_msg = state["messages"][-1]

    # 1. 工具调用 -> 继续
    if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
        return "tool_node"

    # 2. 出现终止标记 -> 结束
    if "FINAL ANSWER" in str(last_msg.content).upper():
        return "__end__"

    # 3. 如果上一轮的 sender 是 road 且没有更多请求 -> 结束
    if state.get("sender") == "road":
        return "__end__"

    # 4. 否则继续
    return "road"


#  8. 构建图
workflow = StateGraph(TyphoonAlertState)
#创建节点
workflow.add_node("analyst", analysis_node)
workflow.add_node("road", road_node)
workflow.add_node("tool_node", tool_node)

workflow.add_edge(START, "analyst")
workflow.add_edge("tool_node", "analyst")

workflow.add_conditional_edges(
    "analyst",
    router,
    {"tool_node": "tool_node", "road": "road", "__end__": END},
)

workflow.add_conditional_edges(
    "road",
    router,
    {"analyst": "analyst", "__end__": END},
)



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
        "image": open("../demo_picture.png", "rb").read()
    },
    {"recursion_limit": 10},
)
for event in events:
    print(event)
    print("-----")
#if __name__ == "__main__":
#    asyncio.run(main())
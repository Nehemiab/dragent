from __future__ import annotations
import asyncio
import json
import operator
import re
from typing import Annotated, Sequence, TypedDict
from typing_extensions import NotRequired
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage,ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

#  LLM 客户端
import llm.Client as Client

llm = Client.LLMClient()
Expert_llm = Client.LLMClient(
    api_key="token-abc123",
    base_url="http://localhost:1234/v1",
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
            "你是台风灾害预警分析专家，可以使用的工具是：typhoon_api。\n"
            "你的助手flood是多模态大模型，可以回答图中是否存在水体以及水体周边的情况。\n"
            "你的任务：查询台风数据，向flood询问是否存在水体,并询问水体细节。\n"
            "在流程的最后，根据所提供的信息作出该地区风险评估、预防方案、疏散建议、所需物资。"
            "请按照以下格式输出对flood的询问和你的分析：\n"
            "query：'向flood询问的问题'\n"
            "```analyses：你的分析```\n"
            "所有人回答完成后，以FINAL ANSWER结尾，让团队知道停止。\n"
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
            "你是water_analyst。一个负责分析水体情况的多模态大模型\n"
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
flood_agent = flood_prompt | Expert_llm




#  5. 消息裁剪函数（防止爆上下文自己加的
def _extract_last_question(messages: Sequence[BaseMessage]) -> Sequence[BaseMessage]:
    last_content = messages[-1].content or ""
    match = re.search(r"query：'(.*?)'", last_content, re.IGNORECASE | re.DOTALL)
    if match is None:
        return [HumanMessage(content="图中是否存在水体")]
    return [HumanMessage(content=match.group(1).strip() if match else "")]


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
def router(state: TyphoonAlertState) -> str:
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
#创建节点
workflow.add_node("analyst", analysis_node)
workflow.add_node("flood", flood_node)
workflow.add_node("tool_node", tool_node)

workflow.add_edge(START, "analyst")
workflow.add_edge("tool_node", "analyst")
workflow.add_edge("flood", "analyst")

workflow.add_conditional_edges(
    "analyst",
    router,
    {"tool_node": "tool_node", "flood": "flood", "__end__": END},
)




# 9. 异步运行入口以及检查点文件输出
async def main():

    location = '广东省梅州市（纬度 24.3，经度 116.1）'

    # 读入卫星云图
    with open("../demo_picture.png", "rb") as f:
        img_bytes = f.read()

    # 初始化对话状态
    initial = {
        "messages": [HumanMessage(content=location)],
        "sender": "analyst",
        "image": img_bytes,
    }

    # 运行工作流
    async with AsyncSqliteSaver.from_conn_string("checkpoints.db") as memory:
        graph = workflow.compile(checkpointer=memory)
        '''
        try:
            graph.get_graph().draw_mermaid_png(output_file_path="./new_warning.png")
        except Exception as e:
            print(e)
        '''
        final_state = await graph.ainvoke(
            initial,
            {"configurable": {"thread_id": "thread-typhoon-1"},"recursion_limit": 10},
        )

        # 取回分析师节点的最终结果
        for msg in reversed(final_state["messages"]):
            if isinstance(msg, AIMessage) and msg.name == "analyst":
                return msg.content
        return "未生成有效预警方案"
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
"""
if __name__ == "__main__":
    asyncio.run(main())
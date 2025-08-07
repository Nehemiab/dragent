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
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver


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



# 1. 状态定义
class PostDisasterState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str
    image: NotRequired[bytes]   # 灾后卫星/航拍图


#  2. 提示词的模板
# 灾后重建方案分析代理（主脑）
rebuild_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "你是灾后重建规划专家。 \n"
     "你的助手damage是多模态大模型，可以向他询问图中房屋、道路等基础设施损毁情况的。\n"
     "等助手回答结束后，以FINAL ANSWER结尾，让团队知道停止。\n"
     "请按照以下格式输出你的分析和对flood的询问：\n"
     "```query：'向flood询问的问题'\n```"
     "```analyses：你的分析```\n"
     "向damage询问是否存在损坏房屋、道路情况后,请你在得到肯定回复后询问损坏细节。\n"
     "之后，根据所提供的信息作出该地区完整灾后重建方案：优先修复顺序、资源调度、人员安置、预算估算、时间表等。\n"),
    MessagesPlaceholder(variable_name="messages")
])
analysis_agent = rebuild_prompt | llm     # 不绑定任何工具

# 房屋道路损坏识别与评估专家（仅做问答）
damage_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "你是damage_analyst。一个负责分析房屋受损情况的多模态大模型，你的任务是分析一张卫星图上面的损毁房屋以及损毁情况.\n"),
    MessagesPlaceholder(variable_name="messages")
])
damage_agent = damage_prompt | Expert_llm


#  3. 图片转为可传递的信息辅助函数
def _make_image_message(image_bytes: bytes) -> HumanMessage:
    """把二进制图片转成 HumanMessage"""
    import base64

    b64 = base64.b64encode(image_bytes).decode()
    url = f"data:image/png;base64,{b64}"
    return HumanMessage(
        content=[
            {"type": "text", "text": "这是灾后的卫星/航拍图，请据此识别房屋、道路损毁情况。"},
            {"type": "image_url", "image_url": {"url": url}}
        ]
    )


#  4. 消息裁剪函数（防止爆上下文自己加的）
def _extract_last_question(messages: Sequence[BaseMessage]) -> Sequence[BaseMessage]:
    last_content = messages[-1].content or ""
    match = re.search(r"query：'(.*?)'\\n```", last_content, re.IGNORECASE | re.DOTALL)
    return [HumanMessage(content=match.group(1).strip() if match else "")]


#  5. 两个智能体的节点函数
def analysis_node(state: PostDisasterState):
    raw = analysis_agent.invoke(state)
    msg = raw if isinstance(raw, AIMessage) else AIMessage(content=str(raw))
    msg.name = "analyst"
    print(f"[DEBUG] analysis_node 生成的消息：{msg.content}")
    return {"messages": [msg], "sender": "analyst"}



def damage_node(state: PostDisasterState):
    concise_msgs = _extract_last_question(state["messages"])
    print("↓↓↓↓ damage_node 收到的消息 ↓↓↓↓")
    for m in concise_msgs:
        print(f"[{type(m).__name__}] {m.content}")
    print("↑↑↑↑ 以上为 damage_node 收到的消息 ↑↑↑↑")

    # 仅在第一次进入 flood_node 时插入图片
    image_msg = _make_image_message(state["image"])
    temp_messages = concise_msgs + [image_msg]

    raw = damage_agent.invoke({"messages": temp_messages})
    msg = raw if isinstance(raw, AIMessage) else AIMessage(content=str(raw))
    msg.name = "damage"
    print(f"[DEBUG] damage_node 返回：{msg.content}")
    return {"messages": [msg], "sender": "damage"}


#  6. 条件路由
def router(state: PostDisasterState) -> str:
    last_msg = state["messages"][-1]

    if "FINAL ANSWER" in str(last_msg.content):
        next_node = "__end__"

    else:
        sender = state.get("sender", "analyst")
        next_node = "damage" if sender == "analyst" else "analyst"
    print(f"[ROUTER] {state.get('sender')} → {next_node}")
    return next_node


#  7. 构建图
workflow = StateGraph(PostDisasterState)
#创建节点
workflow.add_node("analyst", analysis_node)
workflow.add_node("damage", damage_node)

workflow.add_edge(START, "analyst")

workflow.add_conditional_edges(
    "analyst",
    router,
    {"damage": "damage", "__end__": END},
)

workflow.add_conditional_edges(
    "damage",
    router,
    {"analyst": "analyst", "__end__": END},
)

# 9. 异步运行入口以及检查点文件输出
async def main():

    location = '广东省梅州市（纬度 24.3，经度 116.1）'

    # 读入卫星云图
    with open("damage_picture.jpg", "rb") as f:
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
            {"configurable": {"thread_id": "thread-damage-1"},"recursion_limit": 10},
        )

        # 取回分析师节点的最终结果
        for msg in reversed(final_state["messages"]):
            if isinstance(msg, AIMessage) and msg.name == "analyst":
                return msg.content
        return "未生成有效预警方案"


# 8. 异步运行入口以及检查点文件输出（暂不启用）
"""
async def run_post_disaster_plan(location: str, post_image: bytes, thread_id: str = "thread-rebuild-1"):
    initial = {
        "messages": [HumanMessage(content=location)],
        "sender": "analyst",
        "image": post_image
    }
    async with AsyncSqliteSaver.from_conn_string("checkpoints.db") as memory:
        graph = workflow.compile(checkpointer=memory)
        try:
            graph.get_graph().draw_mermaid_png(output_file_path="./post_disaster.png")
        except Exception:
            pass

        final_state = await graph.ainvoke(
            initial,
            {"configurable": {"thread_id": thread_id}}
        )
        for msg in reversed(final_state["messages"]):
            if isinstance(msg, AIMessage) and msg.name == "analyst":
                return msg.content
        return "未生成有效重建方案"


# -------------------------------------------------
# 8. 示例运行
# -------------------------------------------------
async def main():
    location = "广东省梅州市"
    with open("post_disaster.png", "rb") as f:
        img_bytes = f.read()
    print(f"正在为 {location} 生成灾后重建方案...\n")
    result = await run_post_disaster_plan(location, img_bytes)
    print("生成的灾后重建方案：\n")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())



graph=workflow.compile()
events=graph.stream(
    {
        "messages": [HumanMessage(content="广东省梅州市（纬度 24.3，经度 116.1）")],
        "sender": "analyst",
        "image": open("../damage_picture.jpg", "rb").read()
    },
    {"recursion_limit": 10},
)
for event in events:
    print(event)
    print("-----")
"""
if __name__ == "__main__":
    asyncio.run(main())
from typing import Annotated, Sequence, TypedDict, Literal
import operator
import asyncio
import functools
import base64
from datetime import datetime
from typing_extensions import NotRequired

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

# LLM 客户端
import llm.Client
llm = llm.Client.LLMClient()
Expert_llm = llm.Client.LLMClient(
    api_key="token-abc123",
    base_url="http://localhost:8888/v1",
    model="/root/MiniCPM-o-2_6"
)

# -------------------------------------------------
# 1. 状态定义
# -------------------------------------------------
class PostDisasterState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str
    image: NotRequired[bytes]   # 灾后卫星/航拍图


# -------------------------------------------------
# 3. 代理提示模板
# -------------------------------------------------

# 灾后重建方案分析代理（主脑）
rebuild_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "你是灾后重建规划专家。工作流程：\n"
     "1. 先询问用户需要分析哪个受灾地区，并请用户给出具体经纬度（或你在对话中把地名解析成经纬度）。\n"
     "2. 可向“房屋道路损坏识别专家”索要受灾区域内的损毁房屋、道路位置及损毁程度等信息。\n"
     "3. 收集足够信息后，对专家说“够了”，并向用户输出完整灾后重建方案：优先修复顺序、资源调度、人员安置、预算估算、时间表等。\n"
     "4. 若数据已充足，在最后一条消息中显式包含 **FINAL ANSWER** 字样结束流程。"),
    MessagesPlaceholder(variable_name="messages")
])
analysis_agent = rebuild_prompt | llm     # 不绑定任何工具

# 房屋道路损坏识别与评估专家（仅做问答）
damage_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "你是房屋道路损坏识别与评估专家。当灾后重建专家向你询问某地的房屋、道路等基础设施损毁情况时，"
     "请基于知识库及提供的灾后影像，给出尽可能详细的损毁位置、类型、损毁等级（轻微/中等/严重/完全毁坏）及建议优先修复顺序，"
     "并继续对话，直到对方说“够了”。"),
    MessagesPlaceholder(variable_name="messages")
])
damage_agent = damage_prompt | Expert_llm


# -------------------------------------------------
# 4. 节点函数
# -------------------------------------------------

def _make_image_message(image_bytes: bytes) -> HumanMessage:
    b64 = base64.b64encode(image_bytes).decode()
    url = f"data:image/png;base64,{b64}"
    return HumanMessage(
        content=[
            {"type": "text", "text": "这是灾后的卫星/航拍图，请据此识别房屋、道路损毁情况。"},
            {"type": "image_url", "image_url": {"url": url}}
        ]
    )


# ---------- analysis_node ----------
def analysis_node(state: PostDisasterState):
    raw = analysis_agent.invoke(state)
    msg = raw if isinstance(raw, AIMessage) else AIMessage(content=str(raw))
    msg.name = "analyst"
    return {"messages": [msg], "sender": "analyst"}


# ---------- damage_node ----------
def damage_node(state: PostDisasterState):
    image_msg = _make_image_message(state["image"])
    temp_messages = state["messages"] + [image_msg]

    raw = damage_agent.invoke({"messages": temp_messages})
    msg = raw if isinstance(raw, AIMessage) else AIMessage(content=str(raw))
    msg.name = "damage"
    return {"messages": [msg], "sender": "damage"}


# -------------------------------------------------
# 5. 条件路由函数
# -------------------------------------------------
def router(state: PostDisasterState) -> Literal["__end__", "continue"]:
    last_msg = state["messages"][-1]

    # 1. 结束信号
    if "FINAL ANSWER" in str(last_msg.content):
        return "__end__"

    # 2. 根据 sender 决定下一步
    sender = state.get("sender", "analyst")
    if sender == "analyst":
        return "damage"
    elif sender == "damage":
        return "analyst"
    else:
        return "continue"


# -------------------------------------------------
# 6. 构建图
# -------------------------------------------------
workflow = StateGraph(PostDisasterState)

workflow.add_node("analyst", analysis_node)
workflow.add_node("damage", damage_node)

workflow.add_edge(START, "analyst")

workflow.add_conditional_edges(
    "analyst",
    router,
    {
        "damage": "damage",
        "__end__": END,
        "continue": "analyst"
    }
)

workflow.add_conditional_edges(
    "damage",
    router,
    {
        "analyst": "analyst",
        "__end__": END,
        "continue": "damage"
    }
)


# -------------------------------------------------
# 7. 运行入口
# -------------------------------------------------
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
from typing import Annotated, Sequence, TypedDict, Literal
import operator
import asyncio
import base64
import time
import uuid
import logging
from typing_extensions import NotRequired

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ------------------- LLM 客户端 -------------------
import llm.Client as Client
llm = Client.LLMClient()
Expert_llm = Client.LLMClient(
    api_key="token-abc123",
    base_url="http://localhost:8888/v1",
    model="lora3"
)

# ------------------- 状态结构 -------------------
class PostDisasterState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str
    image: NotRequired[bytes]

# ------------------- Prompt 模板 -------------------
rebuild_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "你是中国应急管理部授权的台风/洪灾应急指挥专家，正在组织灾中应急资源调度方案。\n"
        "你将基于专家智能体提供的图像评估信息，制定一份面向实际执行的《灾中资源调度与抢险方案》，必须涵盖以下内容：\n\n"
        "1. 【灾情概况】：明确受灾地点（精确到区/镇）、灾害类型、受损等级与范围；\n"
        "2. 【紧急资源调度】：饮用水、食品、药品、医疗的调拨计划（数量、类型、优先级）；\n"
        "3. 【道路抢通与布控】：主干通道抢通计划、临时路线与布控点；\n"
        "4. 【救援队与装备部署】：各类救援队伍与装备的布点；\n"
        "5. 【避难安置规划】：临时安置点位置、容量与物资；\n"
        "6. 【通信与指挥体系】：中继站与指挥链路；\n"
        "7. 【重点风险控制】：次生灾害点的监测响应机制；\n"
        "8. 【时间节点与责任分工】：各项任务的时间与负责人。\n\n"
        "**最后以【最终方案】六个字开头输出完整内容，禁止输出多个版本。**"
    )),
    MessagesPlaceholder(variable_name="messages")
])

damage_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "你是中国遥感中心的灾害图像评估专家，负责识别台风引发的洪水、塌方、道路中断、建筑受损等情况。\n"
        "请逐条列出图像中识别出的每处关键灾损，建议按以下结构：\n"
        "1. 【类型】：如道路阻断/桥梁垮塌/堤坝溃决等；\n"
        "2. 【位置】：精确坐标或地名；\n"
        "3. 【影响】：完全不可通行/部分塌方/群众被困等；\n"
        "4. 【建议】：是否需立即抢通、绕行建议等。\n"
        "如主脑智能体提出问题，请优先回复。"
    )),
    MessagesPlaceholder(variable_name="messages")
])

analysis_agent = rebuild_prompt | llm
damage_agent = damage_prompt | Expert_llm

# ------------------- 图像消息封装 -------------------
def _make_image_message(img_bytes: bytes) -> HumanMessage:
    return HumanMessage(content=[{
        "type": "image_url",
        "image_url": {
            "url": "data:image/png;base64," + base64.b64encode(img_bytes).decode(),
            "detail": "high"
        }
    }])

# ------------------- 节点函数 -------------------
async def analysis_node(state: PostDisasterState):
    try:
        raw = await asyncio.wait_for(analysis_agent.ainvoke(state), timeout=60)
    except asyncio.TimeoutError:
        raw = AIMessage(content="[超时] 主脑响应超时，请检查服务状态。")
    except Exception as e:
        raw = AIMessage(content=f"[异常] 主脑响应失败：{str(e)}")
    msg = raw if isinstance(raw, AIMessage) else AIMessage(content=str(raw))
    msg.name = "主脑"
    msg.content = msg.content.strip()
    return {"messages": [msg], "sender": "主脑"}

async def damage_node(state: PostDisasterState):
    already_sent = any(
        isinstance(m, HumanMessage) and isinstance(m.content, list)
        and any(isinstance(c, dict) and c.get("type") == "image_url" for c in m.content)
        for m in state["messages"]
    )
    temp = state["messages"] if already_sent else state["messages"] + [_make_image_message(state["image"])]

    try:
        raw = await asyncio.wait_for(damage_agent.ainvoke({"messages": temp}), timeout=60)
    except asyncio.TimeoutError:
        raw = AIMessage(content="[超时] 专家响应超时，请检查服务状态。")
    except Exception as e:
        raw = AIMessage(content=f"[异常] 专家响应失败：{str(e)}")
    msg = raw if isinstance(raw, AIMessage) else AIMessage(content=str(raw))
    msg.name = "专家"
    msg.content = msg.content.strip()
    return {"messages": [msg], "sender": "专家"}

# ------------------- 路由函数 -------------------
MAX_TURN = 20

def router(state: PostDisasterState) -> Literal["__end__", "主脑", "专家"]:
    last_msg = str(state["messages"][-1].content)
    if last_msg.startswith("【最终方案】") or len(state["messages"]) >= MAX_TURN:
        return "__end__"
    return "专家" if state["sender"] == "主脑" else "主脑"

# ------------------- 构建 LangGraph -------------------
workflow = StateGraph(PostDisasterState)
workflow.add_node("主脑", analysis_node)
workflow.add_node("专家", damage_node)
workflow.add_edge(START, "主脑")
workflow.add_conditional_edges("主脑", router, {"专家": "专家", "__end__": END})
workflow.add_conditional_edges("专家", router, {"主脑": "主脑", "__end__": END})

# ------------------- 主流程入口 -------------------
async def run_post_disaster_plan(location: str, post_image: bytes, thread_id: str = None):
    thread_id = thread_id or f"thread-{uuid.uuid4()}"
    initial = {
        "messages": [HumanMessage(content=location)],
        "sender": "主脑",
        "image": post_image
    }

    async with AsyncSqliteSaver.from_conn_string("checkpoints.db") as memory:
        graph = workflow.compile(checkpointer=memory)
        final_state = await graph.ainvoke(initial, {"configurable": {"thread_id": thread_id}})

        print("\n=== 智能体对话记录 ===")
        for idx, msg in enumerate(final_state["messages"], start=1):
            if isinstance(msg, (HumanMessage, AIMessage)):
                role = msg.name or "用户"
                label = {
                    "主脑": "主脑智能体",
                    "专家": "专家智能体",
                    "用户": "用户输入"
                }.get(role, role)
                print(f"\n【轮次 {idx} - {label}】\n{msg.content.strip()}")

        for msg in reversed(final_state["messages"]):
            if isinstance(msg, AIMessage) and msg.name == "主脑" and msg.content.strip().startswith("【最终方案】"):
                print("[final answer]\n" + msg.content.strip())
                return "[final answer]\n" + msg.content.strip()

        print("未生成有效重建方案")
        return "未生成有效重建方案"

# ------------------- 本地测试 -------------------
async def main():
    location = "广东省梅州市，中心坐标 24.3°N, 116.1°E，请基于灾后航拍图输出资源调度方案。"
    with open("post_disaster.png", "rb") as f:
        img_bytes = f.read()
    print("正在生成灾中资源调度方案...\n")
    await run_post_disaster_plan(location, img_bytes)

if __name__ == "__main__":
    asyncio.run(main())

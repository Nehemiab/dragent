from typing import Annotated, Sequence, TypedDict, Literal
import operator
import asyncio
import base64
import time
from typing_extensions import NotRequired
import logging

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

# 设置日志输出格式
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# -------------------------------------------------
# LLM 客户端初始化（普通模型与专家模型）
# -------------------------------------------------
import llm.Client as Client
llm = Client.LLMClient()  # 普通模型：用于生成调度方案
Expert_llm = Client.LLMClient(
    api_key="token-abc123",
    base_url="http://localhost:8888/v1",
    model="lora3"  # 专家模型：用于图像评估
)

# -------------------------------------------------
# 1. 定义 LangGraph 中传递的状态结构
# -------------------------------------------------
class PostDisasterState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]  # 对话记录累加
    sender: str  # 当前角色
    image: NotRequired[bytes]  # 灾中图像（二进制）

# -------------------------------------------------
# 2. 提示词模板：两个智能代理
# -------------------------------------------------

# 主脑分析任务
rebuild_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "你是中国应急管理部授权的台风/洪灾应急指挥专家...\n"
        "必须包含：\n"
        "1. 受灾地区和坐标\n"
        "2. 道路通行情况\n"
        "3. 人口密集点\n"
        "4. 可调资源\n"
        "5. 次生灾害风险\n"
        "6. 救援优先级\n"
        "7. 物资路线表\n"
        "8. 支援点布设建议\n"
        "最后以【最终方案】开头输出完整方案，不要英文术语"
    )),
    MessagesPlaceholder(variable_name="messages")
])

# 图像分析任务
damage_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "你是中国遥感中心灾害评估专家...\n"
        "请逐条列出房屋、道路、桥梁的损毁情况及通行建议，必须是纯中文"
    )),
    MessagesPlaceholder(variable_name="messages")
])

# 绑定模型与提示词
analysis_agent = rebuild_prompt | llm
damage_agent = damage_prompt | Expert_llm

# -------------------------------------------------
# 3. 图像转 LangChain 消息格式
# -------------------------------------------------
def _make_image_message(img_bytes: bytes) -> HumanMessage:
    return HumanMessage(content=[
        {
            "type": "image_url",
            "image_url": {
                "url": "data:image/png;base64," + base64.b64encode(img_bytes).decode(),
                "detail": "high"
            }
        }
    ])

# -------------------------------------------------
# 4. 主脑分析节点（异步）
# -------------------------------------------------
async def analysis_node(state: PostDisasterState):
    logging.info(">>> analysis_node start")
    t0 = time.time()
    try:
        raw = await asyncio.wait_for(
            asyncio.to_thread(analysis_agent.invoke, state),
            timeout=15
        )
    except asyncio.TimeoutError:
        logging.warning("analysis_node timeout")
        raw = AIMessage(content="**FINAL ANSWER**\n[analysis_node 超时，请检查 LLM 服务]")
    except Exception as e:
        logging.exception("analysis_node 异常")
        raw = AIMessage(content=f"**FINAL ANSWER**\n[analysis_node 异常：{str(e)}]")
    logging.info("analysis_node done in %.2fs", time.time() - t0)

    msg = raw if isinstance(raw, AIMessage) else AIMessage(content=str(raw))
    msg.name = "analyst"
    return {"messages": [msg], "sender": "analyst"}

# -------------------------------------------------
# 5. 图像损毁节点（异步）
# -------------------------------------------------
async def damage_node(state: PostDisasterState):
    logging.info(">>> damage_node start")
    t0 = time.time()

    already_sent_image = any(
        isinstance(m, HumanMessage) and any(
            isinstance(c, dict) and c.get("type") == "image_url"
            for c in (m.content if isinstance(m.content, list) else [])
        )
        for m in state["messages"]
    )
    temp_messages = state["messages"] if already_sent_image else state["messages"] + [_make_image_message(state["image"])]

    try:
        raw = await asyncio.wait_for(
            asyncio.to_thread(damage_agent.invoke, {"messages": temp_messages}),
            timeout=15
        )
    except asyncio.TimeoutError:
        logging.warning("damage_node timeout")
        raw = AIMessage(content="**FINAL ANSWER**\n[damage_node 超时，请检查 Expert_llm 服务]")
    except Exception as e:
        logging.exception("damage_node 异常")
        raw = AIMessage(content=f"**FINAL ANSWER**\n[damage_node 异常：{str(e)}]")
    logging.info("damage_node done in %.2fs", time.time() - t0)

    msg = raw if isinstance(raw, AIMessage) else AIMessage(content=str(raw))
    msg.name = "damage"
    return {"messages": [msg], "sender": "damage"}

# -------------------------------------------------
# 6. 路由函数：决定流程是否结束
# -------------------------------------------------
MAX_TURN = 20

def router(state: PostDisasterState) -> Literal["__end__", "continue"]:
    last_msg = str(state["messages"][-1].content)
    if "FINAL ANSWER" in last_msg or len(state["messages"]) >= MAX_TURN:
        return "__end__"
    return "damage" if state.get("sender") == "analyst" else "analyst"

# -------------------------------------------------
# 7. 构建 LangGraph 图结构
# -------------------------------------------------
workflow = StateGraph(PostDisasterState)
workflow.add_node("analyst", analysis_node)
workflow.add_node("damage", damage_node)
workflow.add_edge(START, "analyst")
workflow.add_conditional_edges("analyst", router, {"damage": "damage", "__end__": END})
workflow.add_conditional_edges("damage", router, {"analyst": "analyst", "__end__": END})

# -------------------------------------------------
# 8. 运行图流程主函数
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

        logging.info(">> 启动图执行流程")
        final_state = await graph.ainvoke(
            initial,
            {"configurable": {"thread_id": thread_id}}
        )
        logging.info(">> 图执行完成")

        for msg in reversed(final_state["messages"]):
            if isinstance(msg, AIMessage) and msg.name == "analyst":
                return msg.content

        return "未生成有效重建方案"

# -------------------------------------------------
# 9. 示例运行入口
# -------------------------------------------------
async def main():
    location = "广东省梅州市，中心坐标 24.3°N, 116.1°E，请基于附件灾后航拍图直接输出 FINAL ANSWER 的《灾中资源调度方案》。"

    with open("post_disaster.png", "rb") as f:
        img_bytes = f.read()

    print("正在为梅州市生成灾中资源调度方案...\n")
    result = await run_post_disaster_plan(location, img_bytes)
    print("生成结果：\n", result)

if __name__ == "__main__":
    asyncio.run(main())

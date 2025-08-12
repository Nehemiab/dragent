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
import base64

#  LLM 客户端
import llm.Client as Client

llm = Client.LLMClient()
Building_llm = Client.LLMClient(api_key="token-abc123",base_url="http://localhost:8888/v1",model="lora2")
Road_llm = Client.LLMClient(api_key="token-abc123",base_url="http://localhost:8888/v1",model="lora3")



# 1. 状态定义
class PostDisasterState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str
    image: bytes   # 灾后卫星/航拍图（带框图，给building）
    raw_image: bytes  # 灾后卫星/航拍图（原图，给 road）
    counter: int  # <- 新增计数器
    labeled_image: bytes # 可保留，也可不用


#  2. 提示词的模板
# 灾后重建方案分析代理（主脑）
# 2. 分阶段提示词列表
ANALYST_PROMPTS = [
    "你是我的灾后重建规划专家，请你严格按照我的步骤一步一步来执行。你的助手building是多模态大模型，可以向他询问图中用方框标记起来的房屋的损毁情况。你的另外一个助手road也是多模态大模型，可以向他询问图中道路等基础设施损毁情况，我会给他们一个当地的卫星图。现在第一步，请你向building询问图中相应细节。请严格按照以下格式输出对building的询问：\n"
    "```query to building：'向building询问的问题'```",
    "现在第二步，请你在输出内容的最后一行输出纯文本问题，向road询问相应细节。请严格按照以下格式输出对road的询问：\n"
    "```query to road：'向road询问的问题'```",
    "现在最后一步，请你综合building/road 的全部信息，作出该地区完整灾后重建方案，包括优先修复顺序、资源调度、人员安置、预算估算、时间表等。请按照以下格式输出你的方案：\n"
    "```analyses：你的方案```\n"
    "输出方案后，请你以 FINAL ANSWER 字样结尾，以便让流程停止。\n"
]


#房屋专家
building_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是building_analyst。一个只负责描述图中房屋分布、损毁情况的多模态大模型"),
    MessagesPlaceholder(variable_name="messages"),
])
building_agent = building_prompt | Building_llm


#道路专家
road_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是road_analyst。一个只负责描述图中道路分布、损毁情况的多模态大模型"),
    MessagesPlaceholder(variable_name="messages"),
])
road_agent = road_prompt | Road_llm



# 2. label_node：生成带框图，并把两份图都塞进 state
def label_node(state: PostDisasterState) -> dict:
    import os
    import tempfile
    from tools.yolo_tool import run_yolo
    state["image"] = open("origin.JPG", "rb").read()
    state["counter"] = 0
    # 1. 把原图写临时文件
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_in:
        tmp_in.write(state["image"])
        tmp_in_name = tmp_in.name

    try:
        # 2. YOLO 检测
        labeled_img_path, _ = run_yolo(tmp_in_name)

        # 3. 读回带框图
        with open(labeled_img_path, "rb") as f:
            labeled_bytes = f.read()

        # 4. 清理临时文件（可选：也清理 YOLO 输出目录）
        import shutil
        shutil.rmtree("F:/损毁房屋result/predict", ignore_errors=True)

    finally:
        os.unlink(tmp_in_name)

    b64_str = base64.b64encode(labeled_bytes).decode("utf-8")

    return {
        "raw_image": state["image"],  # 原图给 road
        "image": labeled_bytes,  # 带框图给 building
        "labeled_image": b64_str,  # 这里把同一份带框图按64位写进 labeled_image
        "sender": "label_tool",
        "counter": state["counter"]
    }


#  3. 图片转为可传递的信息辅助函数
def _make_image_message(image_bytes: bytes) -> HumanMessage:
    """把二进制图片转成 HumanMessage"""
    import base64

    b64 = base64.b64encode(image_bytes).decode()
    url = f"data:image/png;base64,{b64}"
    return HumanMessage(
        content=[
            {"type": "text", "text": "这是灾后的卫星/航拍图，请根据你的角色分析图中损毁信息。。"},
            {"type": "image_url", "image_url": {"url": url}}
        ]
    )


#  4. 消息裁剪函数（防止爆上下文自己加的）
def _extract_last_question(messages: Sequence[BaseMessage]) -> Sequence[BaseMessage]:
    last_content = messages[-1].content or ""
    match = re.search(r"：'(.*?)'", last_content, re.IGNORECASE | re.DOTALL)
    return [HumanMessage(content=match.group(1).strip() if match else "")]


#  5. 三个智能体的节点函数
def analysis_node(state: PostDisasterState):
    idx = min(state["counter"], len(ANALYST_PROMPTS) - 1)
    prompt = ChatPromptTemplate.from_messages([
        ("system", ANALYST_PROMPTS[idx]),
        MessagesPlaceholder(variable_name="messages"),
    ])
    model = prompt | llm
    raw = model.invoke(state)
    msg = raw if isinstance(raw, AIMessage) else AIMessage(content=str(raw))
    msg.name = "analyst"
    print(f"[DEBUG] analysis_node 生成的消息：{msg.content}")
    return {"messages": [msg], "sender": "analyst", "counter": state["counter"] + 1}



def building_node(state: PostDisasterState):
    concise_msgs = _extract_last_question(state["messages"])
    print("↓↓↓↓ building_node 收到的消息 ↓↓↓↓")
    for m in concise_msgs:
        print(f"[{type(m).__name__}] {m.content}")
    print("↑↑↑↑ 以上为 building_node 收到的消息 ↑↑↑↑")

    # 仅在第一次进入 building_node 时插入图片
    image_msg = _make_image_message(state["image"])  # 带框图
    temp_messages = concise_msgs + [image_msg]

    raw = building_agent.invoke({"messages": temp_messages})
    msg = raw if isinstance(raw, AIMessage) else AIMessage(content=str(raw))
    msg.name = "building"
    print(f"[DEBUG] building_node 返回：{msg.content}")
    return {"messages": [msg], "sender": "building", "counter": state["counter"]}


def road_node(state: PostDisasterState):
    concise_msgs = _extract_last_question(state["messages"])
    print("↓↓↓↓ road_node 收到的消息 ↓↓↓↓")
    for m in concise_msgs:
        print(f"[{type(m).__name__}] {m.content}")
    print("↑↑↑↑ 以上为 road_node 收到的消息 ↑↑↑↑")

    # 仅在第一次进入 road_node 时插入图片
    image_msg = _make_image_message(state["raw_image"])  # 原图
    temp_messages = concise_msgs + [image_msg]

    raw = road_agent.invoke({"messages": temp_messages})
    msg = raw if isinstance(raw, AIMessage) else AIMessage(content=str(raw))
    msg.name = "road"
    print(f"[DEBUG] road_node 返回：{msg.content}")
    return {"messages": [msg], "sender": "road", "counter": state["counter"]}

#  6. 条件路由
def router(state: PostDisasterState) -> str:
    if state["counter"] >= 3:
        print(f"[ROUTER] {state.get('sender')} → __end__ ")
        return "__end__"

    last_msg = state["messages"][-1]


    if "FINAL ANSWER" in str(last_msg.content):
        next_node = "__end__"
    else:
        sender = state.get("sender", "analyst")
        if sender == "analyst":
            content = str(last_msg.content).lower()
            if "query to road" in content:
                next_node = "road"
            else:
                next_node = "building"
    print(f"[ROUTER] {state.get('sender')} → {next_node}")
    return next_node


#  7. 构建图
workflow = StateGraph(PostDisasterState)
#创建节点
workflow.add_node("label_tool", label_node)
workflow.add_node("analyst", analysis_node)
workflow.add_node("building", building_node)
workflow.add_node("road", road_node)

workflow.add_edge(START, "label_tool")
workflow.add_edge("label_tool", "analyst")
workflow.add_edge("building", "analyst")
workflow.add_edge("road", "analyst")

workflow.add_conditional_edges(
    "analyst",
    router,
    {"building": "building", "road": "road", "__end__": END},
)



# 9. 异步运行入口以及检查点文件输出
async def main():

    location = '广东省梅州市'

    # 读入卫星云图
    with open("origin.JPG", "rb") as f:
        img_bytes = f.read()

    # 初始化对话状态
    initial = {
        "messages": [HumanMessage(content=location)],
        "sender": "analyst",
        "image": img_bytes,
        "counter": 0
    }

    # 运行工作流
    async with AsyncSqliteSaver.from_conn_string("checkpoints.db") as memory:
        graph = workflow.compile(checkpointer=memory)
        '''
        try:
            graph.get_graph().draw_mermaid_png(output_file_path="./post_disaster_rebuild.png")
        except Exception as e:
            print(e)
        '''
        final_state = await graph.ainvoke(
            initial,
            {"configurable": {"thread_id": "thread-damage-1"},"recursion_limit": 20},
        )

        # 4. 把带框图保存到磁盘，用户可直接查看
        with open("labeled_damage_picture.jpg", "wb") as out_file:
            out_file.write(final_state["labeled_image"])

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
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
from dragent_tools.gen_mask import gen_mask as _gen_mask   # 新增蒙版
from tools.yolo_tool import run_yolo
import base64
#  LLM 客户端
import llm.Client as Client

llm = Client.LLMClient()
Water_llm = Client.LLMClient(api_key="token-abc123",base_url="http://localhost:8888/v1",model="lora1")
Building_llm = Client.LLMClient(api_key="token-abc123",base_url="http://localhost:8888/v1",model="lora2")
Road_llm = Client.LLMClient(api_key="token-abc123",base_url="http://localhost:8888/v1",model="lora3")

#  1. 状态定义
class TyphoonAlertState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str  # 最后一次是谁调用了工具
    image: bytes  # 卫星图，仅 flood、building、road 节点使用
    water_raw_image: bytes  #专门供给flood专家的水体卫星图原图
    counter: int  # <- 新增计数器
    query: Annotated[Sequence[BaseMessage], operator.add]  # 暂时储存analyst向building输出的问题




#label_node 函数
def label_node(state: TyphoonAlertState) -> dict:
    import os
    import tempfile
    from tools.yolo_tool import run_yolo

    # 读取原始图像
    state["image"] = open("origin.JPG", "rb").read()
    state["counter"] = 0

    # 1. 把原图写入临时文件
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
        shutil.rmtree("agents/predict", ignore_errors=True)

    finally:
        os.unlink(tmp_in_name)

    return {
        "image": labeled_bytes,      # 带框图给 building
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
            {"type": "text", "text": "这是当地的卫星图，请根据你的角色分析图中信息。"},
            {"type": "image_url", "image_url": {"url": url}},
        ]
    )



#  4. 提示词的模板
# 台风分析代理（主脑）
# 2. 分阶段提示词列表
ANALYST_PROMPTS = [
    "你是我的台风灾中资源调度方案制定专家，请你严格按照我的步骤一步一步来执行。你的助手flood是多模态大模型，可以回答图中水体位置以及水体周边的情况，我会给他一个当地卫星图。现在第一步，向flood询问图中相应细节。请严格按照以下格式输出对flood的询问（请记得加单引号）：\n"
    "```query to flood：'向flood询问的问题'```",
    "你的助手building是多模态大模型，可以回答图中房屋分布、建筑密度、脆弱性、损毁程度等情况，我会给他一个当地卫星图。现在第二步,请你在输出内容的最后一行输出纯文本问题，向building询问图中相应细节。请严格按照以下格式输出对building的询问：\n"
    "```query to building：'向building询问的问题'```",
    "你的助手road是多模态大模型，可以回答图中道路分布、通行能力、易中断路段等情况，我会给他一个当地卫星图。现在第三步，请你在输出内容的最后一行输出纯文本问题，向road询问相应细节。请严格按照以下格式输出对road的询问：\n"
    "```query to road：'向road询问的问题'```",
    "现在最后一步，请你综合台风数据、flood/building/road 的全部信息，给出完整的风险评估、所需物资、资源调度、人力分配，请按照以下格式输出你的分析：\n"
    "```analyses：你的分析```\n"
    "输出方案后，请你以 FINAL ANSWER 字样结尾，以便让流程停止。\n"
]


# 地形-水体专家
flood_prompt = ChatPromptTemplate.from_messages([
    ("system","你是water_analyst。一个只负责描述图中水体情况的多模态大模型\n"),
    MessagesPlaceholder(variable_name="messages"),
])
flood_agent = flood_prompt | Water_llm


#房屋专家
building_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是building_analyst。一个只负责描述图中房屋分布、建筑密度、脆弱性、损毁程度的多模态大模型"),
    MessagesPlaceholder(variable_name="messages"),
])
building_agent = building_prompt | Building_llm


#道路专家
road_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是road_analyst。一个只负责描述图中道路分布、通行能力、易中断路段的多模态大模型"),
    MessagesPlaceholder(variable_name="messages"),
])
road_agent = road_prompt | Road_llm






#  5. 消息裁剪函数（防止爆上下文自己加的
def _extract_last_question(messages: Sequence[BaseMessage]) -> Sequence[BaseMessage]:
    last_content = messages[-1].content or ""
    match = re.search(r"：'(.*?)'", last_content, re.IGNORECASE | re.DOTALL)
    return [HumanMessage(content=match.group(1).strip() if match else "")]


#  6. 四个智能体的节点函数
def analysis_node(state: TyphoonAlertState):
    idx = min(state["counter"], len(ANALYST_PROMPTS) - 1)
    prompt = ChatPromptTemplate.from_messages([
        ("system", ANALYST_PROMPTS[idx]),
        MessagesPlaceholder(variable_name="messages"),
    ])
    model = prompt | llm#.bind_tools([typhoon_api])
    raw = model.invoke({k: v for k, v in state.items() if k != 'image'})
    msg = raw if isinstance(raw, AIMessage) else AIMessage(content=str(raw))
    msg.name = "analyst"
    print(f"[DEBUG] analysis_node 生成的消息：{msg.content}")
    return {"messages": [msg], "sender": "analyst", "counter": state["counter"] + 1}



def mask_node(state: TyphoonAlertState):
    raw_byte = open("demo_picture.png", "rb").read()
    concise_msgs = _extract_last_question(state["messages"])  # 把前面analyst在messages里面输出给flood的文本暂时放到state里面的query里面去

    # 2. 调用工具生成带蒙版图，但不塞进专家模型
    mask_result = _gen_mask(raw_byte)  # 返回 {'text':..., 'result':...}
    blended_bytes = mask_result["result"]  # 合成后的图片字节

    b64_str = base64.b64encode(blended_bytes).decode("utf-8")
    return {"messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "这是蒙版处理后的水体卫星图，请根据你的角色分析图中水体信息。"},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "mime": "image/png",
                        "data": b64_str,
                    },
                },
            ]
        }
    ], "sender": "mask", "query": concise_msgs }



def flood_node(state: TyphoonAlertState):
    raw_byte = open("demo_picture.png", "rb").read()
    concise_msgs = list(state.get("query", []))  # 先把query里面的文本直接给到专家
    print("↓↓↓↓ flood_node 收到的消息 ↓↓↓↓")
    for m in concise_msgs:
        print(f"[{type(m).__name__}] {m.content}")
    print("↑↑↑↑ 以上为 flood_node 收到的消息 ↑↑↑↑")


    # 仅在第一次进入 flood_node 时插入图片
    image_msg = _make_image_message(raw_byte)
    temp_messages = concise_msgs + [image_msg]

    raw = flood_agent.invoke({"messages": temp_messages})
    msg = raw if isinstance(raw, AIMessage) else AIMessage(content=str(raw))
    msg.name = "flood"
    print(f"[DEBUG] flood_node 返回：{msg.content}")
    return {"messages": [msg],
            "sender": "flood",
            "counter": state["counter"],
            }


# 展示64位编码图片给ui的节点
def display_node(state: TyphoonAlertState):
    concise_msgs = _extract_last_question(state["messages"])  # 把前面analyst在messages里面输出给building的文本暂时放到state里面的query里面去
    # 在message里面放上b64_str
    b64= base64.b64encode(state["image"]).decode("utf-8")
    return {"messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "这是灾后的卫星/航拍图，请根据你的角色分析图中损毁信息。"},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "mime": "image/png",
                        "data": b64,
                    },
                },
            ]
        }
    ], "sender": "display", "query": concise_msgs}


def building_node(state: TyphoonAlertState):

    concise_msgs = list(state.get("query", []))  # 先把query里面的文本直接给到专家
    print("↓↓↓↓ building_node 收到的消息 ↓↓↓↓")
    for m in concise_msgs:
        print(f"[{type(m).__name__}] {m.content}")
    print("↑↑↑↑ 以上为 building_node 收到的消息 ↑↑↑↑")

    # 仅在第一次进入 building_node 时插入图片

    image_msg = _make_image_message(state["image"])
    temp_messages = concise_msgs + [image_msg]

    raw = building_agent.invoke({"messages": temp_messages})
    msg = raw if isinstance(raw, AIMessage) else AIMessage(content=str(raw))
    msg.name = "building"
    print(f"[DEBUG] building_node 返回：{msg.content}")
    return {"messages": [msg], "sender": "building", "counter": state["counter"]}


def road_node(state: TyphoonAlertState):
    concise_msgs = _extract_last_question(state["messages"])
    raw_image = open("origin.JPG", "rb").read()
    print("↓↓↓↓ road_node 收到的消息 ↓↓↓↓")
    for m in concise_msgs:
        print(f"[{type(m).__name__}] {m.content}")
    print("↑↑↑↑ 以上为 road_node 收到的消息 ↑↑↑↑")

    # 仅在第一次进入 road_node 时插入图片

    image_msg = _make_image_message(raw_image)
    temp_messages = concise_msgs + [image_msg]

    raw = road_agent.invoke({"messages": temp_messages})
    msg = raw if isinstance(raw, AIMessage) else AIMessage(content=str(raw))
    msg.name = "road"
    print(f"[DEBUG] road_node 返回：{msg.content}")
    return {"messages": [msg], "sender": "road", "counter": state["counter"]}



#  7. 条件路由
def router(state: TyphoonAlertState) -> str:
    if state["counter"] >= 4:
        print(f"[ROUTER] {state.get('sender')} → __end__ ")
        return "__end__"

    last_msg = state["messages"][-1]

    if "FINAL ANSWER" in str(last_msg.content):
        next_node = "__end__"
    else:
        sender = state.get("sender", "analyst")
        if sender == "analyst":
            content = str(last_msg.content).lower()
            if state["counter"] == 2:
                next_node = "display"
            elif state["counter"] == 3:
                next_node = "road"
            else:
                next_node = "mask"
    print(f"[ROUTER] {state.get('sender')} → {next_node}")
    return next_node



#  8. 构建图
workflow = StateGraph(TyphoonAlertState)
#创建节点
workflow.add_node("label_tool", label_node)
workflow.add_node("analyst", analysis_node)
workflow.add_node("mask", mask_node)
workflow.add_node("flood", flood_node)
workflow.add_node("display", display_node)
workflow.add_node("building", building_node)
workflow.add_node("road", road_node)

workflow.add_edge(START, "label_tool")
workflow.add_edge("label_tool", "analyst")
workflow.add_edge("mask", "flood")
workflow.add_edge("flood", "analyst")
workflow.add_edge("display", "building")
workflow.add_edge("building", "analyst")
workflow.add_edge("road", "analyst")

workflow.add_conditional_edges(
    "analyst",
    router,
    {"mask": "mask", "display": "display", "road": "road", "__end__": END},
)


during=workflow.compile(
    checkpointer=AsyncSqliteSaver.from_conn_string("checkpoints.db"),
    name="dtd1"
)


'''
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
        "counter": 0
    }

    # 运行工作流
    async with AsyncSqliteSaver.from_conn_string("checkpoints.db") as memory:
        graph = workflow.compile(checkpointer=memory)

        try:
            graph.get_graph().draw_mermaid_png(output_file_path="./dtd1.png")
        except Exception as e:
            print(e)

        final_state = await graph.ainvoke(
            initial,
            {"configurable": {"thread_id": "thread-typhoon-1"},"recursion_limit": 20},
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
'''
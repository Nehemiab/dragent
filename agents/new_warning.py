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
    counter: int  # <- 新增计数器
    flood_blended_image: bytes  # 蒙版图


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
            {"type": "text", "text": "这是当地的卫星图，请根据你的角色分析图中信息。"},
            {"type": "image_url", "image_url": {"url": url}},
        ]
    )



#  4. 提示词的模板
# 台风分析代理（主脑）
# 2. 分阶段提示词列表
ANALYST_PROMPTS = [
    "你是我的台风灾害预警分析专家，请你严格按照我的步骤一步一步来执行。现在第一步，请你请你先使用工具typhoon_api查询台风数据。",
    "你的助手flood是多模态大模型，可以回答图中水体位置以及水体周边的情况，我会给他一个当地卫星图。现在第二步,请不要调用工具，不要生成toolcall，请你在输出内容的最后一行输出纯文本问题，向flood询问图中相应细节。请严格按照以下格式在最后一行输出对flood的询问（请记得加单引号）：\n"
    "```query to flood：'向flood询问的问题'```",
    "你的助手building是多模态大模型，可以回答图中房屋分布、建筑密度、脆弱性、损毁程度等情况，我会给他一个当地卫星图。现在第三步,请不要生成toolcall,请你在输出内容的最后一行输出纯文本问题，向building询问图中相应细节。请严格按照以下格式输出对building的询问：\n"
    "```query to building：'向building询问的问题'```",
    "你的助手road是多模态大模型，可以回答图中道路分布、通行能力、易中断路段等情况，我会给他一个当地卫星图。现在第四步，请不要生成toolcall,请你在输出内容的最后一行输出纯文本问题，向road询问相应细节。请严格按照以下格式输出对road的询问：\n"
    "```query to road：'向road询问的问题'```",
    "现在最后一步，请你综合台风数据、flood/building/road 的全部信息，给出完整的风险评估、预防方案、疏散建议、所需物资，请按照以下格式输出你的分析：\n"
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
    model = prompt | llm.bind_tools([typhoon_api])
    raw = model.invoke(state)
    msg = raw if isinstance(raw, AIMessage) else AIMessage(content=str(raw))
    msg.name = "analyst"
    print(f"[DEBUG] analysis_node 生成的消息：{msg.content}")
    return {"messages": [msg], "sender": "analyst", "counter": state["counter"] + 1}


def flood_node(state: TyphoonAlertState):
    concise_msgs = _extract_last_question(state["messages"])
    print("↓↓↓↓ flood_node 收到的消息 ↓↓↓↓")
    for m in concise_msgs:
        print(f"[{type(m).__name__}] {m.content}")
    print("↑↑↑↑ 以上为 flood_node 收到的消息 ↑↑↑↑")

    # 2. 调用工具生成带蒙版图，但不塞进专家模型
    mask_result = _gen_mask(state["image"])  # 返回 {'text':..., 'result':...}
    blended_bytes = mask_result["result"]  # 合成后的图片字节

    # 仅在第一次进入 flood_node 时插入图片
    image_msg = _make_image_message(state["image"])
    temp_messages = concise_msgs + [image_msg]

    raw = flood_agent.invoke({"messages": temp_messages})
    msg = raw if isinstance(raw, AIMessage) else AIMessage(content=str(raw))
    msg.name = "flood"
    print(f"[DEBUG] flood_node 返回：{msg.content}")
    return {"messages": [msg],
            "sender": "flood",
            "counter": state["counter"],
            "flood_blended_image": blended_bytes  # 要输出蒙版图片的话，直接取出字节就可以了
            }


def building_node(state: TyphoonAlertState):
    concise_msgs = _extract_last_question(state["messages"])
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
    return {"messages": [msg], "sender": "road", "counter": state["counter"]}



#  7. 条件路由
def router(state: TyphoonAlertState) -> str:
    if state["counter"] >= 5:
        print(f"[ROUTER] {state.get('sender')} → __end__ ")
        return "__end__"

    last_msg = state["messages"][-1]

    if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
        next_node = "tool_node"
    elif "FINAL ANSWER" in str(last_msg.content):
        next_node = "__end__"
    else:
        sender = state.get("sender", "analyst")
        if sender == "analyst":
            content = str(last_msg.content).lower()
            if "query to building" in content:
                next_node = "building"
            elif "query to road" in content:
                next_node = "road"
            else:
                next_node = "flood"
    print(f"[ROUTER] {state.get('sender')} → {next_node}")
    return next_node



#  8. 构建图
workflow = StateGraph(TyphoonAlertState)
#创建节点
workflow.add_node("analyst", analysis_node)
workflow.add_node("flood", flood_node)
workflow.add_node("building", building_node)
workflow.add_node("road", road_node)
workflow.add_node("tool_node", tool_node)

workflow.add_edge(START, "analyst")
workflow.add_edge("tool_node", "analyst")
workflow.add_edge("flood", "analyst")
workflow.add_edge("building", "analyst")
workflow.add_edge("road", "analyst")

workflow.add_conditional_edges(
    "analyst",
    router,
    {"tool_node": "tool_node", "flood": "flood", "building": "building", "road": "road", "__end__": END},
)




# 9. 异步运行入口以及检查点文件输出
async def main():

    location = '广东省梅州市（纬度 24.3，经度 116.1）'

    # 读入卫星云图
    with open("demo_picture.png", "rb") as f:
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
            graph.get_graph().draw_mermaid_png(output_file_path="./new_warning.png")
        except Exception as e:
            print(e)
        '''
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
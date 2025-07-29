# 导入必要的模块
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, StateGraph, START
from typing import Annotated, Sequence, TypedDict
import operator
import functools
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
import llm.Client


# 定义状态对象，用于在节点间传递数据
class TyphoonAlertState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


# 1. 创建API工具节点

# 台风API工具 - 模拟获取实时台风数据
@tool
def typhoon_api(location: Annotated[str, "要查询台风信息的地区名称"]):
    """使用此工具获取指定地区的实时台风数据，包括路径、风速、降雨量预测等信息"""
    # 这里应该是调用真实API的代码，以下是模拟数据
    return {
        "location": location,
        "typhoon_name": "台风山竹",
        "wind_speed": "35m/s",
        "predicted_path": "正向西北方向移动",
        "arrival_time": "预计24小时后影响该地区",
        "rainfall_prediction": "24小时累计降雨量预计200-300mm"
    }



# 2. 创建AI代理节点

#创建专家大模型代理
def create_flood_agent(llm, tools, system_message: str):
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "你是一个专业的，为台风灾害预警分析专家智能体提供某地的地形、山体、水体等数据的地形水体数据专家。"
         "当台风灾害预警分析专家智能体向你询问某地的地形、山体、水体等数据时，你要向他提供这些数据，如果他继续问一些细节，你要尽可能用你的数据回答他的问题，满足他的要求\n"
         "系统指令: {system_message}\n"
         "可用工具: {tool_names}"),
        MessagesPlaceholder(variable_name="messages")
    ])
    prompt = prompt.partial(system_message=system_message)
    prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
    return prompt | llm.bind_tools(tools)


# 创建台风分析AI代理
def create_typhoon_agent(llm, tools, system_message: str):
    """创建台风分析代理"""
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "你是一个专业的台风灾害预警分析专家。你随时可以通过调用typhoon_api工具，来获取你想要的{location}当地的台风预警数据，\n"
         "你还可以和地形水体数据专家智能体对话，只要你说你是台风灾害预警分析专家智能体，并提出你想要某地的地形、水体、山体等数据，他就会告诉你，\n"
         "如果你还有想要知道的，或者不满意他的回答，你可以继续问他，\n"
         "如果你觉得数据够了，就停止调用工具以及停止和地形水体数据专家智能体对话，开始分析可能的灾害风险并提出详细的预防方案给用户。\n"
         "系统指令: {system_message}\n"
         "可用工具: {tool_names}"),
        MessagesPlaceholder(variable_name="messages")
    ])
    prompt = prompt.partial(system_message=system_message)
    prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
    return prompt | llm.bind_tools(tools)


# 初始化AI模型
llm = llm.Client.LLMClient()

#创建专家模型代理
flood_agent = create_flood_agent(
    llm,
    [],  # 这个代理不需要工具，它只做分析
    system_message=(
        "根据提供的台风数据和地形数据，进行综合分析:\n"
        "1. 评估灾害风险等级(低、中、高、极高)\n"
        "2. 预测可能的灾害类型(如内涝、风灾等)\n"
        "3. 提出详细的预防和应对方案\n"
        "4. 给出疏散建议(如需要)\n"
        "5. 提供物资准备清单\n"
        "你的回答必须专业、详细且可操作。"
    )
)

# 创建分析代理
analysis_agent = create_typhoon_agent(
    llm,
    [typhoon_api],
    system_message=(
        "根据提供的台风数据和地形数据，进行综合分析:\n"
        "1. 评估灾害风险等级(低、中、高、极高)\n"
        "2. 预测可能的灾害类型(如内涝、风灾等)\n"
        "3. 提出详细的预防和应对方案\n"
        "4. 给出疏散建议(如需要)\n"
        "5. 提供物资准备清单\n"
        "你可以先调用台风api工具以及和地形水体数据专家智能体进行对话获取数据，你最后给用户的回答必须专业、详细且可操作。"
    )
)


# 3. 定义节点函数

# API节点函数
def api_node(state, api_tool, name):
    """处理API调用的节点"""
    # 从消息中提取位置信息
    last_message = state["messages"][-1]
    location = last_message.content  # 假设消息内容就是位置

    # 调用API工具

    result = typhoon_api.invoke({"location": location})

    # 创建消息对象
    message = AIMessage(
        content=str(result),
        name=name
    )

    return {
        "messages": [message]
    }


#专家模型代理函数
def flood_node(state):
    """处理分析任务的节点"""
    # 调用专家模型代理
    result = flood_agent.invoke(state)

    # 确保结果是AIMessage
    if not isinstance(result, AIMessage):
        result = AIMessage(content=str(result), name="analyst")
    else:
        result = AIMessage(content=result.content, name="analyst")

    return {"messages": [result]}


# 分析节点函数
def analysis_node(state):
    """处理分析任务的节点"""
    # 调用分析代理
    result = analysis_agent.invoke(state)

    # 确保结果是AIMessage
    if not isinstance(result, AIMessage):
        result = AIMessage(content=str(result), name="analyst")
    else:
        result = AIMessage(content=result.content, name="analyst")

    return {"messages": [result]}


# 4. 构建工作流图

# 创建状态图
workflow = StateGraph(TyphoonAlertState)

# 添加节点
workflow.add_node("typhoon_api", functools.partial(api_node, api_tool=typhoon_api, name="typhoon_api"))
workflow.add_node("flood", flood_node)
workflow.add_node("analyst", analysis_node)

# 设置边关系
workflow.add_edge(START, "typhoon_api")  # 从开始到台风API
workflow.add_edge(START, "flood")  # 从开始到水体专家
workflow.add_edge("typhoon_api", "analyst")  # 台风数据到分析节点
workflow.add_edge("flood", "analyst")  # 地形数据到分析节点
workflow.add_edge("analyst", END)  # 分析结果到结束

# 编译图
graph = workflow.compile()


# 5. 运行示例
def run_typhoon_alert(location):
    """运行台风预警系统"""
    # 初始消息只包含位置信息
    initial_message = HumanMessage(content=location)

    # 运行图
    results = graph.invoke({
        "messages": [initial_message],
    })

    # 返回最终分析结果
    for message in results["messages"]:
        print(f"[DEBUG] message.name = {getattr(message, 'name', None)}")
        if getattr(message, "name", None) == "analyst":
            return message.content
    return "未生成有效预警方案"


# 示例运行
if __name__ == "__main__":
    location = "广东省梅州市"
    print(f"正在为{location}生成台风预警方案...")
    alert = run_typhoon_alert(location)
    print("\n生成的预警方案:")
    print(alert)
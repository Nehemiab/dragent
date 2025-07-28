import llm.Client
from typing import Optional, Dict, Any
import json


class WarningAgent:
    def __init__(self):
        self.llm = llm.Client.LLMClient()

    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        prompt = f"""作为中央气象台台风首席预报员，请分析：

        目标地区：{state['location']}
        当前风速：{state.get('wind_speed', '未知')}
        气压数据：{state.get('pressure', '未知')}

        请返回JSON包含：
        - warning_level: 预警等级（白/蓝/黄/橙/红）
        - reasons: 分析依据
        - prevention_advice: 3条防范建议
        - trigger_response: 是否需启动灾后响应"""

        try:
            result = await self.llm.analyze_json(prompt)
            analysis = json.loads(result)

            return {
                **state,
                "warning_level": analysis.get("warning_level", "未知"),
                "prevention_advice": analysis.get("prevention_advice", []),
                "trigger_response": analysis.get("trigger_response", False),
                "messages": state["messages"] + [{
                    "role": "kimi-warning",
                    "content": result
                }]
            }
        except json.JSONDecodeError:
            # 处理JSON解析错误
            return {**state, "error": "Invalid JSON response from LLM"}
        except Exception as e:
            # 处理其他错误
            return {**state, "error": str(e)}
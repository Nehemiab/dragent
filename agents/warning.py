from kimi.client import KimiAI
from typing import Optional
import json


class WarningAgent:
    def __init__(self):
        self.kimi = KimiAI()

    async def execute(self, state: dict) -> dict:
        prompt = f"""作为中央气象台台风首席预报员，请分析：

        目标地区：{state['location']}
        当前风速：{state.get('wind_speed', '未知')}
        气压数据：{state.get('pressure', '未知')}

        请返回JSON包含：
        - warning_level: 预警等级（白/蓝/黄/橙/红）
        - reasons: 分析依据
        - prevention_advice: 3条防范建议
        - trigger_response: 是否需启动灾后响应"""

        result = await self.kimi.analyze_json(prompt)
        analysis = json.loads(result)

        return {
            **state,
            "warning_level": analysis["warning_level"],
            "prevention_advice": analysis["prevention_advice"],
            "trigger_response": analysis["trigger_response"],
            "messages": state["messages"] + [{
                "role": "kimi-warning",
                "content": result
            }]
        }
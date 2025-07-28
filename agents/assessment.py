import llm.Client
import json
from typing import Dict, Any


class AssessmentAgent:
    def __init__(self):
        self.llm = llm.Client.LLMClient()

    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        prompt = f"""作为住建部灾害评估专家，请分析以下房屋损毁情况：

        **输入数据**：
        - 受灾地区：{state['location']}
        - 台风等级：{state.get('warning_level', '未知')}
        - 灾情报告：{json.dumps(state['damage_report'], indent=2, ensure_ascii=False)}
        - 建筑数据库：{state.get('building_db', '暂无')}

        **请返回JSON**：
        {{
            "damage_level": "轻度/中度/严重",
            "dangerous_buildings": {{
                "count": int,
                "locations": [list]
            }},
            "safety_assessment": {{
                "immediate_risks": [list],
                "long_term_risks": [list]
            }},
            "inspection_priority": [list]
        }}"""

        try:
            analysis = await self.llm.analyze_json(prompt)
            assessment = json.loads(analysis)

            return {
                **state,
                "building_assessment": assessment,
                "messages": state["messages"] + [{
                    "role": "kimi-assessment",
                    "content": analysis
                }]
            }
        except Exception as e:
            return {
                **state,
                "error": f"房屋评估失败: {str(e)}",
                "messages": state["messages"] + [{
                    "role": "error",
                    "content": f"评估错误: {str(e)}"
                }]
            }
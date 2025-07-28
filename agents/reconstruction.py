from kimi.client import KimiAI
import json
from typing import Dict, Any


class ReconstructionAgent:
    def __init__(self):
        self.kimi = KimiAI()

    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        prompt = f"""作为国家发改委重建规划专家，请制定：

        **灾情输入**：
        - 地区：{state['location']}
        - 房屋评估：{json.dumps(state['building_assessment'], indent=2, ensure_ascii=False)}
        - 资源调度：{json.dumps(state.get('resource_plan', {}), indent=2, ensure_ascii=False)}

        **请生成JSON方案**：
        {{
            "phases": [
                {{
                    "name": "紧急阶段(0-7天)",
                    "tasks": [list],
                    "budget": float
                }},
                {{
                    "name": "中期恢复(8-30天)",
                    "tasks": [list],
                    "budget": float
                }},
                {{
                    "name": "长期重建(1-3年)",
                    "tasks": [list],
                    "budget": float
                }}
            ],
            "total_cost": float,
            "key_metrics": {{
                "housing_rebuild": str,
                "economic_recovery": str
            }}
        }}"""

        try:
            plan = await self.kimi.analyze_json(prompt)
            reconstruction = json.loads(plan)

            return {
                **state,
                "reconstruction_plan": reconstruction,
                "messages": state["messages"] + [{
                    "role": "kimi-reconstruction",
                    "content": plan
                }]
            }
        except Exception as e:
            return {
                **state,
                "error": f"重建规划失败: {str(e)}",
                "messages": state["messages"] + [{
                    "role": "error",
                    "content": f"规划错误: {str(e)}"
                }]
            }
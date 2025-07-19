class ResourceAgent:
    def __init__(self):
        self.kimi = KimiAI()

    async def execute(self, state: dict) -> dict:
        prompt = f"""紧急资源调度方案：

        灾情报告：{state['damage_report']}
        可用资源：{state.get('resources', '无')}

        请返回：
        - 物资分配方案
        - 运输路线建议
        - 优先级排序

        JSON格式："""

        plan = await self.kimi.analyze_json(prompt)
        return {
            **state,
            "resource_plan": json.loads(plan),
            "messages": state["messages"] + [{
                "role": "kimi-resource",
                "content": plan
            }]
        }
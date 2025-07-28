class PreventionAgent:
    def __init__(self):
        self.kimi = KimiAI()

    async def execute(self, state: dict) -> dict:
        prompt = f"""根据{state['warning_level']}预警级别：

        地区特征：{state['location']}
        历史灾情：{state.get('history', '无')}

        请生成：
        1. 具体防范措施（按优先级排序）
        2. 需调动的应急资源
        3. 执行时间表

        用JSON格式返回"""

        measures = await self.kimi.analyze_json(prompt)
        return {
            **state,
            "prevention_measures": json.loads(measures),
            "messages": state["messages"] + [{
                "role": "kimi-prevention",
                "content": measures
            }]
        }
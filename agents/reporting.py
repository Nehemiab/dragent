class ReportingAgent:
    def __init__(self):
        self.kimi = KimiAI()

    async def execute(self, state: dict) -> dict:
        prompt = f"""生成台风灾情报告：

        地区：{state['location']}
        预警等级：{state['warning_level']}
        实时数据：{state.get('sensor_data', {})}

        需包含：
        - 受灾人口估算
        - 基础设施损坏清单
        - 紧急需求分类

        JSON格式返回："""

        report = await self.kimi.analyze_json(prompt)
        return {
            **state,
            "damage_report": json.loads(report),
            "messages": state["messages"] + [{
                "role": "kimi-report",
                "content": report
            }]
        }
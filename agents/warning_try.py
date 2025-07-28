import llm.Client
from typing import Dict, Any


class WarningAgent:
    def __init__(self):
        self.llm = llm.Client.LLMClient()

    def _get_user_input(self, prompt: str) -> str:
        """获取用户输入"""
        print(f"\n[系统提示] {prompt}")
        return input(">>> 请输入: ").strip()

    def _display_output(self, title: str, content: str):
        """格式化显示纯文本输出"""
        print(f"\n=== {title} ===\n{content}\n" + "=" * 30)

    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        # 获取用户输入数据
        if 'location' not in state:
            state['location'] = self._get_user_input("请输入目标地区(如:广东省深圳市):")
        if 'wind_speed' not in state:
            state['wind_speed'] = self._get_user_input("请输入当前风速(如:40m/s):")
        if 'pressure' not in state:
            state['pressure'] = self._get_user_input("请输入气压数据(如:980hPa):")

        prompt = f"""作为中央气象台台风首席预报员，请根据以下数据进行分析并直接返回纯文本结果（不要用JSON格式）：

        目标地区：{state['location']}
        当前风速：{state['wind_speed']}
        气压数据：{state['pressure']}

        请按以下格式返回结果：
        1. 预警等级：白/蓝/黄/橙/红
        2. 分析依据：简要说明原因
        3. 防范建议：列出3条具体建议
        4. 是否需要启动灾后响应：是/否"""

        try:
            print("\n[系统] 正在分析气象数据，请稍候...")
            response = await self.llm.ainvoke(prompt)
            analysis_result = response.content

            # 直接显示纯文本结果
            self._display_output("气象分析结果", analysis_result)

            # 将结果存入state（如果需要后续处理）
            return {
                **state,
                "analysis_result": analysis_result,
                "messages": state.get("messages", []) + [{
                    "role": "kimi-warning",
                    "content": analysis_result
                }]
            }

        except Exception as e:
            error_msg = f"分析失败: {str(e)}"
            self._display_output("错误", error_msg)
            return {**state, "error": error_msg}


# 示例使用
async def demo():
    agent = WarningAgent()
    await agent.execute({"messages": []})

if __name__ == "__main__":
    import asyncio
    asyncio.run(demo())
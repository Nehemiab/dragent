import httpx
import yaml
from pathlib import Path

class KimiAI:
    def __init__(self):
        config = yaml.safe_load(Path("config.yaml").read_text())
        self.client = httpx.AsyncClient(
            base_url="https://api.moonshot.cn/v1",
            headers={"Authorization": f"Bearer {config['kimi_api_key']}"},
            timeout=30.0
        )
        self.model = config["model"]

    async def analyze(self, prompt: str) -> str:
        """通用分析接口"""
        response = await self.client.post(
            "/chat/completions",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3
            }
        )
        return response.json()["choices"][0]["message"]["content"]

    async def analyze_json(self, prompt: str) -> dict:
        """获取结构化JSON响应"""
        response = await self.client.post(
            "/chat/completions",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "response_format": {"type": "json_object"},
                "temperature": 0.1  # 更低温度保证稳定性
            }
        )
        return response.json()["choices"][0]["message"]["content"]
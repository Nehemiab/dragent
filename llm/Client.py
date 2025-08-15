from __future__ import annotations
import os
import yaml
from langchain_openai import ChatOpenAI

class LLMClient:
    """
    支持两种初始化方式：
    1. LLMClient()              -> 自动读取 ../config.yaml
    2. LLMClient( api_key, base_url, model) -> 覆盖/忽略 yaml 中的配置
    本项目agent直接使用 LLMClient()
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        *,
        config_path: str = "config.yaml",
        **chat_kwargs,
    ) -> None:

        yaml_cfg = {}
        if (model is None or api_key is None or base_url is None) and os.path.isfile(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                yaml_cfg = yaml.safe_load(f) or {}

        model = model or yaml_cfg.get("model", "")
        api_key = api_key or yaml_cfg.get("api_key", "")
        base_url = base_url or yaml_cfg.get("base_url", "")

        self._llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            stream_usage=True,
            **chat_kwargs,
        )

    def __getattr__(self, name):
        return getattr(self._llm, name)

    def __call__(self, *args, **kwargs):
        return self._llm.invoke(*args, **kwargs)


if __name__ == "__main__":
    # 默认使用../config.yaml
    llm = LLMClient()

    # llm = LLMClient(model="gpt-4o", api_key="sk-xxx", base_url="https://api.xxx.com/v1")

    response = llm.invoke("你好，世界！")
    print(response.content)
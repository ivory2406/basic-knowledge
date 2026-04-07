"""
模型配置加载工具

从 agent/config.toml 读取模型配置，返回 OpenAIChatCompletionClient 实例。
"""

import os
from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:
    import tomli as tomllib  # Python < 3.11 需要 pip install tomli

from autogen_ext.models.openai import OpenAIChatCompletionClient


# 配置文件路径：agent/config.toml
CONFIG_PATH = Path(__file__).parent.parent / "config.toml"


def load_config() -> dict:
    """加载 config.toml 并返回完整配置字典。"""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"配置文件不存在: {CONFIG_PATH}\n"
            f"请参考 config.toml 模板创建配置文件。"
        )
    with open(CONFIG_PATH, "rb") as f:
        return tomllib.load(f)


def get_model_client(model_key: str = "gpt-5-2") -> OpenAIChatCompletionClient:
    """根据 model_key 创建并返回对应的模型客户端。

    Args:
        model_key: config.toml 中 [models.xxx] 的 key 名，默认 "gpt-5-2"

    Returns:
        OpenAIChatCompletionClient 实例

    Example:
        >>> client = get_model_client("gpt-5-2")
        >>> client = get_model_client("deepseek")
    """
    config = load_config()

    models = config.get("models", {})
    if model_key not in models:
        available = list(models.keys())
        raise KeyError(
            f"模型 '{model_key}' 未在 config.toml 中配置。\n"
            f"可用模型: {available}"
        )

    model_conf = models[model_key]
    api_key = model_conf["api_key"]
    base_url = model_conf["base_url"]
    model_name = model_conf["model_name"]

    return OpenAIChatCompletionClient(
        model=model_name,
        base_url=base_url,
        api_key=api_key,
        # Azure OpenAI 等非标准模型需要声明能力
        model_info={
            "vision": False,
            "function_calling": True,
            "json_output": True,
            "family": "unknown",
        },
    )


def list_available_models() -> list[str]:
    """列出 config.toml 中所有可用的模型 key。"""
    config = load_config()
    return list(config.get("models", {}).keys())

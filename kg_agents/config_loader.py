import json
from pathlib import Path

def load_config(config_path) -> dict:
    """Load configuration from config.json"""
    if config_path is None:
        raise ValueError(f"Invalid config_path={config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


# Lazy-loaded config singleton
_config: dict = None


def get_config(config_path='config.json') -> dict:
    """Get cached config"""
    global _config
    if _config is None:
        _config = load_config(config_path)
    return _config


def get_neo4j_config() -> dict:
    """Get Neo4j configuration"""
    return get_config()["neo4j"]


def get_deepseek_config() -> dict:
    """Get DeepSeek configuration"""
    return get_config()["deepseek"]


# Convenience accessors
NEO4J_URI = lambda: get_neo4j_config()["uri"]
NEO4J_AUTH = lambda: (get_neo4j_config()["username"], get_neo4j_config()["password"])
DEEPSEEK_API_KEY = lambda: get_deepseek_config()["api_key"]
DEEPSEEK_BASE_URL = lambda: get_deepseek_config()["base_url"]
DEEPSEEK_MODEL = lambda: get_deepseek_config().get("model", "deepseek-chat")

"""
LLM Factory - Factory pattern for selecting between local and API LLM
"""
import os
from typing import Optional
from config_loader import get_config


def is_local_mode() -> bool:
    """Check if local model mode is enabled"""
    try:
        config = get_config()
        local_path = config.get("local_model_path")
        return local_path is not None and os.path.exists(local_path)
    except Exception:
        return False


def get_local_model_path() -> Optional[str]:
    """Get the local model path from config"""
    try:
        config = get_config()
        return config.get("local_model_path")
    except Exception:
        return None


def get_llm_type() -> str:
    """Get the current LLM type: 'local' or 'api'"""
    if is_local_mode():
        return "local"
    return "api"


def should_use_local() -> bool:
    """Check if we should use local model (config exists, but may fail at load time)"""
    try:
        config = get_config()
        local_path = config.get("local_model_path")
        return local_path is not None and local_path.strip() != ""
    except Exception:
        return False

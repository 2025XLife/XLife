# Core LLM Module
from .client import LLMClient, get_llm_client, get_llm
from .interface import (
    UnifiedLLM,
    get_unified_llm,
    chat,
    chat_with_json,
    extract_keywords,
    is_local_mode,
    get_llm_type
)
from .factory import should_use_local, get_local_model_path
from .local_llm import get_local_llm, unload_local_model

__all__ = [
    # Client (API mode)
    "LLMClient",
    "get_llm_client",
    "get_llm",
    # Unified interface (dual mode)
    "UnifiedLLM",
    "get_unified_llm",
    "chat",
    "chat_with_json",
    "extract_keywords",
    "is_local_mode",
    "get_llm_type",
    "should_use_local",
    "get_local_model_path",
    # Local LLM
    "get_local_llm",
    "unload_local_model"
]

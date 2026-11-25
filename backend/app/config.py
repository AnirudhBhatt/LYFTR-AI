import os
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class ApiKeyEntry:
    key_id: str
    api_key: str


@dataclass(frozen=True)
class Settings:
    api_keys: List[ApiKeyEntry]
    database_url: str
    allowed_origins: List[str]
    enable_failure_simulation: bool
    llm_model: str
    default_system_prompt: str
    cache_type: str
    cache_dir: str


def _parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def load_settings() -> Settings:
    raw_keys = os.getenv("LLM_API_KEYS", "")
    parsed_keys = [k.strip() for k in raw_keys.split(",") if k.strip()]
    api_keys = [
        ApiKeyEntry(key_id=f"key_{idx + 1}", api_key=key)
        for idx, key in enumerate(parsed_keys)
    ]

    default_db_url = (
        "sqlite:////data/app.db"
        if os.path.isdir("/data") and os.access("/data", os.W_OK)
        else "sqlite:///./data/app.db"
    )
    database_url = os.getenv("DATABASE_URL", default_db_url).strip()

    raw_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000")
    allowed_origins = [o.strip() for o in raw_origins.split(",") if o.strip()]

    enable_failure_simulation = _parse_bool(
        os.getenv("ENABLE_FAILURE_SIMULATION"), default=False
    )
    llm_model = os.getenv("LLM_MODEL", "gemini/gemini-2.0-flash").strip()
    default_system_prompt = os.getenv(
        "DEFAULT_SYSTEM_PROMPT",
        "You are a helpful, concise AI assistant.",
    ).strip()

    raw_cache_type = os.getenv("CACHE_TYPE", "local").strip().lower()
    if raw_cache_type in {"in-memory", "memory", "local"}:
        cache_type = "local"
    elif raw_cache_type == "disk":
        cache_type = "disk"
    else:
        cache_type = "local"

    cache_dir = os.getenv("CACHE_DIR", "/data/litellm_cache").strip()

    return Settings(
        api_keys=api_keys,
        database_url=database_url,
        allowed_origins=allowed_origins,
        enable_failure_simulation=enable_failure_simulation,
        llm_model=llm_model,
        default_system_prompt=default_system_prompt,
        cache_type=cache_type,
        cache_dir=cache_dir,
    )

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# Prevent LiteLLM from attempting a remote GitHub fetch of the cost map on startup.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

import litellm  # noqa: E402
from diskcache import Cache as DiskCacheStore  # noqa: E402
from litellm.caching.base_cache import BaseCache  # noqa: E402
from litellm.caching.caching import Cache  # noqa: E402
import litellm.caching.caching_handler as caching_handler  # noqa: E402

from app.config import ApiKeyEntry, Settings  # noqa: E402

logger = logging.getLogger("llm_gateway")

SUPPORTED_SIMULATION_STATUSES = {429, 500, 502, 503}


def _install_safe_cache_callback_guard() -> None:
    """Guard against optional proxy audio imports during non-streaming cache hits."""
    orig_fn = getattr(caching_handler, "_should_defer_streaming_cache_hit_callbacks", None)
    if orig_fn is None or getattr(orig_fn, "_is_safe_guarded", False):
        return

    def _safe_defer(*, cached_result: object) -> bool:
        if isinstance(cached_result, litellm.ModelResponse):
            return False
        return orig_fn(cached_result=cached_result)

    _safe_defer._is_safe_guarded = True  # type: ignore[attr-defined]
    caching_handler._should_defer_streaming_cache_hit_callbacks = _safe_defer


@dataclass(frozen=True)
class GatewayResult:
    reply: str
    resolution_source: str
    model: str
    input_tokens: int
    output_tokens: int
    key_id: Optional[str]
    fallback_attempts: int


class AllProvidersUnavailableError(Exception):
    """Raised when every configured API key fails with a retryable error."""


class NonRetryableProviderError(Exception):
    """Raised on 4xx client errors (e.g. 400 BadRequest) that must not fail over."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


class InvalidSimulationHeaderError(Exception):
    """Raised when X-Simulate-Failure contains invalid syntax or unsupported statuses."""


def parse_simulation_header(header_value: Optional[str]) -> Dict[str, int]:
    """Parse X-Simulate-Failure header into a target -> HTTP status map."""
    if not header_value or not header_value.strip():
        return {}
    mapping: Dict[str, int] = {}
    for raw_pair in header_value.split(","):
        pair = raw_pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            raise InvalidSimulationHeaderError(
                "X-Simulate-Failure entries must use <key_id>=<status> format"
            )
        target, status_str = pair.split("=", 1)
        target = target.strip()
        status_str = status_str.strip()
        if not target:
            raise InvalidSimulationHeaderError("Empty key_id in X-Simulate-Failure")
        try:
            status_code = int(status_str)
        except ValueError as exc:
            raise InvalidSimulationHeaderError(
                f"Invalid status code '{status_str}' in X-Simulate-Failure"
            ) from exc
        if status_code not in SUPPORTED_SIMULATION_STATUSES:
            raise InvalidSimulationHeaderError(
                f"Unsupported simulation status {status_code}; allowed: 429, 500, 502, 503"
            )
        mapping[target] = status_code
    return mapping


def _build_simulated_exception(status_code: int, model: str) -> Exception:
    provider = model.split("/", 1)[0] if "/" in model else "gemini"
    if status_code == 429:
        return litellm.RateLimitError(
            message="Simulated 429 rate limit error",
            llm_provider=provider,
            model=model,
        )
    if status_code == 500:
        return litellm.InternalServerError(
            message="Simulated 500 internal server error",
            llm_provider=provider,
            model=model,
        )
    if status_code == 502:
        return litellm.BadGatewayError(
            message="Simulated 502 bad gateway error",
            llm_provider=provider,
            model=model,
        )
    return litellm.ServiceUnavailableError(
        message="Simulated 503 service unavailable error",
        llm_provider=provider,
        model=model,
    )


def _is_retryable_exception(exc: Exception) -> bool:
    retryable_types = (
        litellm.RateLimitError,
        litellm.InternalServerError,
        litellm.ServiceUnavailableError,
        litellm.BadGatewayError,
        litellm.Timeout,
        litellm.APIConnectionError,
    )
    if isinstance(exc, retryable_types):
        return True

    # Explicit non-retryable client errors
    non_retryable_types = (
        litellm.BadRequestError,
        litellm.ContextWindowExceededError,
        litellm.ContentPolicyViolationError,
        litellm.NotFoundError,
        litellm.UnprocessableEntityError,
    )
    if isinstance(exc, non_retryable_types):
        return False

    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        if status_code == 429 or status_code >= 500:
            return True
        if 400 <= status_code < 500:
            return False

    msg = str(exc).lower()
    if any(
        token in msg
        for token in (
            "resourceexhausted",
            "resource_exhausted",
            "quota",
            "rate limit",
            "429",
            "503",
            "502",
            "500",
            "timed out",
            "timeout",
            "connection error",
        )
    ):
        return True

    return isinstance(exc, litellm.APIError)


def _is_cache_hit(response: object) -> bool:
    hidden = getattr(response, "_hidden_params", None)
    if isinstance(hidden, dict) and hidden.get("cache_hit") is True:
        return True
    return False


def _extract_usage(response: object) -> Tuple[int, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, 0
    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0
    return int(prompt_tokens), int(completion_tokens)


class _DiskCacheBackend(BaseCache):
    """Disk-backed LiteLLM cache adapter powered by diskcache."""

    def __init__(self, directory: str, default_ttl: int = 3600) -> None:
        super().__init__(default_ttl=default_ttl)
        os.makedirs(directory, exist_ok=True)
        self._store = DiskCacheStore(directory=directory)

    def set_cache(self, key, value, **kwargs):
        ttl = self.get_ttl(**kwargs)
        self._store.set(key, value, expire=ttl)

    async def async_set_cache(self, key, value, **kwargs):
        self.set_cache(key, value, **kwargs)

    async def async_set_cache_pipeline(self, cache_list, ttl=None, **kwargs):
        for cache_key, cache_value in cache_list:
            self.set_cache(cache_key, cache_value, ttl=ttl)

    def get_cache(self, key, **kwargs):
        return self._store.get(key, default=None)

    async def async_get_cache(self, key, **kwargs):
        return self.get_cache(key, **kwargs)

    async def batch_cache_write(self, key, value, **kwargs):
        self.set_cache(key, value, **kwargs)

    async def disconnect(self):
        self._store.close()

    async def test_connection(self) -> dict:
        return {"status": "success", "message": "diskcache ready", "error": None}


class LLMGateway:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._token_backup: Dict[str, Tuple[int, int]] = {}
        self.cache_enabled = False
        self._init_cache()

    def _init_cache(self) -> None:
        _install_safe_cache_callback_guard()
        litellm.suppress_debug_info = True
        litellm.drop_params = True

        if self.settings.cache_type == "disk":
            backend = _DiskCacheBackend(directory=self.settings.cache_dir)
            litellm.cache = Cache(_backend=backend, max_messages=None)
        else:
            litellm.cache = Cache(type="local", max_messages=None)
        self.cache_enabled = litellm.cache is not None

    async def _flush_pending_cache_writes(self) -> None:
        pending = getattr(caching_handler, "_PENDING_CACHE_WRITES", None)
        if pending:
            tasks = list(pending)
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

    async def complete(
        self,
        messages: List[Dict[str, str]],
        simulate_failure_header: Optional[str] = None,
    ) -> GatewayResult:
        if not self.settings.api_keys:
            raise AllProvidersUnavailableError("No LLM API keys configured")

        sim_map: Dict[str, int] = {}
        if self.settings.enable_failure_simulation and simulate_failure_header:
            sim_map = parse_simulation_header(simulate_failure_header)

        bypass_cache_read = bool(sim_map)
        model = self.settings.llm_model
        fallback_attempts = 0

        # Deterministic cache key backup for token usage if a response lacks usage on hit
        cache_key = None
        if litellm.cache is not None:
            try:
                cache_key = litellm.cache.get_cache_key(
                    model=model,
                    messages=messages,
                    temperature=0,
                )
            except Exception:
                cache_key = None

        # Always start from key_1 so failover behavior is deterministic and observable.
        for key_entry in self.settings.api_keys:
            simulated_status = sim_map.get(key_entry.key_id, sim_map.get("*"))
            try:
                if simulated_status is not None:
                    raise _build_simulated_exception(simulated_status, model)

                call_kwargs = {
                    "model": model,
                    "messages": messages,
                    "temperature": 0,
                    "api_key": key_entry.api_key,
                    "timeout": 30.0,
                }
                if bypass_cache_read:
                    call_kwargs["cache"] = {"no-cache": True}

                response = await litellm.acompletion(**call_kwargs)
                await self._flush_pending_cache_writes()

                reply_text = ""
                choices = getattr(response, "choices", None)
                if choices:
                    msg = getattr(choices[0], "message", None)
                    reply_text = (getattr(msg, "content", None) or "").strip()
                if not reply_text:
                    raise litellm.ServiceUnavailableError(
                        message="Empty response from LLM provider",
                        llm_provider=model.split("/", 1)[0] if "/" in model else "gemini",
                        model=model,
                    )

                cache_hit = _is_cache_hit(response) and not bypass_cache_read
                input_tokens, output_tokens = _extract_usage(response)
                if cache_hit and input_tokens == 0 and output_tokens == 0 and cache_key:
                    input_tokens, output_tokens = self._token_backup.get(
                        cache_key, (0, 0)
                    )
                elif not cache_hit and cache_key:
                    self._token_backup[cache_key] = (input_tokens, output_tokens)

                return GatewayResult(
                    reply=reply_text,
                    resolution_source="LLM_CACHE" if cache_hit else "LIVE_INFERENCE",
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    key_id=None if cache_hit else key_entry.key_id,
                    fallback_attempts=0 if cache_hit else fallback_attempts,
                )

            except Exception as exc:
                if _is_retryable_exception(exc):
                    fallback_attempts += 1
                    logger.warning(
                        "LLM call failed for %s (%s); failing over",
                        key_entry.key_id,
                        type(exc).__name__,
                    )
                    continue

                status_code = getattr(exc, "status_code", None)
                if not isinstance(status_code, int) or status_code < 400 or status_code >= 500:
                    status_code = 400
                raise NonRetryableProviderError(
                    f"Provider rejected request ({type(exc).__name__})",
                    status_code=status_code,
                ) from exc

        raise AllProvidersUnavailableError("all providers unavailable")

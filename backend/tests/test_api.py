import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, patch

import httpx
import litellm
import pytest
from fastapi.testclient import TestClient

from app.config import ApiKeyEntry, Settings
from app.main import create_app


def _make_settings(
    tmp_path: Path,
    *,
    enable_failure_simulation: bool = True,
    cache_type: str = "local",
) -> Settings:
    db_file = tmp_path / "test_app.db"
    return Settings(
        api_keys=[
            ApiKeyEntry(key_id="key_1", api_key="secret-key-alpha"),
            ApiKeyEntry(key_id="key_2", api_key="secret-key-beta"),
        ],
        database_url=f"sqlite:///{db_file}",
        allowed_origins=["http://localhost:3000"],
        enable_failure_simulation=enable_failure_simulation,
        llm_model="gemini/gemini-2.0-flash",
        default_system_prompt="You are a concise test assistant.",
        cache_type=cache_type,
        cache_dir=str(tmp_path / "litellm_cache"),
    )


def _gemini_response(
    text: str,
    prompt_tokens: int = 20,
    completion_tokens: int = 8,
    status_code: int = 200,
) -> httpx.Response:
    body = {
        "candidates": [
            {
                "content": {"parts": [{"text": text}], "role": "model"},
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": prompt_tokens,
            "candidatesTokenCount": completion_tokens,
            "totalTokenCount": prompt_tokens + completion_tokens,
        },
    }
    return httpx.Response(
        status_code=status_code,
        json=body,
        request=httpx.Request(
            "POST",
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
        ),
    )


@pytest.fixture()
def client_factory(tmp_path: Path):
    created_apps = []

    def _factory(
        *,
        enable_failure_simulation: bool = True,
        cache_type: str = "local",
    ) -> TestClient:
        settings = _make_settings(
            tmp_path,
            enable_failure_simulation=enable_failure_simulation,
            cache_type=cache_type,
        )
        app = create_app(custom_settings=settings)
        created_apps.append(app)
        return TestClient(app)

    yield _factory

    for app in created_apps:
        app.state.db.close()
    litellm.cache = None


def test_health_endpoint(client_factory) -> None:
    client = client_factory()
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {
        "status": "ok",
        "keys_configured": 2,
        "cache_enabled": True,
    }


def test_validation_422_cases_do_not_persist_history(client_factory) -> None:
    client = client_factory()

    invalid_payloads: List[Dict[str, Any]] = [
        {"session_id": "s-bad", "message": ""},
        {"session_id": "s-bad", "message": "   \n\t  "},
        {"session_id": "s-bad", "message": "x" * 4001},
        {"session_id": "", "message": "hello"},
        {"session_id": "   ", "message": "hello"},
        {"session_id": "s" * 129, "message": "hello"},
        {"session_id": "s-bad"},
        {"message": "hello"},
    ]
    for payload in invalid_payloads:
        resp = client.post("/chat", json=payload)
        assert resp.status_code == 422
        assert "detail" in resp.json()

    # Malformed JSON
    malformed = client.post(
        "/chat",
        content="{not valid json",
        headers={"Content-Type": "application/json"},
    )
    assert malformed.status_code == 422
    assert "detail" in malformed.json()

    # Failed 422 requests must not create session history
    hist = client.get("/sessions/s-bad/messages")
    assert hist.status_code == 404
    assert hist.json() == {"detail": "session not found"}


def test_trimmed_message_of_4000_chars_is_accepted(client_factory) -> None:
    client = client_factory()
    padded = "   " + ("a" * 4000) + "   "
    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        new_callable=AsyncMock,
        return_value=_gemini_response("Accepted!", prompt_tokens=100, completion_tokens=2),
    ):
        resp = client.post("/chat", json={"session_id": "s-4000", "message": padded})
    assert resp.status_code == 200
    assert resp.json()["reply"] == "Accepted!"


def test_multi_turn_memory_and_session_isolation(client_factory) -> None:
    client = client_factory()
    recorded_payloads: List[Dict[str, Any]] = []

    async def fake_post(*args, **kwargs):
        raw_data = kwargs.get("data") or kwargs.get("json")
        if isinstance(raw_data, (str, bytes)):
            parsed = json.loads(raw_data)
        else:
            parsed = raw_data or {}
        recorded_payloads.append(parsed)
        contents = parsed.get("contents", [])
        last_text = contents[-1]["parts"][-1]["text"]
        if "name is Alice" in last_text:
            return _gemini_response("Nice to meet you, Alice.", prompt_tokens=25, completion_tokens=6)
        if "What is my name" in last_text:
            joined = " ".join(p["text"] for c in contents for p in c["parts"])
            reply = "Your name is Alice." if "Alice" in joined else "I do not know your name."
            return _gemini_response(reply, prompt_tokens=45, completion_tokens=5)
        return _gemini_response("Hello from session B.", prompt_tokens=22, completion_tokens=4)

    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        new_callable=AsyncMock,
        side_effect=fake_post,
    ):
        r1 = client.post(
            "/chat",
            json={
                "session_id": "sess-a",
                "message": "My name is Alice.",
                "system_prompt": "Always answer politely.",
            },
        )
        assert r1.status_code == 200
        assert r1.json()["reply"] == "Nice to meet you, Alice."

        # Turn 2 in sess-a should include turn 1 and persist custom system prompt
        r2 = client.post(
            "/chat",
            json={"session_id": "sess-a", "message": "What is my name?"},
        )
        assert r2.status_code == 200
        assert r2.json()["reply"] == "Your name is Alice."

        # Different session should be completely isolated
        r3 = client.post(
            "/chat",
            json={"session_id": "sess-b", "message": "What is my name?"},
        )
        assert r3.status_code == 200
        assert r3.json()["reply"] == "I do not know your name."

    # Verify history endpoint shape for sess-a: oldest first, no system prompt
    hist_a = client.get("/sessions/sess-a/messages")
    assert hist_a.status_code == 200
    body_a = hist_a.json()
    assert body_a["session_id"] == "sess-a"
    msgs = body_a["messages"]
    assert len(msgs) == 4
    assert [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant"]
    assert msgs[0] == {"role": "user", "content": "My name is Alice."}
    assert msgs[1]["content"] == "Nice to meet you, Alice."
    assert msgs[1]["metadata"]["resolution_source"] == "LIVE_INFERENCE"
    assert msgs[1]["metadata"]["input_tokens"] == 25
    assert msgs[1]["metadata"]["output_tokens"] == 6
    assert msgs[2] == {"role": "user", "content": "What is my name?"}
    assert msgs[3]["content"] == "Your name is Alice."
    assert msgs[3]["metadata"]["input_tokens"] == 45
    assert msgs[3]["metadata"]["output_tokens"] == 5

    # Ensure system prompt from sess-a was persisted and sent on turn 2
    assert len(recorded_payloads) == 3
    assert "Always answer politely." in json.dumps(recorded_payloads[1])


def test_cache_hit_across_sessions_is_faster_and_null_key_id(client_factory) -> None:
    client = client_factory()

    async def slow_live_post(*args, **kwargs):
        await asyncio.sleep(0.03)
        raw_data = kwargs.get("data") or kwargs.get("json")
        parsed = json.loads(raw_data) if isinstance(raw_data, (str, bytes)) else raw_data
        text = parsed["contents"][-1]["parts"][-1]["text"]
        if "capital of France" in text:
            return _gemini_response("Paris.", prompt_tokens=30, completion_tokens=2)
        return _gemini_response("Berlin.", prompt_tokens=31, completion_tokens=2)

    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        new_callable=AsyncMock,
        side_effect=slow_live_post,
    ) as mock_post:
        first = client.post(
            "/chat",
            json={"session_id": "cache-s1", "message": "What is the capital of France?"},
        )
        assert first.status_code == 200
        d1 = first.json()
        assert d1["reply"] == "Paris."
        assert d1["metadata"]["resolution_source"] == "LIVE_INFERENCE"
        assert d1["metadata"]["key_id"] == "key_1"
        assert d1["metadata"]["fallback_attempts"] == 0
        assert d1["metadata"]["input_tokens"] == 30
        assert d1["metadata"]["output_tokens"] == 2
        assert mock_post.call_count == 1

        # Identical first message in a NEW session must hit cache
        second = client.post(
            "/chat",
            json={"session_id": "cache-s2", "message": "What is the capital of France?"},
        )
        assert second.status_code == 200
        d2 = second.json()
        assert d2["reply"] == d1["reply"]
        assert d2["metadata"]["resolution_source"] == "LLM_CACHE"
        assert d2["metadata"]["key_id"] is None
        assert d2["metadata"]["fallback_attempts"] == 0
        assert d2["metadata"]["input_tokens"] == d1["metadata"]["input_tokens"]
        assert d2["metadata"]["output_tokens"] == d1["metadata"]["output_tokens"]
        assert d2["metadata"]["latency_ms"] < d1["metadata"]["latency_ms"]
        assert mock_post.call_count == 1

        # Different message must NOT hit cache
        third = client.post(
            "/chat",
            json={"session_id": "cache-s3", "message": "What is the capital of Germany?"},
        )
        assert third.status_code == 200
        d3 = third.json()
        assert d3["reply"] == "Berlin."
        assert d3["metadata"]["resolution_source"] == "LIVE_INFERENCE"
        assert d3["metadata"]["key_id"] == "key_1"
        assert mock_post.call_count == 2


def test_failover_simulation_and_history_unchanged_on_503(client_factory) -> None:
    client = client_factory(enable_failure_simulation=True)

    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        new_callable=AsyncMock,
        return_value=_gemini_response("Recovered answer.", prompt_tokens=18, completion_tokens=3),
    ) as mock_post:
        # Warm cache first so we also verify X-Simulate-Failure bypasses cache read
        warm = client.post(
            "/chat",
            json={"session_id": "sim-warm", "message": "Failover test"},
        )
        assert warm.status_code == 200
        assert warm.json()["metadata"]["resolution_source"] == "LIVE_INFERENCE"
        assert mock_post.call_count == 1

        # key_1=429 -> key_2 answers, fallback_attempts=1, bypasses cache
        r_429 = client.post(
            "/chat",
            json={"session_id": "sim-s1", "message": "Failover test"},
            headers={"X-Simulate-Failure": "key_1=429"},
        )
        assert r_429.status_code == 200
        m_429 = r_429.json()["metadata"]
        assert r_429.json()["reply"] == "Recovered answer."
        assert m_429["resolution_source"] == "LIVE_INFERENCE"
        assert m_429["key_id"] == "key_2"
        assert m_429["fallback_attempts"] == 1
        assert "secret-key" not in r_429.text
        assert mock_post.call_count == 2

        # key_1=500 -> key_2 answers
        r_500 = client.post(
            "/chat",
            json={"session_id": "sim-s2", "message": "Failover test 500"},
            headers={"X-Simulate-Failure": "key_1=500"},
        )
        assert r_500.status_code == 200
        assert r_500.json()["metadata"]["key_id"] == "key_2"
        assert r_500.json()["metadata"]["fallback_attempts"] == 1
        assert mock_post.call_count == 3

        # *=429 -> 503 and nothing added to history
        r_all = client.post(
            "/chat",
            json={"session_id": "sim-s1", "message": "Should fail completely"},
            headers={"X-Simulate-Failure": "*=429"},
        )
        assert r_all.status_code == 503
        assert r_all.json() == {"detail": "all providers unavailable"}
        assert mock_post.call_count == 3

    # Verify sim-s1 still only has its first successful turn (2 messages)
    hist = client.get("/sessions/sim-s1/messages")
    assert hist.status_code == 200
    assert len(hist.json()["messages"]) == 2


def test_simulation_header_ignored_when_disabled(client_factory) -> None:
    client = client_factory(enable_failure_simulation=False)

    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        new_callable=AsyncMock,
        return_value=_gemini_response("Normal key_1 reply.", prompt_tokens=16, completion_tokens=3),
    ):
        resp = client.post(
            "/chat",
            json={"session_id": "sim-disabled", "message": "Hello"},
            headers={"X-Simulate-Failure": "*=429"},
        )
    assert resp.status_code == 200
    assert resp.json()["metadata"]["key_id"] == "key_1"
    assert resp.json()["metadata"]["fallback_attempts"] == 0


def test_real_provider_429_failover_and_non_retryable_400(client_factory) -> None:
    client = client_factory()

    call_idx = 0

    async def fake_post(*args, **kwargs):
        nonlocal call_idx
        call_idx += 1
        if call_idx == 1:
            return httpx.Response(
                status_code=429,
                json={"error": {"message": "Rate limit exceeded"}},
                request=httpx.Request("POST", "https://generativelanguage.googleapis.com"),
            )
        return _gemini_response("Answered by key_2", prompt_tokens=19, completion_tokens=4)

    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        new_callable=AsyncMock,
        side_effect=fake_post,
    ):
        ok = client.post("/chat", json={"session_id": "real-failover", "message": "Hi"})
    assert ok.status_code == 200
    assert ok.json()["metadata"]["key_id"] == "key_2"
    assert ok.json()["metadata"]["fallback_attempts"] == 1

    # Non-retryable 400 error must NOT fail over to key_2 and must NOT save history
    bad_req_resp = httpx.Response(
        status_code=400,
        json={"error": {"message": "Invalid argument"}},
        request=httpx.Request("POST", "https://generativelanguage.googleapis.com"),
    )
    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        new_callable=AsyncMock,
        return_value=bad_req_resp,
    ) as mock_400:
        err = client.post("/chat", json={"session_id": "bad-400-sess", "message": "Trigger 400"})
    assert err.status_code == 400
    assert "detail" in err.json()
    assert mock_400.call_count == 1
    assert client.get("/sessions/bad-400-sess/messages").status_code == 404


def test_tokens_are_per_request_not_cumulative(client_factory) -> None:
    client = client_factory()

    async def token_post(*args, **kwargs):
        raw_data = kwargs.get("data") or kwargs.get("json")
        parsed = json.loads(raw_data) if isinstance(raw_data, (str, bytes)) else raw_data
        contents = parsed.get("contents", [])
        # Estimate prompt tokens proportional to message count + chars in this request
        char_count = sum(len(p["text"]) for c in contents for p in c["parts"])
        return _gemini_response(
            "Ok",
            prompt_tokens=10 + char_count,
            completion_tokens=5,
        )

    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        new_callable=AsyncMock,
        side_effect=token_post,
    ):
        a1 = client.post("/chat", json={"session_id": "tok-a", "message": "Alpha!"})
        b1 = client.post("/chat", json={"session_id": "tok-b", "message": "Bravos"})
        a2 = client.post("/chat", json={"session_id": "tok-a", "message": "Second"})

    assert a1.status_code == 200
    assert b1.status_code == 200
    assert a2.status_code == 200

    # Equal-length first messages in two fresh sessions get identical prompt token counts
    assert a1.json()["metadata"]["input_tokens"] == b1.json()["metadata"]["input_tokens"]
    assert a1.json()["metadata"]["output_tokens"] == 5
    # Turn 2 reports THIS request's prompt tokens (not running sum of turn 1 + turn 2)
    sys_len = len("You are a concise test assistant.")
    assert a2.json()["metadata"]["input_tokens"] == 10 + sys_len + len("Alpha!") + len("Ok") + len("Second")
    assert a2.json()["metadata"]["output_tokens"] == 5


def test_disk_cache_mode_initializes_and_hits(client_factory) -> None:
    client = client_factory(cache_type="disk")
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["cache_enabled"] is True

    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        new_callable=AsyncMock,
        return_value=_gemini_response("Disk cached reply.", prompt_tokens=12, completion_tokens=3),
    ) as mock_post:
        r1 = client.post("/chat", json={"session_id": "disk-1", "message": "Disk hello"})
        r2 = client.post("/chat", json={"session_id": "disk-2", "message": "Disk hello"})
    assert r1.status_code == 200
    assert r1.json()["metadata"]["resolution_source"] == "LIVE_INFERENCE"
    assert r2.status_code == 200
    assert r2.json()["metadata"]["resolution_source"] == "LLM_CACHE"
    assert r2.json()["reply"] == "Disk cached reply."
    assert mock_post.call_count == 1


def test_unknown_session_returns_404(client_factory) -> None:
    client = client_factory()
    resp = client.get("/sessions/does-not-exist/messages")
    assert resp.status_code == 404
    assert resp.json() == {"detail": "session not found"}

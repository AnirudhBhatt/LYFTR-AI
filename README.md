# LLM Chatbot with API-Key Failover and Response Caching

A full-stack, deployable LLM chatbot built with **React (Vite)**, **FastAPI**, **LiteLLM**, and **SQLite**. It supports multi-turn conversation memory per session, automatic API-key pool failover, deterministic response caching, per-request token tracking, and an `X-Simulate-Failure` test hook for demonstrating resilience on demand.

---

## 1. Live URLs

- **Frontend (Vercel):** https://lyftr-ai-eta.vercel.app
- **Backend (Render):** https://lyftr-llm-backend-w8h5.onrender.com
- **Health check:** https://lyftr-llm-backend-w8h5.onrender.com/health

---

## 2. Run locally: with Docker and without

### Prerequisites

- At least two LLM API keys (default model: `gemini/gemini-2.0-flash`).
- Copy `.env.example` to `.env` and fill in your real keys (never commit `.env`):

```bash
cp .env.example .env
```

### With Docker (`make up`)

```bash
make up
```

This builds the multi-stage backend (`python:3.11-slim`, non-root `appuser`) and frontend (`node:20-alpine` -> `nginx:1.27-alpine`) images, mounts the named volume `sqlite_data` at `/data`, and starts:

- **Frontend:** `http://localhost:3000`
- **Backend API:** `http://localhost:8000`
- **Health endpoint:** `http://localhost:8000/health`

Useful commands:

```bash
make logs   # tail container logs
make down   # stop and remove containers (named volume /data persists)
```

### Without Docker

#### Backend

```bash
cd backend
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt

export LLM_API_KEYS="your-key-1,your-key-2"
export DATABASE_URL="sqlite:///./data/app.db"
export ALLOWED_ORIGINS="http://localhost:3000"
export ENABLE_FAILURE_SIMULATION="true"

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Run backend tests (mocks LiteLLM HTTP transport so no real API keys are required):

```bash
make test
# or:
backend/.venv/bin/pytest -c backend/pytest.ini backend/tests -v
```

#### Frontend

```bash
cd frontend
npm install
VITE_API_URL="http://localhost:8000" npm run dev
```

Open `http://localhost:3000`.

---

## 3. Design decisions: key fallback, caching, SQLite session storage

### Key fallback (`backend/app/llm.py`)

- `LLM_API_KEYS` is parsed in order into `key_1`, `key_2`, ... (`ApiKeyEntry`). Raw keys are never logged or returned in API responses or error payloads.
- **Deterministic starting order:** Every uncached call starts from `key_1` and walks sequentially (`key_1` -> `key_2` -> ...) instead of round-robin rotation. This guarantees predictable primary-key usage and makes `X-Simulate-Failure: key_1=429` reliably fail over to `key_2` with `fallback_attempts=1`.
- **Retryable vs. non-retryable errors:** Retryable errors (`RateLimitError`/429, `InternalServerError`/500, `BadGatewayError`/502, `ServiceUnavailableError`/503, `Timeout`, `APIConnectionError`, and quota/`ResourceExhausted` messages) increment `fallback_attempts` and move to the next key. Non-retryable client errors (`BadRequestError`/400, context window/content policy errors) return cleanly without failing over. If all keys fail, the API returns HTTP `503` with `{"detail": "all providers unavailable"}`.
- **Failure simulation hook:** When `ENABLE_FAILURE_SIMULATION=true`, `X-Simulate-Failure` is parsed before calling the provider and raises the matching LiteLLM exception (`RateLimitError`, `InternalServerError`, `BadGatewayError`, `ServiceUnavailableError`) for the targeted key(s), exercising the exact same failover code path while bypassing cache reads (`cache={"no-cache": True}`).

### Caching (`backend/app/llm.py`)

- Uses LiteLLM's built-in `Cache` (`CACHE_TYPE=local` for in-memory, or `CACHE_TYPE=disk` backed by `diskcache` at `CACHE_DIR`) with `temperature=0` so identical `[model + system_prompt + history + new_message]` requests produce deterministic cache keys.
- `api_key` is excluded from LiteLLM's cache key hash, so a reply cached under `key_1` serves as `LLM_CACHE` regardless of which key would have been tried.
- On `LLM_CACHE` hits (`response._hidden_params["cache_hit"] is True`), `key_id` is `null`, `fallback_attempts` is `0`, and per-request `input_tokens`/`output_tokens` are preserved from the cached response (with an internal token backup fallback if any provider omits usage on a cached payload).
- After each live `acompletion`, pending LiteLLM async cache write tasks are flushed before returning `200` so immediate back-to-back identical requests always hit the cache reliably.

### SQLite session storage (`backend/app/db.py`)

- Schema is auto-created at startup with two tables:
  - `sessions(session_id PRIMARY KEY, system_prompt, created_at)` to persist per-session system prompt overrides.
  - `messages(id, session_id, role, content, metadata, created_at)` with `idx_messages_session_id` index.
- On `POST /chat`, the backend builds `[effective_system_prompt] + stored_history + [new_user_message]`.
- **Transactional integrity:** `save_turn` writes the session row, user message, and assistant message atomically *only* after a successful LLM response. Any `422`, `400`, or `503` request leaves session history completely untouched.
- `GET /sessions/{session_id}/messages` returns messages oldest-first without the system prompt, or `404 {"detail": "session not found"}` for unknown sessions.

---

## 4. Verification: how to see failover, cache hits and per-message token usage

Replace `http://localhost:8000` with your deployed backend URL when verifying production.

### 1) Health check

```bash
curl -s http://localhost:8000/health | jq
```

Expected:
```json
{
  "status": "ok",
  "keys_configured": 2,
  "cache_enabled": true
}
```

### 2) Live inference & per-request token usage

```bash
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"demo-s1","message":"What is the capital of France?"}' | jq
```

Expected: `"resolution_source": "LIVE_INFERENCE"`, `"key_id": "key_1"`, `"fallback_attempts": 0`, and non-zero `input_tokens` / `output_tokens` for this request only.

### 3) Cache hit in a new session

Send the exact same first message under a new `session_id`:

```bash
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"demo-s2","message":"What is the capital of France?"}' | jq
```

Expected: `"resolution_source": "LLM_CACHE"`, identical `"reply"`, `"key_id": null`, `"fallback_attempts": 0`, same token counts, and noticeably lower `"latency_ms"`.

### 4) API-key failover via `X-Simulate-Failure`

Simulate `key_1` returning HTTP 429 (cache read is bypassed automatically):

```bash
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "X-Simulate-Failure: key_1=429" \
  -d '{"session_id":"demo-s3","message":"What is the capital of France?"}' | jq
```

Expected: HTTP `200`, `"resolution_source": "LIVE_INFERENCE"`, `"key_id": "key_2"`, `"fallback_attempts": 1`.

Simulate all keys failing (`*=429`):

```bash
curl -i -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "X-Simulate-Failure: *=429" \
  -d '{"session_id":"demo-s3","message":"This should fail"}'
```

Expected: HTTP `503` with `{"detail":"all providers unavailable"}` and nothing added to `demo-s3` history.

### 5) Multi-turn history & validation errors

```bash
# Inspect stored history (oldest first, no system prompt)
curl -s http://localhost:8000/sessions/demo-s1/messages | jq

# Unknown session -> 404
curl -i -s http://localhost:8000/sessions/unknown-session/messages

# Whitespace-only message -> 422
curl -i -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"demo-bad","message":"   "}'
```

---

## 5. Setup Used: your editor and AI tools

- **Editor / IDE:** VS Code / Antigravity IDE
- **AI Assistant:** Google Antigravity coding agent (used for architecture scaffolding, LiteLLM integration checks, pytest test coverage, and Dockerfile/nginx setup)

---

## 6. Bonus features (if any)

- **Optional UI Controls Panel:** Collapsible toolbar in the React frontend allowing users to override the session `system_prompt` (`data-testid="system-prompt-input"`) and toggle `X-Simulate-Failure` presets (`key_1=429`, `key_1=500`, `*=429`) directly from the browser for interactive failover demos.
- **Persistent Disk Cache Option (`CACHE_TYPE=disk`):** Added a `diskcache`-backed `BaseCache` adapter so response caches can also survive backend restarts under `/data` alongside SQLite.
- **Session Persistence Table:** Added a `sessions` table in SQLite so custom `system_prompt` overrides persist cleanly across multi-turn requests within a session.

---

## Deployment Guide

### Backend (Render / Railway / Fly.io)

1. Create a Web Service pointing to `/backend` using `backend/Dockerfile` (exposes port `8000`).
2. Attach a persistent disk/volume mounted at `/data`.
3. Set environment variables:
   - `LLM_API_KEYS=<key1>,<key2>`
   - `DATABASE_URL=sqlite:////data/app.db`
   - `ALLOWED_ORIGINS=https://<your-frontend-domain>`
   - `ENABLE_FAILURE_SIMULATION=true`
   - `LLM_MODEL=gemini/gemini-2.0-flash`
4. Set health check path to `/health`.

### Frontend (Vercel / Netlify / Cloudflare Pages)

1. Connect the repository with root directory `frontend`.
2. Build command: `npm run build` · Output directory: `dist`.
3. Set environment variable `VITE_API_URL=https://<your-backend-domain>`.
4. Deploy and update `ALLOWED_ORIGINS` on the backend to include the final frontend URL, then verify with the `curl` commands in Section 4.

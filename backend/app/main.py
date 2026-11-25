import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import Settings, load_settings
from app.db import Database
from app.llm import (
    AllProvidersUnavailableError,
    InvalidSimulationHeaderError,
    LLMGateway,
    NonRetryableProviderError,
)
from app.schemas import (
    ChatMetadata,
    ChatRequest,
    ChatResponse,
    HealthResponse,
    SessionMessageItem,
    SessionMessagesResponse,
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("app")


def create_app(custom_settings: Optional[Settings] = None) -> FastAPI:
    settings = custom_settings or load_settings()
    db = Database(settings.database_url)
    gateway = LLMGateway(settings)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        db.init_schema()
        yield
        db.close()

    app = FastAPI(
        title="Lyftr LLM Chatbot Gateway",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.state.settings = settings
    app.state.db = db
    app.state.gateway = gateway

    if custom_settings is not None:
        db.init_schema()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins or ["*"],
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization", "X-Simulate-Failure"],
        expose_headers=[],
    )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(_request: Request, exc: HTTPException) -> JSONResponse:
        detail = exc.detail if exc.detail is not None else "request error"
        return JSONResponse(status_code=exc.status_code, content={"detail": detail})

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        errors = exc.errors()
        sanitized = [
            {
                "loc": err.get("loc"),
                "msg": err.get("msg"),
                "type": err.get("type"),
            }
            for err in errors
        ]
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": sanitized},
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
        logger.error("Unhandled server error (%s)", type(exc).__name__)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "internal server error"},
        )

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        if not db.check_health():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="database unavailable",
            )
        if len(settings.api_keys) < 1:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="no LLM API keys configured",
            )
        return HealthResponse(
            status="ok",
            keys_configured=len(settings.api_keys),
            cache_enabled=gateway.cache_enabled,
        )

    @app.post("/chat", response_model=ChatResponse)
    async def chat(
        payload: ChatRequest,
        x_simulate_failure: Optional[str] = Header(default=None, alias="X-Simulate-Failure"),
    ) -> ChatResponse:
        start_time = time.perf_counter()

        exists, stored_prompt, history = db.get_session_context(payload.session_id)

        # Resolve effective system prompt: request override > stored session prompt > default
        if payload.system_prompt is not None:
            effective_system_prompt = payload.system_prompt
            prompt_to_persist: Optional[str] = payload.system_prompt
        elif exists and stored_prompt is not None:
            effective_system_prompt = stored_prompt
            prompt_to_persist = None
        else:
            effective_system_prompt = settings.default_system_prompt
            prompt_to_persist = None

        llm_messages: List[Dict[str, str]] = []
        if effective_system_prompt:
            llm_messages.append({"role": "system", "content": effective_system_prompt})
        llm_messages.extend(history)
        llm_messages.append({"role": "user", "content": payload.message})

        try:
            result = await gateway.complete(
                messages=llm_messages,
                simulate_failure_header=x_simulate_failure,
            )
        except InvalidSimulationHeaderError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            ) from exc
        except NonRetryableProviderError as exc:
            raise HTTPException(
                status_code=exc.status_code,
                detail=str(exc),
            ) from exc
        except AllProvidersUnavailableError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="all providers unavailable",
            ) from exc

        latency_ms = int(round((time.perf_counter() - start_time) * 1000))
        metadata_dict = {
            "resolution_source": result.resolution_source,
            "model": result.model,
            "latency_ms": latency_ms,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "key_id": result.key_id,
            "fallback_attempts": result.fallback_attempts,
        }

        db.save_turn(
            session_id=payload.session_id,
            system_prompt=prompt_to_persist,
            user_content=payload.message,
            assistant_content=result.reply,
            metadata=metadata_dict,
        )

        return ChatResponse(
            session_id=payload.session_id,
            reply=result.reply,
            metadata=ChatMetadata(**metadata_dict),
        )

    @app.get(
        "/sessions/{session_id}/messages",
        response_model=SessionMessagesResponse,
        response_model_exclude_none=True,
    )
    async def get_session_messages(session_id: str) -> SessionMessagesResponse:
        messages = db.get_session_messages(session_id)
        if messages is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="session not found",
            )
        return SessionMessagesResponse(
            session_id=session_id,
            messages=[SessionMessageItem(**item) for item in messages],
        )

    return app


app = create_app()

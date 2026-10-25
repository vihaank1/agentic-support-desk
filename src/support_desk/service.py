"""FastAPI service: POST /ask, GET /healthz. Structured JSON logs to stdout (Cloud Run/Logging friendly).

Configure with env vars: LLM_PROVIDER (mock|openai|gemini, default mock), MAX_USD_PER_PROCESS (spend guard).
"""
import json
import logging
import os
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .graph import ask, build_graph
from .llm import BudgetExceeded, make_llm
from .retrieval import Retriever
from .tracing import Tracer

ROOT = Path(__file__).resolve().parents[2]
log = logging.getLogger("support_desk")
logging.basicConfig(level=logging.INFO, format="%(message)s")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)


def create_app(llm=None) -> FastAPI:
    app = FastAPI(title="Agentic Support Desk")
    provider = os.environ.get("LLM_PROVIDER", "mock")
    max_usd = float(os.environ["MAX_USD_PER_PROCESS"]) if "MAX_USD_PER_PROCESS" in os.environ else None
    llm = llm or make_llm(provider, max_usd=max_usd)
    retriever = Retriever(ROOT / "data" / "docs")

    @app.get("/healthz")
    def healthz():
        return {"status": "ok", "provider": llm.name}

    @app.post("/ask")
    def ask_endpoint(req: AskRequest):
        trace_id, t0 = uuid.uuid4().hex[:12], time.perf_counter()
        tracer = Tracer()
        try:
            result = ask(build_graph(llm, retriever, tracer), tracer, req.question)
        except BudgetExceeded as e:
            log.error(json.dumps({"trace_id": trace_id, "event": "budget_exceeded", "detail": str(e)}))
            raise HTTPException(status_code=503, detail="spend guard tripped")
        except Exception as e:
            log.error(json.dumps({"trace_id": trace_id, "event": "error", "detail": str(e)[:200]}))
            raise HTTPException(status_code=502, detail="upstream model error")
        metrics = tracer.summary()
        log.info(json.dumps({"trace_id": trace_id, "event": "ask", "route": result["route"],
                             "grounded": result["grounded"], "attempts": result["attempts"],
                             "latency_ms": round((time.perf_counter() - t0) * 1000, 2), **metrics}))
        return {**result, "trace_id": trace_id, "metrics": metrics}

    return app


app = create_app() if os.environ.get("SUPPORT_DESK_EAGER") else None  # `uvicorn support_desk.service:create_app --factory`

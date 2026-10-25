"""Step-level tracing: latency, tokens, and cost for every node and LLM call.

Cost uses ILLUSTRATIVE per-token prices (see PRICE_*). Replace them with your
provider's real prices before treating the numbers as real spend.
"""
import json
import time
from contextlib import contextmanager
from pathlib import Path

import os

# ILLUSTRATIVE defaults. Set PRICE_IN_PER_1K / PRICE_OUT_PER_1K to your model's real prices.
PRICE_IN_PER_1K = float(os.environ.get("PRICE_IN_PER_1K", 0.00015))
PRICE_OUT_PER_1K = float(os.environ.get("PRICE_OUT_PER_1K", 0.0006))


def cost_usd(tokens_in: int, tokens_out: int) -> float:
    return tokens_in / 1000 * PRICE_IN_PER_1K + tokens_out / 1000 * PRICE_OUT_PER_1K


class Tracer:
    def __init__(self):
        self.events: list[dict] = []

    @contextmanager
    def span(self, name: str, **attrs):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.events.append({"type": "span", "name": name,
                                "ms": round((time.perf_counter() - t0) * 1000, 3), **attrs})

    def llm(self, node: str, tokens_in: int, tokens_out: int, seconds: float):
        cost = cost_usd(tokens_in, tokens_out)
        self.events.append({"type": "llm", "node": node, "tokens_in": tokens_in,
                            "tokens_out": tokens_out, "seconds": round(seconds, 6),
                            "cost_usd": round(cost, 8)})

    def summary(self) -> dict:
        llm = [e for e in self.events if e["type"] == "llm"]
        spans = [e for e in self.events if e["type"] == "span"]
        t_in = sum(e["tokens_in"] for e in llm)
        t_out = sum(e["tokens_out"] for e in llm)
        secs = sum(e["seconds"] for e in llm)
        return {
            "llm_calls": len(llm), "tokens_in": t_in, "tokens_out": t_out,
            "cost_usd": round(sum(e["cost_usd"] for e in llm), 8),
            "llm_seconds": round(secs, 6),
            "tokens_per_sec": round(t_out / secs, 1) if secs > 0 else None,
            "wall_ms": round(sum(e["ms"] for e in spans if e["name"] == "request"), 3),
        }

    def dump(self, path: str | Path):
        with open(path, "a") as f:
            for e in self.events:
                f.write(json.dumps(e) + "\n")

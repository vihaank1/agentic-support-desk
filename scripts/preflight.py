"""Cheap sanity check before a real eval run: one tiny call, prints the reply, token usage, and cost.

Usage: GEMINI_API_KEY=... PYTHONPATH=src python scripts/preflight.py --llm gemini
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from support_desk.llm import make_llm  # noqa: E402
from support_desk.tracing import PRICE_IN_PER_1K, PRICE_OUT_PER_1K, cost_usd  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--llm", choices=["openai", "gemini"], required=True)
a = ap.parse_args()
llm = make_llm(a.llm, max_usd=0.05)
r = llm.complete("ROLE=specialist Reply with exactly: Final: ok", "QUESTION: ping\nSCRATCHPAD:\n")
print("reply:", repr(r.text))
print(f"tokens in/out: {r.tokens_in}/{r.tokens_out}  latency: {r.seconds:.2f}s")
print(f"cost @ PRICE_IN={PRICE_IN_PER_1K} PRICE_OUT={PRICE_OUT_PER_1K} per 1K: ${cost_usd(r.tokens_in, r.tokens_out):.6f}")
print("OK" if "ok" in r.text.lower() else "UNEXPECTED REPLY: check model name / prompt format")

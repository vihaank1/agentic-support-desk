"""CLI: python -m support_desk.run "How much is the Plus plan?" [--llm mock|openai]"""
import argparse
import json
from pathlib import Path

from .graph import ask, build_graph
from .llm import make_llm
from .retrieval import Retriever
from .tracing import Tracer

ROOT = Path(__file__).resolve().parents[2]


def make_app(llm_name: str = "mock"):
    tracer = Tracer()
    llm = make_llm(llm_name)
    return build_graph(llm, Retriever(ROOT / "data" / "docs"), tracer), tracer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("question")
    p.add_argument("--llm", choices=["mock", "openai", "gemini"], default="mock")
    p.add_argument("--trace", help="append JSONL trace events to this file")
    a = p.parse_args()
    app, tracer = make_app(a.llm)
    result = ask(app, tracer, a.question)
    print(json.dumps({**result, "metrics": tracer.summary()}, indent=2))
    if a.trace:
        tracer.dump(a.trace)


if __name__ == "__main__":
    main()

# Agentic Support Desk

A small multi-agent system built on **LangGraph**: a supervisor routes each question to a billing or technical specialist, the specialist runs a **ReAct** loop (retrieve / calculate tools), and a **self-reflection critic** checks the answer against retrieved evidence and sends it back once if it is not grounded. Every step is traced (latency, tokens, cost), and a labeled eval set measures routing and answer accuracy.

```
question -> supervisor -> billing | technical (ReAct: retrieve / calculate) -> critic -> answer
                                   ^------------- retry once with feedback + wider retrieval
```

## What is in here

| Piece | File | What it does |
|---|---|---|
| Hierarchical delegation | `src/support_desk/graph.py` | Supervisor only routes; specialists own the tools |
| ReAct loop | `graph.py` (`make_specialist`) | Thought / Action / Observation, max 4 steps, tool errors fed back as observations |
| Self-reflection critic | `graph.py` (`critic`) | Rejects amounts not in evidence, refusals despite evidence, ungrounded answers; one retry with feedback and `top_k` 3 to 5 |
| Retrieval | `retrieval.py` | Pure-Python **TF-IDF cosine** over the docs in `data/docs/` |
| Safe calculator tool | `tools.py` | AST-based arithmetic only; rejects names, calls, and huge exponents |
| Tracing | `tracing.py` | Per-node latency plus per-call tokens, cost, tokens/sec; JSONL export |
| Eval harness | `scripts/run_eval.py` | 12 labeled cases; route accuracy, answer accuracy, p50/p95 latency, avg tokens, avg cost |
| Tests | `tests/` | 23 tests, including critic catch-and-retry and persistent-hallucination flagging |

## Run it

```bash
pip install -r requirements.txt
PYTHONPATH=src python -m support_desk.run "How much is the overage charge if I use 8 GB on the Basic plan?"
PYTHONPATH=src python scripts/run_eval.py
python -m pytest -q
```

## Honest limitations (read before quoting any number)

- **The default model is a deterministic mock, not an LLM.** `MockLLM` is rule-based so everything runs offline and reproducibly. The eval results (12/12) therefore measure the **orchestration** (routing, ReAct plumbing, grounding critic), not language-model quality. The 12 cases were written alongside the mock, so 12/12 is a smoke test, not a benchmark.
- **Real-LLM mode is untested here.** `OpenAILLM` (`--llm openai`) is a thin adapter I have not run in CI; it needs an API key and network. Token and tokens/sec numbers are only meaningful in that mode. The mock approximates tokens by word count.
- **Costs are illustrative.** `PRICE_*` in `tracing.py` are placeholder prices, not any provider's real rates.
- **Retrieval is TF-IDF, not embeddings or a vector database.**
- **Not deployed to any cloud**, and the knowledge base is four short synthetic docs.
- The critic is heuristic (amount matching plus word overlap). It catches the failure modes in `tests/test_graph.py`; it is not a general hallucination detector.

## Where to take it next

Swap `MockLLM` for a real model and re-run the eval, replace TF-IDF with embeddings and a vector store, add per-step cost budgets, and grow the eval set with cases written independently of the agent.

# Agentic Support Desk

A small multi-agent system built on **LangGraph**: a supervisor routes each question to a billing or technical specialist, the specialist runs a **ReAct** loop (retrieve / calculate tools), and a **self-reflection critic** checks the answer against retrieved evidence and sends it back once if it is not grounded. Every step is traced (latency, tokens, cost). It ships with a labeled eval harness, a FastAPI service, a Dockerfile, and a Cloud Run deployment guide.

```
question -> supervisor -> billing | technical (ReAct: retrieve / calculate) -> critic -> answer
                                   ^------------- retry once with feedback + wider retrieval
```

## What is in here

| Piece | Where | What it does |
|---|---|---|
| Hierarchical delegation | `graph.py` | Supervisor only routes; specialists own the tools |
| ReAct loop | `graph.py` | Thought / Action / Observation, max 4 steps; tool errors go back as observations |
| Self-reflection critic | `graph.py` | Rejects amounts not found in evidence, refusals despite evidence, ungrounded answers; one retry with feedback and wider retrieval |
| Retrieval | `retrieval.py` | Pure-Python TF-IDF cosine over `data/docs/` |
| Safe calculator | `tools.py` | AST-based arithmetic only |
| Tracing | `tracing.py` | Per-node latency; per-call tokens, cost, tokens/sec; JSONL export |
| Providers | `llm.py` | `MockLLM` (offline), `OpenAILLM`, `GeminiLLM`, wrapped by `ResilientLLM` (retries with backoff, hard spend guard) |
| Service | `service.py` | `POST /ask`, `GET /healthz`, input validation, 502/503 error mapping, structured JSON logs |
| Eval | `scripts/run_eval.py` | Route/answer accuracy, p50/p95 latency, avg tokens and cost; `--cases` and `--max-usd` options |
| Deploy | `Dockerfile`, `DEPLOY.md` | Cloud Run source-deploy steps with secrets-based key handling |

## Run it

```bash
pip install -r requirements.txt
python -m pytest -q                                   # 34 tests
PYTHONPATH=src python scripts/run_eval.py              # dev set (written alongside the mock)
PYTHONPATH=src python scripts/run_eval.py --cases data/eval_cases_independent.json
PYTHONPATH=src python -m support_desk.run "How much is the overage charge if I use 8 GB on the Basic plan?"
```

Real-model runs (not performed in this repo's tests): first run `scripts/preflight.py --llm gemini` (one tiny call that prints tokens and cost), then set `OPENAI_API_KEY` or `GEMINI_API_KEY`, then `--llm openai|gemini --max-usd 0.50`. Set `PRICE_IN_PER_1K` / `PRICE_OUT_PER_1K` to your model's real prices; the defaults are illustrative.

## Results so far (mock model only)

| Set | Route accuracy | Answer accuracy | Notes |
|---|---|---|---|
| Dev set, 12 cases | 12/12 | 12/12 | Written alongside the mock, so this is a smoke test |
| Independent set, 19 cases | 16/19 | 14/19 | Paraphrases, multi-step math, unanswerable questions, prompt-injection attempts |

The independent set is deliberately harder than the mock, and the failures are informative:

- The mock's keyword router misroutes paraphrases ("no bars", "no internet").
- **The heuristic critic cannot detect topical drift.** On unanswerable questions ("CEO's phone number", "Platinum plan") the mock retrieves a loosely related sentence, and the critic accepts it because the words overlap the evidence. A lexical check cannot tell "related" from "answers the question". An LLM-as-judge critic is the natural next step.
- Multi-fact reasoning ("Plus price after autopay discount") fails because the mock only chains one calculation pattern.
- Prompt-injection attempts in the question did not change answers, but that is a property of the mock ignoring instructions, not evidence of robustness in a real model.

These numbers say nothing about how a real LLM would score. Run the independent set with a real model for that.

## Honest limitations

- **Default model is a deterministic mock, not an LLM.** Scores measure orchestration, not model quality.
- **`OpenAILLM`, `GeminiLLM`, and the Docker/Cloud Run path have not been run** by the author of this repo; they are written to documented SDK usage and are untested here. `DEPLOY.md` marks the deployment as not yet done.
- Costs are illustrative unless you set real prices. The spend guard is per process, not a global budget.
- Retrieval is TF-IDF, not embeddings or a vector database. The knowledge base is four short synthetic documents.
- The critic is heuristic: amount matching plus word overlap.

## Next steps

Run both eval sets with a real model and record the results here; deploy to Cloud Run and note the observed latency and cost per request; add an LLM-as-judge critic and compare it against the heuristic one on the independent set; swap TF-IDF for embeddings plus a vector store; grow the eval set with questions from real users.

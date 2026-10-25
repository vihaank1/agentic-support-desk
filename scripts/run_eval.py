"""Run the labeled eval set and print routing/answer accuracy + latency/token/cost stats."""
import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from support_desk.graph import ask, build_graph  # noqa: E402
from support_desk.llm import make_llm  # noqa: E402
from support_desk.retrieval import Retriever  # noqa: E402
from support_desk.tracing import Tracer  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--llm", choices=["mock", "openai", "gemini"], default="mock")
    p.add_argument("--cases", default=str(ROOT / "data" / "eval_cases.json"))
    p.add_argument("--max-usd", type=float, default=None, help="hard spend guard for real-model runs")
    p.add_argument("--out", default=str(ROOT / "eval_report.json"))
    a = p.parse_args()
    cases = json.loads(Path(a.cases).read_text())
    retriever = Retriever(ROOT / "data" / "docs")
    llm = make_llm(a.llm, max_usd=a.max_usd)

    rows, route_ok, ans_ok, wall, toks, cost = [], 0, 0, [], [], []
    for c in cases:
        tracer = Tracer()
        res = ask(build_graph(llm, retriever, tracer), tracer, c["question"])
        m = tracer.summary()
        r_ok = res["route"] == c["route"]
        a_ok = c["expect"].lower() in res["answer"].lower()
        route_ok += r_ok
        ans_ok += a_ok
        wall.append(m["wall_ms"]); toks.append(m["tokens_in"] + m["tokens_out"]); cost.append(m["cost_usd"])
        rows.append({"id": c["id"], "route_ok": r_ok, "answer_ok": a_ok, "attempts": res["attempts"],
                     "answer": res["answer"], **m})
        print(f"{'PASS' if (r_ok and a_ok) else 'FAIL'}  {c['id']:<18} route={res['route']:<9} {res['answer']}")

    n = len(cases)
    report = {
        "llm": llm.name, "cases": n,
        "route_accuracy": round(route_ok / n, 3), "answer_accuracy": round(ans_ok / n, 3),
        "p50_wall_ms": round(statistics.median(wall), 2),
        "p95_wall_ms": round(sorted(wall)[int(0.95 * (n - 1))], 2),
        "avg_tokens_per_request": round(statistics.mean(toks), 1),
        "avg_cost_usd_per_request_illustrative": round(statistics.mean(cost), 8),
        "rows": rows,
    }
    Path(a.out).write_text(json.dumps(report, indent=2))
    print(f"\n[{llm.name}] route {route_ok}/{n}  answer {ans_ok}/{n}  p50 {report['p50_wall_ms']} ms  "
          f"avg tokens {report['avg_tokens_per_request']}")


if __name__ == "__main__":
    main()

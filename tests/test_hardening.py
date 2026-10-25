import pytest
from fastapi.testclient import TestClient

from conftest import ROOT
from support_desk.llm import BudgetExceeded, LLMResult, MockLLM, ResilientLLM
from support_desk.service import create_app


class Flaky:
    name = "flaky"

    def __init__(self, fail_times):
        self.fail_times, self.calls = fail_times, 0

    def complete(self, system, prompt):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise ConnectionError("boom")
        return LLMResult("ok", 1000, 1000, 0.01)


def test_retries_then_succeeds_with_backoff():
    sleeps = []
    llm = ResilientLLM(Flaky(2), retries=3, base_delay=0.5, sleep=sleeps.append)
    assert llm.complete("s", "p").text == "ok"
    assert sleeps == [0.5, 1.0] and llm.failed_attempts == 2


def test_gives_up_after_retries():
    llm = ResilientLLM(Flaky(99), retries=2, sleep=lambda s: None)
    with pytest.raises(RuntimeError, match="3 attempts"):
        llm.complete("s", "p")


def test_failed_calls_do_not_count_as_spend():
    llm = ResilientLLM(Flaky(1), retries=2, sleep=lambda s: None)
    llm.complete("s", "p")
    assert llm.spent_usd == pytest.approx(0.00015 + 0.0006)


def test_spend_guard_blocks_next_call():
    llm = ResilientLLM(Flaky(0), max_usd=0.0005, sleep=lambda s: None)
    llm.complete("s", "p")  # spends 0.00075 > cap
    with pytest.raises(BudgetExceeded):
        llm.complete("s", "p")


def test_api_ask_and_health():
    client = TestClient(create_app(llm=MockLLM()))
    assert client.get("/healthz").json() == {"status": "ok", "provider": "mock"}
    r = client.post("/ask", json={"question": "What is the autopay discount?"})
    body = r.json()
    assert r.status_code == 200 and "$5" in body["answer"] and body["grounded"] and body["trace_id"]


def test_api_validates_input():
    client = TestClient(create_app(llm=MockLLM()))
    assert client.post("/ask", json={"question": "hi"}).status_code == 422
    assert client.post("/ask", json={"question": "x" * 501}).status_code == 422


def test_api_surfaces_upstream_failure_as_502():
    class Down:
        name = "down"

        def complete(self, system, prompt):
            raise ConnectionError("no network")

    client = TestClient(create_app(llm=Down()))
    assert client.post("/ask", json={"question": "What is the late fee?"}).status_code == 502


def test_independent_eval_regression_floor_for_mock():
    """Pins the mock's known score on the independent set so regressions are visible (it is NOT a quality claim)."""
    import json

    from support_desk.graph import ask, build_graph
    from support_desk.retrieval import Retriever
    from support_desk.tracing import Tracer

    cases = json.loads((ROOT / "data" / "eval_cases_independent.json").read_text())
    r = Retriever(ROOT / "data" / "docs")
    ok = 0
    for c in cases:
        t = Tracer()
        res = ask(build_graph(MockLLM(), r, t), t, c["question"])
        ok += c["expect"].lower() in res["answer"].lower()
    assert ok >= 14


class Scripted:
    """Returns canned specialist replies in order (supervisor always routes to billing)."""
    name = "scripted"

    def __init__(self, replies):
        self.replies, self.i = list(replies), 0

    def complete(self, system, prompt):
        if system.startswith("ROLE=supervisor"):
            return LLMResult("billing", 5, 1, 0.0)
        r = self.replies[min(self.i, len(self.replies) - 1)]
        self.i += 1
        return LLMResult(r, 20, 10, 0.0)


def _run(llm, q="What is the late fee?"):
    from support_desk.graph import ask, build_graph
    from support_desk.retrieval import Retriever
    from support_desk.tracing import Tracer
    t = Tracer()
    return ask(build_graph(llm, Retriever(ROOT / "data" / "docs"), t), t, q)


def test_action_and_premature_final_in_same_message_ignores_the_final():
    llm = Scripted([
        "Thought: look it up\nAction: retrieve[late fee]\nObservation: made up\nFinal: The late fee is $99.",
        "Final: A late fee of $7 applies when a payment is more than 10 days overdue.",
    ])
    res = _run(llm)
    assert "$7" in res["answer"] and "$99" not in res["answer"] and res["grounded"]


def test_malformed_reply_without_action_or_final_degrades_to_refusal():
    from support_desk.llm import REFUSAL
    res = _run(Scripted(["I think the fee might be high."]))
    assert res["answer"] == REFUSAL


def test_calculator_error_is_returned_as_observation_not_crash():
    llm = Scripted(["Action: calculate[__import__('os')]", "Final: I don't have enough information to answer that."])
    res = _run(llm)
    assert "don't have enough information" in res["answer"]

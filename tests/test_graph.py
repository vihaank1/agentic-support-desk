import json

import pytest

from conftest import ROOT
from support_desk.graph import ask, build_graph
from support_desk.llm import REFUSAL, LLMResult, MockLLM
from support_desk.retrieval import Retriever
from support_desk.tracing import Tracer

CASES = json.loads((ROOT / "data" / "eval_cases.json").read_text())
RETRIEVER = Retriever(ROOT / "data" / "docs")


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_eval_cases_with_mock(case):
    t = Tracer()
    res = ask(build_graph(MockLLM(), RETRIEVER, t), t, case["question"])
    assert res["route"] == case["route"]
    assert case["expect"].lower() in res["answer"].lower()
    assert res["grounded"]


class HallucinatingThenCorrect:
    """Stub model: first attempt invents a fee; after critic feedback it answers from evidence."""
    name = "stub"

    def __init__(self):
        self.mock = MockLLM()

    def complete(self, system, prompt):
        if system.startswith("ROLE=supervisor"):
            return self.mock.complete(system, prompt)
        if "CRITIC FEEDBACK" not in prompt:
            return LLMResult("Final: The late fee is $99.", 10, 6, 0.0)
        return self.mock.complete(system, prompt)


def test_critic_catches_ungrounded_amount_and_retry_fixes_it():
    t = Tracer()
    res = ask(build_graph(HallucinatingThenCorrect(), RETRIEVER, t), t, "What is the late fee?")
    assert res["attempts"] == 2
    assert "$7" in res["answer"] and "$99" not in res["answer"]


class AlwaysHallucinates:
    name = "stub"

    def __init__(self):
        self.mock = MockLLM()

    def complete(self, system, prompt):
        if system.startswith("ROLE=supervisor"):
            return self.mock.complete(system, prompt)
        return LLMResult("Final: The late fee is $99.", 10, 6, 0.0)


def test_persistent_hallucination_is_flagged_not_hidden():
    t = Tracer()
    res = ask(build_graph(AlwaysHallucinates(), RETRIEVER, t), t, "What is the late fee?")
    assert res["grounded"] is False  # surfaced to the caller instead of silently returned
    assert res["attempts"] == 2


def test_unanswerable_refuses():
    t = Tracer()
    res = ask(build_graph(MockLLM(), RETRIEVER, t), t, "What is the favorite color of the founder?")
    assert res["answer"] == REFUSAL


def test_tracer_records_tokens_and_cost():
    t = Tracer()
    ask(build_graph(MockLLM(), RETRIEVER, t), t, "What is the autopay discount?")
    s = t.summary()
    assert s["llm_calls"] >= 2 and s["tokens_in"] > 0 and s["cost_usd"] > 0

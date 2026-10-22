"""LLM providers behind one tiny interface.

MockLLM is a deterministic, rule-based stand-in so the graph, tests, and eval
harness run offline with no API key. It is NOT a language model; its eval
numbers measure the orchestration (routing, ReAct loop, grounding critic),
not model quality. OpenAILLM is an optional adapter for real runs.
"""
import os
import re
import time
from dataclasses import dataclass

from .retrieval import tokenize

REFUSAL = "I don't have enough information to answer that."
TECH_WORDS = ("no service", "reset", "restart", "not working", "sim", "apn", "signal",
              "outage", "network")


@dataclass
class LLMResult:
    text: str
    tokens_in: int
    tokens_out: int
    seconds: float


def _approx_tokens(text: str) -> int:
    return max(1, int(len(text.split()) * 1.3))


class MockLLM:
    name = "mock"

    def complete(self, system: str, prompt: str) -> LLMResult:
        t0 = time.perf_counter()
        role = system.split()[0].replace("ROLE=", "")
        text = {"supervisor": self._supervisor, "specialist": self._specialist}[role](prompt)
        return LLMResult(text, _approx_tokens(system + prompt), _approx_tokens(text),
                         time.perf_counter() - t0)

    @staticmethod
    def _question(prompt: str) -> str:
        return re.search(r"QUESTION: (.*)", prompt).group(1)

    def _supervisor(self, prompt: str) -> str:
        q = self._question(prompt).lower()
        return "technical" if any(w in q for w in TECH_WORDS) else "billing"

    def _specialist(self, prompt: str) -> str:
        q = self._question(prompt)
        scratch = prompt.split("SCRATCHPAD:", 1)[1]
        if "Observation:" not in scratch:
            return f"Thought: I need evidence from the knowledge base.\nAction: retrieve[{q}]"
        if "Action: calculate[" in scratch:
            result = re.findall(r"Observation: ([\d.]+)\s*$", scratch.strip(), re.M)[-1]
            return f"Final: Based on the policy, the amount is ${float(result):g}."
        obs = scratch.split("Observation:", 1)[1].strip()
        if obs == "NO_EVIDENCE":
            return f"Final: {REFUSAL}"
        ql = q.lower()
        m_gb = re.search(r"(\d+)\s*gb", ql)
        m_plan = re.search(r"\b(basic|plus)\b", ql)
        if "overage" in ql and m_gb and m_plan:
            allow = re.search(rf"{m_plan.group(1)} plan costs \$\d+ per month and includes (\d+) GB",
                              obs, re.I)
            rate = re.search(r"\$(\d+) per GB", obs)
            if allow and rate:
                expr = f"({m_gb.group(1)}-{allow.group(1)})*{rate.group(1)}"
                return f"Thought: Overage is usage minus allowance times the rate.\nAction: calculate[{expr}]"
        m_days = re.search(r"(\d+)\s*days?", ql)
        price = re.search(r"day pass costs \$(\d+)", obs)
        if "pass" in ql and m_days and price:
            return f"Thought: Multiply days by the daily price.\nAction: calculate[{m_days.group(1)}*{price.group(1)}]"
        qtok = set(tokenize(q))
        sentences = [s.strip() for s in obs.split("\n") if s.strip()]
        best = max(sentences, key=lambda s: len(qtok & set(tokenize(s))))
        return f"Final: {best}"


class OpenAILLM:
    """Optional real provider. Untested in this repo's CI (needs network + key)."""
    name = "openai"

    def __init__(self, model: str = "gpt-4o-mini"):
        from openai import OpenAI  # imported lazily so the base install stays light
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model = model

    def complete(self, system: str, prompt: str) -> LLMResult:
        t0 = time.perf_counter()
        r = self.client.chat.completions.create(
            model=self.model, temperature=0,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}])
        return LLMResult(r.choices[0].message.content, r.usage.prompt_tokens,
                         r.usage.completion_tokens, time.perf_counter() - t0)

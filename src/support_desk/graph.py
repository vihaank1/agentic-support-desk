"""LangGraph workflow: supervisor -> specialist (ReAct loop) -> critic -> (retry | end).

Hierarchical delegation: the supervisor only routes; specialists own tools.
Self-reflection: a grounding critic checks the answer against retrieved
evidence and sends the request back once, with feedback and wider retrieval.
"""
import re
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from .llm import REFUSAL
from .retrieval import Retriever, tokenize
from .tools import calculate
from .tracing import Tracer

MAX_STEPS = 4
MAX_ATTEMPTS = 2

SPECIALIST_SYSTEM = (
    "ROLE=specialist You are a telecom support specialist ({domain}). Use the ReAct format. "
    "Reply with either 'Thought: ...\\nAction: retrieve[query]', "
    "'Action: calculate[expression]', or 'Final: answer'. "
    "Answer only from retrieved evidence; if there is none, say you don't have the information."
)


class State(TypedDict, total=False):
    question: str
    route: str
    evidence: list
    calc_values: list
    answer: str
    attempts: int
    feedback: str
    top_k: int
    critic_ok: bool


def _money(text: str) -> set:
    return set(re.findall(r"\$(\d+(?:\.\d+)?)", text))


def build_graph(llm, retriever: Retriever, tracer: Tracer):
    def supervisor(state: State):
        with tracer.span("supervisor"):
            r = llm.complete("ROLE=supervisor Route the question to 'billing' or 'technical'.",
                             f"QUESTION: {state['question']}")
            tracer.llm("supervisor", r.tokens_in, r.tokens_out, r.seconds)
            route = r.text.strip().lower()
            route = route if route in ("billing", "technical") else "billing"
        return {"route": route, "attempts": 0, "top_k": 3, "feedback": "",
                "evidence": [], "calc_values": []}

    def make_specialist(domain: str):
        def run(state: State):
            with tracer.span(f"{domain}_agent", attempt=state.get("attempts", 0)):
                scratch, evidence, calc_values, answer = "", [], [], None
                for _ in range(MAX_STEPS):
                    fb = f"\nCRITIC FEEDBACK: {state['feedback']}" if state.get("feedback") else ""
                    prompt = f"QUESTION: {state['question']}{fb}\nSCRATCHPAD:\n{scratch}"
                    r = llm.complete(SPECIALIST_SYSTEM.format(domain=domain), prompt)
                    tracer.llm(f"{domain}_agent", r.tokens_in, r.tokens_out, r.seconds)
                    text = r.text.strip()
                    # Prefer an Action over a Final in the same message: a model that writes both has
                    # not seen the observation yet, so its Final would be unsupported.
                    m = re.search(r"Action:\s*(retrieve|calculate)\[([^\]\n]*)\]", text)
                    if not m and "Final:" in text:
                        answer = text.split("Final:", 1)[1].strip()
                        break
                    if not m:
                        break
                    text = text[:m.end()]  # drop any hallucinated Observation/Final after the action
                    tool, arg = m.groups()
                    if tool == "retrieve":
                        hits = retriever.search(arg, top_k=state.get("top_k", 3))
                        evidence = [h["text"] for h in hits]
                        obs = "\n".join(evidence) if evidence else "NO_EVIDENCE"
                    else:
                        try:
                            value = calculate(arg)
                            calc_values.append(value)
                            obs = f"{value:g}"
                        except Exception as e:  # tool errors go back to the model as observations
                            obs = f"ERROR: {e}"
                    scratch += f"{text}\nObservation: {obs}\n"
                return {"answer": answer or REFUSAL, "evidence": evidence,
                        "calc_values": calc_values}
        return run

    def critic(state: State):
        with tracer.span("critic"):
            answer, evidence = state["answer"], state.get("evidence", [])
            ev_text = " ".join(evidence)
            computed = {f"{v:g}" for v in state.get("calc_values", [])}
            ok, why = True, ""
            if answer == REFUSAL:
                if evidence:
                    ok, why = False, "You refused although evidence was retrieved; use it."
            else:
                bad = _money(answer) - _money(ev_text) - computed
                content = set(tokenize(answer))
                support = set(tokenize(ev_text)) | set(tokenize(state["question"]))
                overlap = len(content & support) / max(1, len(content))
                if bad:
                    ok, why = False, f"Amounts not found in evidence: {sorted(bad)}."
                elif not evidence:
                    ok, why = False, "No evidence was retrieved to support the answer."
                elif computed and _money(answer) & computed:
                    pass  # amount comes from a tool call over retrieved evidence
                elif overlap < 0.5:
                    ok, why = False, "Answer is not grounded in the retrieved evidence."
        attempts = state.get("attempts", 0) + 1
        return {"critic_ok": ok, "attempts": attempts,
                "feedback": "" if ok else why, "top_k": 5}

    def after_critic(state: State):
        if state["critic_ok"] or state["attempts"] >= MAX_ATTEMPTS:
            return END
        return state["route"]

    g = StateGraph(State)
    g.add_node("supervisor", supervisor)
    g.add_node("billing", make_specialist("billing"))
    g.add_node("technical", make_specialist("technical"))
    g.add_node("critic", critic)
    g.add_edge(START, "supervisor")
    g.add_conditional_edges("supervisor", lambda s: s["route"], ["billing", "technical"])
    g.add_edge("billing", "critic")
    g.add_edge("technical", "critic")
    g.add_conditional_edges("critic", after_critic, ["billing", "technical", END])
    return g.compile()


def ask(app, tracer: Tracer, question: str) -> dict:
    with tracer.span("request"):
        out = app.invoke({"question": question})
    return {"question": question, "route": out["route"], "answer": out["answer"],
            "grounded": out["critic_ok"], "attempts": out["attempts"]}

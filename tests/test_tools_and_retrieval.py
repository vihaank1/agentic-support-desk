import pytest

from conftest import ROOT
from support_desk.retrieval import Retriever
from support_desk.tools import calculate


def test_calculate_arithmetic():
    assert calculate("(8-5)*10") == 30
    assert calculate("-3 + 2*4") == 5


@pytest.mark.parametrize("bad", ["__import__('os').system('ls')", "open('x')", "a+1", "2**9999"])
def test_calculate_rejects_non_arithmetic(bad):
    with pytest.raises(Exception):
        calculate(bad)


def test_retrieval_finds_relevant_chunk():
    r = Retriever(ROOT / "data" / "docs")
    hits = r.search("What is the autopay discount?", top_k=1)
    assert "autopay" in hits[0]["text"].lower()


def test_retrieval_returns_nothing_for_offtopic():
    r = Retriever(ROOT / "data" / "docs")
    assert r.search("favorite color of the company founder") == []

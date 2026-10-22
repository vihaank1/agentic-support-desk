"""Pure-Python TF-IDF retrieval over the knowledge-base docs.

This is sparse TF-IDF cosine similarity, NOT dense embeddings or a vector
database. It keeps the project dependency-free and deterministic for tests.
"""
import math
import re
from collections import Counter
from pathlib import Path

STOP = {
    "a", "an", "the", "is", "are", "of", "to", "and", "or", "in", "on", "if", "i",
    "my", "it", "do", "does", "how", "what", "when", "can", "get", "after", "for",
    "be", "that", "this", "with", "at", "as", "by", "from", "you", "your", "we",
    "should", "would", "will", "try", "first", "much", "many",
}


def tokenize(text: str) -> list[str]:
    out = []
    for w in re.findall(r"[a-z0-9$]+", text.lower()):
        if w in STOP:
            continue
        if len(w) > 3 and w.endswith("s"):
            w = w[:-1]  # crude stemming: plans -> plan, costs -> cost
        out.append(w)
    return out


class Retriever:
    def __init__(self, docs_dir: str | Path, min_score: float = 0.12):
        self.min_score = min_score
        self.chunks: list[dict] = []
        for path in sorted(Path(docs_dir).glob("*.md")):
            for line in path.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    self.chunks.append({"source": path.name, "text": line})
        n = len(self.chunks)
        df = Counter()
        for c in self.chunks:
            c["tf"] = Counter(tokenize(c["text"]))
            df.update(c["tf"].keys())
        self.idf = {t: math.log((1 + n) / (1 + d)) + 1 for t, d in df.items()}
        for c in self.chunks:
            c["vec"] = self._vec(c["tf"])

    def _vec(self, tf: Counter) -> dict:
        v = {t: f * self.idf.get(t, 0.0) for t, f in tf.items() if t in self.idf}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {t: x / norm for t, x in v.items()}

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        q = self._vec(Counter(tokenize(query)))
        scored = []
        for c in self.chunks:
            s = sum(w * c["vec"].get(t, 0.0) for t, w in q.items())
            if s >= self.min_score:
                scored.append({"score": round(s, 3), "source": c["source"], "text": c["text"]})
        scored.sort(key=lambda x: -x["score"])
        return scored[:top_k]

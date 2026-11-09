"""Source pack loading and BM25 retrieval. Only passages returned here are shown to the model."""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .textutil import normalize

_STOPWORDS = set("""
a an the and or but if of to in on at by for with from as is are was were be been being it its this that these those
there their they them he she his her we our you your i me my not no do does did has have had will would shall should
can could may might must so than then too very just about into over under also any all some such only own same
""".split())
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    text = re.sub(r"(?<=\d),(?=\d)", "", normalize(text)).casefold()
    tokens = []
    for t in _TOKEN_RE.findall(text):
        if t in _STOPWORDS or (len(t) < 2 and not t.isdigit()):
            continue
        if len(t) > 4 and t.endswith("s") and not t.endswith("ss"):
            t = t[:-1]
        tokens.append(t)
    return tokens


@dataclass
class SourceDoc:
    id: str
    title: str
    date: str
    category: str
    url: str
    licence: str
    body: str
    body_norm: str = field(init=False)

    def __post_init__(self) -> None:
        self.body_norm = normalize(self.body)


@dataclass
class Passage:
    source_id: str
    index: int
    text: str
    title: str
    date: str
    score: float = 0.0


def parse_doc(path: Path) -> SourceDoc:
    raw = path.read_text(encoding="utf-8")
    header, sep, body = raw.partition("\n---\n")
    if not sep:
        raise ValueError(f"{path.name}: missing '---' line between header and body")
    meta: dict[str, str] = {}
    for line in header.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip().lower()] = v.strip()
    if not meta.get("id"):
        raise ValueError(f"{path.name}: header needs an 'id:' line")
    return SourceDoc(
        id=meta["id"], title=meta.get("title", meta["id"]), date=meta.get("date", ""),
        category=meta.get("category", "general"), url=meta.get("url", ""),
        licence=meta.get("licence", ""), body=body.strip(),
    )


class BM25:
    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tfs = [Counter(doc) for doc in corpus]
        self.lens = [len(doc) for doc in corpus]
        self.avgdl = (sum(self.lens) / len(corpus)) if corpus else 1.0
        n = len(corpus)
        df = Counter(t for doc in corpus for t in set(doc))
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def scores(self, query: list[str]) -> list[float]:
        out = []
        for tf, dl in zip(self.tfs, self.lens):
            s = 0.0
            for t in query:
                f = tf.get(t, 0)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            out.append(s)
        return out


class SourcePack:
    def __init__(self, docs: list[SourceDoc]):
        ids = [d.id for d in docs]
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            raise ValueError(f"duplicate source ids: {sorted(dupes)}")
        self.docs: dict[str, SourceDoc] = {d.id: d for d in docs}
        self.passages: list[Passage] = []
        corpus: list[list[str]] = []
        for d in docs:
            title_tokens = tokenize(d.title)
            for i, para in enumerate(p for p in re.split(r"\n\s*\n", d.body) if p.strip()):
                self.passages.append(Passage(d.id, i, normalize(para), d.title, d.date))
                corpus.append(tokenize(para) + title_tokens)
        self.index = BM25(corpus)

    @classmethod
    def load(cls, directory: Path) -> "SourcePack":
        files = sorted(Path(directory).glob("*.txt"))
        return cls([parse_doc(f) for f in files])

    def search(self, query: str, k: int = 3) -> list[Passage]:
        q = tokenize(query)
        if not q or not self.passages:
            return []
        scored = sorted(zip(self.index.scores(q), self.passages), key=lambda x: x[0], reverse=True)
        out = []
        for score, p in scored[:k]:
            if score <= 0:
                break
            out.append(Passage(p.source_id, p.index, p.text, p.title, p.date, round(score, 3)))
        return out

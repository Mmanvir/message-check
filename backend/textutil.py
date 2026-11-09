"""Deterministic text helpers: normalisation, link extraction/defanging, injection scan."""
from __future__ import annotations

import json
import re
import unicodedata

_CHAR_MAP = str.maketrans({
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "–": "-", "—": "-", "−": "-",
    " ": " ", "​": "", "‌": "", "‍": "", "﻿": "",
})


def normalize(text: str) -> str:
    """NFC, straight quotes, plain dashes, collapsed whitespace and line breaks."""
    text = unicodedata.normalize("NFC", text or "").translate(_CHAR_MAP)
    return re.sub(r"\s+", " ", text).strip()


# ---------- links ----------

_TLDS = r"(?:com|in|org|net|co|info|xyz|link|ly|me|io|biz|online|site|top|club|app|gov\.in|nic\.in|co\.in)"
_URL_RE = re.compile(
    r"(?i)(?<![\w@])("
    r"(?:https?|hxxps?)://[^\s<>\"']+"
    r"|www\.[^\s<>\"']+"
    r"|(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?(?:\.|\[\.\]))+" + _TLDS + r"(?:/[^\s<>\"']*)?"
    r")(?![\w-])"
)


def defang(url: str) -> str:
    url = url.strip().rstrip(".,;:!?)]}'\"")
    m = re.match(r"(?i)^(https?|hxxps?)://", url)
    scheme, rest = ("", url)
    if m:
        scheme = m.group(1).lower().replace("http", "hxxp") + "://"
        rest = url[m.end():]
    host, sep, path = rest.partition("/")
    host = host.replace("[.]", ".").replace(".", "[.]")
    return f"{scheme}{host}{sep}{path}"


def extract_links(text: str) -> list[str]:
    """Return defanged links found in text. Links are never fetched."""
    out: list[str] = []
    for m in _URL_RE.finditer(text or ""):
        d = defang(m.group(1))
        if d and d not in out:
            out.append(d)
    return out


def strip_links(text: str) -> str:
    return re.sub(r"\s{2,}", " ", _URL_RE.sub("", text or "")).strip()


# ---------- injection / pressure ----------

_INJECTION_PATTERNS: list[tuple[str, str]] = [
    (r"\b(?:ai|assistant|chatbot|chat\s*gpt|gpt|bot|model|llm)\s*[:,-]\s*\w", "Text addresses an AI directly"),
    (r"\bignore\s+(?:all\s+|any\s+|the\s+)?(?:previous|prior|above|earlier)\b", "Text asks to ignore previous instructions"),
    (r"\b(?:say|confirm|mark|reply|tell|answer)\b[^.\n]{0,30}\b(?:true|real|genuine|verified|correct|authentic)\b",
     "Text asks for a 'true' verdict"),
    (r"\bsystem\s+prompt\b|\byou\s+are\s+now\b|\bdisregard\b|\bnew\s+instructions?\b", "Text tries to change instructions"),
]
_PRESSURE_PATTERNS: list[tuple[str, str]] = [
    (r"\b(?:share|forward|send)\b[^.\n]{0,25}\b\d+\s*(?:people|persons|groups|friends|contacts)\b", "Chain-message pressure to forward"),
    (r"\b(?:urgent|last\s+date|before\s+(?:sunday|midnight|tonight)|only\s+today)\b", "Urgency pressure"),
]


def scan_injection(text: str) -> list[str]:
    """Flag instruction-like or pressure text. Flags are shown to the volunteer; text is never obeyed."""
    flags: list[str] = []
    t = normalize(text)
    for pattern, label in _INJECTION_PATTERNS + _PRESSURE_PATTERNS:
        m = re.search(pattern, t, re.IGNORECASE)
        if m:
            snippet = t[max(0, m.start() - 10): m.end() + 25].strip()
            flags.append(f"{label}: \"…{snippet}…\" — ignored as data")
    return flags


def is_injection_like(sentence: str) -> bool:
    return any(re.search(p, normalize(sentence), re.IGNORECASE) for p, _ in _INJECTION_PATTERNS)


# ---------- language / JSON ----------

def detect_script_language(text: str) -> str:
    if re.search(r"[਀-੿]", text):
        return "pa"
    if re.search(r"[ऀ-ॿ]", text):
        return "hi"
    return "en"


def parse_json_object(text: str) -> dict:
    """Parse a JSON object from model output, tolerating ``` fences or stray prose."""
    t = (text or "").strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.IGNORECASE)
    try:
        obj = json.loads(t)
    except json.JSONDecodeError:
        start, end = t.find("{"), t.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("no JSON object found in model output")
        obj = json.loads(t[start:end + 1])
    if not isinstance(obj, dict):
        raise ValueError("model output is not a JSON object")
    return obj

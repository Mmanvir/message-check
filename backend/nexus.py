"""Nexus API client (OpenAI-compatible endpoints assumed) plus an offline mock.

Endpoints used — confirm paths against the Nexus docs and adjust here only:
  POST {base}/chat/completions      LLM + vision (image as data URL)
  POST {base}/audio/transcriptions  speech-to-text (multipart)
  POST {base}/audio/speech          text-to-speech (returns audio bytes, WAV requested)
"""
from __future__ import annotations

import base64
import io
import json
import re
import time
import wave
from dataclasses import dataclass
from typing import Protocol

import httpx

from .config import Settings
from .prompts import FORWARD_CLOSE, FORWARD_OPEN, VERDICTS_CLOSE, VERDICTS_OPEN
from .textutil import detect_script_language, is_injection_like, strip_links


class NexusError(RuntimeError):
    """Raised with a message that is safe to show (never contains the key)."""


@dataclass
class CallResult:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0


class NexusLike(Protocol):
    def chat(self, messages: list[dict], *, model: str | None = None, task: str = "",
             json_mode: bool = True) -> CallResult: ...
    def vision(self, image: bytes, mime: str, prompt: str) -> CallResult: ...
    def transcribe(self, audio: bytes, filename: str, mime: str) -> CallResult: ...
    def tts(self, text: str) -> bytes: ...


_RETRY_STATUS = {408, 429, 500, 502, 503, 504}


class NexusClient:
    def __init__(self, settings: Settings, http: httpx.Client | None = None):
        self.s = settings
        self.http = http or httpx.Client(
            base_url=settings.base_url.rstrip("/") + "/",
            timeout=settings.timeout,
            headers={"Authorization": f"Bearer {settings.api_key}"},
        )

    def _post(self, path: str, **kwargs) -> httpx.Response:
        delay, last = 1.0, ""
        for attempt in range(3):
            try:
                r = self.http.post(path, **kwargs)
            except (httpx.TimeoutException, httpx.TransportError) as e:
                last = type(e).__name__
            else:
                if r.status_code < 400:
                    return r
                if r.status_code not in _RETRY_STATUS:
                    raise NexusError(f"Nexus returned HTTP {r.status_code} for {path}: {r.text[:300]}")
                last = f"HTTP {r.status_code}"
            if attempt < 2:
                time.sleep(delay)
                delay *= 2
        raise NexusError(f"Nexus is not responding ({last}) for {path} after 3 attempts. Please try again.")

    def chat(self, messages, *, model=None, task="", json_mode=True) -> CallResult:
        model = model or self.s.llm_model
        body: dict = {"model": model, "messages": messages, "temperature": 0}
        if json_mode and self.s.json_mode:
            body["response_format"] = {"type": "json_object"}
        t0 = time.perf_counter()
        r = self._post("chat/completions", json=body)
        dt = time.perf_counter() - t0
        try:
            d = r.json()
            text = d["choices"][0]["message"]["content"] or ""
        except (ValueError, KeyError, IndexError, TypeError) as e:
            raise NexusError(f"Unexpected chat response from Nexus ({type(e).__name__})") from None
        usage = d.get("usage") or {}
        return CallResult(text, model, int(usage.get("prompt_tokens") or 0),
                          int(usage.get("completion_tokens") or 0), round(dt, 3))

    def vision(self, image, mime, prompt) -> CallResult:
        url = f"data:{mime};base64,{base64.b64encode(image).decode()}"
        messages = [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": url}},
        ]}]
        return self.chat(messages, model=self.s.vision_model, task="vision", json_mode=False)

    def transcribe(self, audio, filename, mime) -> CallResult:
        t0 = time.perf_counter()
        r = self._post("audio/transcriptions", files={"file": (filename, audio, mime)},
                       data={"model": self.s.stt_model})
        dt = time.perf_counter() - t0
        try:
            text = r.json().get("text", "")
        except ValueError:
            text = r.text
        return CallResult(text or "", self.s.stt_model, seconds=round(dt, 3))

    def tts(self, text) -> bytes:
        r = self._post("audio/speech", json={"model": self.s.tts_model, "input": text,
                                             "voice": self.s.tts_voice, "response_format": "wav"})
        return r.content


# ---------------- offline mock ----------------

def make_silent_wav(seconds: float, rate: int = 8000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(rate * max(seconds, 0.5)))
    return buf.getvalue()


class MockNexus:
    """Deterministic stand-in so the whole app runs without a Nexus key. Not for evaluation."""

    words_per_second = 2.3

    def chat(self, messages, *, model=None, task="", json_mode=True) -> CallResult:
        prompt = messages[-1]["content"] if isinstance(messages[-1]["content"], str) else ""
        if task == "extract":
            out = self._extract(prompt)
        elif task == "verdict":
            out = {"verdict": "Not covered", "source_id": None, "quote": None,
                   "reason": "Mock mode: no model was called."}
        elif task == "reply":
            out = self._reply(prompt)
        else:
            out = {}
        text = json.dumps(out, ensure_ascii=False)
        return CallResult(text, f"mock:{model or 'llm'}", len(prompt) // 4, len(text) // 4, 0.0)

    def vision(self, image, mime, prompt) -> CallResult:
        return CallResult("", "mock:vision")

    def transcribe(self, audio, filename, mime) -> CallResult:
        return CallResult("", "mock:stt")

    def tts(self, text) -> bytes:
        return make_silent_wav(len(text.split()) / self.words_per_second)

    @staticmethod
    def _extract(prompt: str) -> dict:
        pattern = re.escape(FORWARD_OPEN) + r" (\d+)[^\n]*\n(.*?)\n" + re.escape(FORWARD_CLOSE)
        claims, lang = [], "en"
        for idx, body in re.findall(pattern, prompt, flags=re.DOTALL):
            lang = detect_script_language(body) if lang == "en" else lang
            for sent in re.split(r"(?<=[.!?।])\s+|\n+", body):
                s = strip_links(sent).strip(" -*")
                if len(s.split()) < 4 or is_injection_like(s) or re.search(r"(?i)\b(share|forward)\b", s):
                    continue
                claims.append({"claim": s, "claim_en": s, "scope": "in_scope",
                               "excerpt": s[:200], "source_index": int(idx)})
        return {"input_language": lang, "claims": claims[:8]}

    @staticmethod
    def _reply(prompt: str) -> dict:
        m = re.search(re.escape(VERDICTS_OPEN) + r"\n(.*?)\n" + re.escape(VERDICTS_CLOSE), prompt, re.DOTALL)
        items = json.loads(m.group(1)) if m else []
        parts = ["Thank you for checking before forwarding."]
        for it in items:
            if it["verdict"] == "Contradicted":
                parts.append(f"About \"{it['claim']}\": {it['source_title']} ({it['source_date']}) says otherwise, "
                             "so please do not forward it.")
            elif it["verdict"] == "Supported":
                parts.append(f"About \"{it['claim']}\": {it['source_title']} ({it['source_date']}) confirms this.")
            else:
                parts.append(f"About \"{it['claim']}\": we could not confirm this from official sources, "
                             "so it is better not to forward it for now.")
        if any(it.get("category") == "health" for it in items):
            parts.append("For health questions, please consult a doctor.")
        return {"text": " ".join(parts) + " [mock reply]"}


def get_client(settings: Settings) -> NexusLike:
    return MockNexus() if settings.mock else NexusClient(settings)

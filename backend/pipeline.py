"""The ForwardCheck workflow: intake -> analyze -> review -> reply (DESIGN.md section 2)."""
from __future__ import annotations

import io
import re
import wave
from dataclasses import dataclass
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from . import prompts
from .config import Settings
from .db import Database
from .nexus import NexusError, NexusLike
from .schemas import (Case, Claim, Evidence, ExtractionOutput, InputSource, Reply, ReplyOutput, Review,
                      ReviewDecision, SourceOfClaim, VerdictOutput, now_iso)
from .source_pack import SourcePack
from .textutil import extract_links, parse_json_object, scan_injection, strip_links
from .verify import apply_model_verdict, clean_quote, verify_quote

T = TypeVar("T", bound=BaseModel)

IMAGE_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}
AUDIO_TYPES = {".mp3": "audio/mpeg", ".wav": "audio/wav", ".ogg": "audio/ogg", ".opus": "audio/ogg",
               ".m4a": "audio/mp4", ".aac": "audio/aac", ".webm": "audio/webm"}


class StateError(Exception):
    """Action not allowed in the case's current status."""


@dataclass
class Upload:
    filename: str
    data: bytes

    @property
    def ext(self) -> str:
        m = re.search(r"\.[A-Za-z0-9]+$", self.filename)
        return m.group(0).lower() if m else ""

    @property
    def kind(self) -> str:
        if self.ext in IMAGE_TYPES:
            return "image"
        if self.ext in AUDIO_TYPES:
            return "audio"
        raise ValueError(f"Unsupported file type '{self.ext or self.filename}'. "
                         f"Allowed: {', '.join(sorted(IMAGE_TYPES | AUDIO_TYPES))}")

    @property
    def mime(self) -> str:
        return {**IMAGE_TYPES, **AUDIO_TYPES}[self.ext]


def audio_seconds(data: bytes) -> float | None:
    try:
        with wave.open(io.BytesIO(data)) as w:
            return round(w.getnframes() / float(w.getframerate()), 2)
    except (wave.Error, EOFError):
        pass
    try:
        from mutagen import File as MutagenFile
        m = MutagenFile(io.BytesIO(data))
        if m is not None and m.info is not None:
            return round(float(m.info.length), 2)
    except Exception:
        pass
    return None


def _safe_name(name: str) -> str:
    base = re.split(r"[\\/]", name)[-1]
    return re.sub(r"[^A-Za-z0-9._-]", "_", base)[:100] or "upload"


class Pipeline:
    def __init__(self, settings: Settings, client: NexusLike, pack: SourcePack, db: Database):
        self.s, self.client, self.pack, self.db = settings, client, pack, db

    # ------------------------------------------------------------------ helpers

    def _save(self, case: Case) -> Case:
        self.db.save_case(case)
        return case

    def _llm_json(self, case_id: str, task: str, messages: list[dict], out_model: type[T],
                  model: str | None) -> T | None:
        """Call the LLM, validate against `out_model`; one retry on invalid output, else None."""
        msgs = list(messages)
        for _ in range(2):
            res = self.client.chat(msgs, model=model, task=task)
            try:
                parsed = out_model.model_validate(parse_json_object(res.text))
                self.db.log_call(case_id, task, res.model, res.input_tokens, res.output_tokens, res.seconds, True)
                return parsed
            except (ValueError, ValidationError) as e:
                self.db.log_call(case_id, task, res.model, res.input_tokens, res.output_tokens, res.seconds, False)
                msgs = msgs + [
                    {"role": "assistant", "content": res.text},
                    {"role": "user", "content": f"That output was not valid ({str(e)[:400]}). "
                                                "Reply with only one JSON object matching the required format."},
                ]
        self.db.audit(case_id, "system", f"{task}_invalid_json", {})
        return None

    # ------------------------------------------------------------------ 1-2 intake + convert

    def intake(self, uploads: list[Upload], pasted_text: str = "") -> Case:
        max_bytes = self.s.max_upload_mb * 1024 * 1024
        for up in uploads:
            _ = up.kind  # raises ValueError for unsupported types
            if len(up.data) > max_bytes:
                raise ValueError(f"{up.filename} is larger than {self.s.max_upload_mb} MB")
            if not up.data:
                raise ValueError(f"{up.filename} is empty")
        if not uploads and not pasted_text.strip():
            raise ValueError("Submit at least one image, voice note or pasted text")

        case_id = self.db.next_case_id()
        updir = self.s.data_dir / "uploads" / case_id
        updir.mkdir(parents=True, exist_ok=True)
        sources: list[InputSource] = []

        for up in uploads:
            name = _safe_name(up.filename)
            (updir / name).write_bytes(up.data)
            flags: list[str] = []
            text = ""
            try:
                if up.kind == "image":
                    res = self.client.vision(up.data, up.mime, prompts.VISION_PROMPT)
                else:
                    res = self.client.transcribe(up.data, name, up.mime)
                text = res.text.strip()
                self.db.log_call(case_id, up.kind, res.model, res.input_tokens, res.output_tokens, res.seconds, True)
            except NexusError as e:
                flags.append(f"conversion_error: {e}")
            if up.kind == "image" and len(text) < self.s.min_image_chars:
                flags.append("image_unclear: little or no text could be read. "
                             "Paste the text below or upload a clearer image.")
            if up.kind == "audio":
                flags.append("transcript_check: please check and correct the transcript before analysis.")
                if len(text.split()) < 3:
                    flags.append("transcript_short: the voice note is very short or unclear.")
            sources.append(InputSource(type=up.kind, file=name, extracted_text=text, flags=flags))

        if pasted_text.strip():
            sources.append(InputSource(type="text", file=None, extracted_text=pasted_text.strip()))

        usable = any(src.extracted_text for src in sources)
        case = Case(case_id=case_id, status="awaiting_text_confirmation" if usable else "needs_input",
                    input_sources=sources)
        self.db.audit(case_id, "system", "intake", {"sources": [s.type for s in sources]})
        return self._save(case)

    # ------------------------------------------------------------------ 3-6 extract, retrieve, verdict, verify

    def analyze(self, case: Case, corrected_texts: dict[int, str] | None = None,
                model_choice: str | None = None, actor: str = "volunteer") -> Case:
        if case.status not in ("awaiting_text_confirmation", "needs_input", "needs_manual_review", "awaiting_review"):
            raise StateError(f"Case is '{case.status}'; it can no longer be re-analysed")
        model = self.s.resolve_model(model_choice)

        for idx, txt in (corrected_texts or {}).items():
            if not 0 <= idx < len(case.input_sources):
                raise ValueError(f"No input source #{idx}")
            src = case.input_sources[idx]
            if txt.strip() != src.extracted_text:
                self.db.audit(case.case_id, actor, "text_corrected", {"source_index": idx})
                src.extracted_text = txt.strip()

        usable = [(i, s.type, s.extracted_text) for i, s in enumerate(case.input_sources) if s.extracted_text]
        if not usable:
            case.status = "needs_input"
            return self._save(case)

        full_text = "\n".join(t for _, _, t in usable)
        case.links_found = extract_links(full_text)
        case.injection_flags = scan_injection(full_text)
        case.claims, case.review, case.reply = [], Review(), Reply()

        out = self._llm_json(case.case_id, "extract", prompts.extraction_messages(usable), ExtractionOutput, model)
        if out is None:
            case.status = "needs_manual_review"
            return self._save(case)
        case.input_language = (out.input_language or "en").lower()[:5]

        valid_idx = {i for i, _, _ in usable}
        manual = False
        for n, ec in enumerate(out.claims[: self.s.max_claims], start=1):
            src_idx = ec.source_index if ec.source_index in valid_idx else usable[0][0]
            src = case.input_sources[src_idx]
            claim = Claim(
                claim_id=f"C{n}", claim=strip_links(ec.claim), claim_en=strip_links(ec.claim_en or ec.claim),
                scope=ec.scope,
                source_of_claim=SourceOfClaim(file=src.file or "pasted_text",
                                              excerpt=(ec.excerpt or ec.claim)[:200]),
            )
            if claim.scope == "out_of_scope":
                claim.flags.append("out_of_scope: political claim or judgement about an individual - not checked")
            else:
                passages = self.pack.search(claim.claim_en, k=3)
                claim.retrieved_ids = list(dict.fromkeys(p.source_id for p in passages))
                if not passages:
                    claim.flags.append("no_matching_sources: nothing in the source pack matched this claim")
                else:
                    v = self._llm_json(case.case_id, "verdict", prompts.verdict_messages(claim.claim_en, passages),
                                       VerdictOutput, model)
                    if v is None:
                        manual = True
                        claim.flags.append("verdict_invalid_output: model output was invalid twice; review manually")
                    else:
                        apply_model_verdict(claim, v, self.pack, set(claim.retrieved_ids))
            case.claims.append(claim)

        case.status = "needs_manual_review" if manual else "awaiting_review"
        self.db.audit(case.case_id, "system", "analyzed",
                      {"model": model, "claims": len(case.claims),
                       "verdicts": {c.claim_id: c.verdict for c in case.claims}})
        return self._save(case)

    # ------------------------------------------------------------------ 7 human review

    def review(self, case: Case, reviewer: str, decisions: list[ReviewDecision],
               reject_case: bool = False, note: str | None = None) -> Case:
        if case.status not in ("awaiting_review", "needs_manual_review"):
            raise StateError(f"Case is '{case.status}'; it is not waiting for review")
        reviewer = (reviewer or "").strip()
        if not reviewer:
            raise ValueError("Reviewer name is required")

        if reject_case:
            case.review = Review(status="rejected", reviewer=reviewer, note=note)
            case.status = "rejected"
            self.db.audit(case.case_id, reviewer, "case_rejected", {"note": note})
            return self._save(case)

        by_id = {c.claim_id: c for c in case.claims}
        given = [d.claim_id for d in decisions]
        if unknown := set(given) - set(by_id):
            raise ValueError(f"Unknown claim ids: {sorted(unknown)}")
        if missing := set(by_id) - set(given):
            raise ValueError(f"Every claim needs a decision; missing: {sorted(missing)}")
        if len(given) != len(set(given)):
            raise ValueError("Each claim can only have one decision")

        # Validate everything before changing anything.
        for d in decisions:
            if d.action != "edit":
                continue
            c = by_id[d.claim_id]
            if d.verdict is None:
                raise ValueError(f"{d.claim_id}: an edit needs a verdict")
            if c.scope == "out_of_scope" and d.verdict != "Not covered":
                raise ValueError(f"{d.claim_id}: out-of-scope claims are not checked and stay Not covered")
            if d.verdict != "Not covered":
                ok, reason = verify_quote(self.pack, d.source_id, d.quote)
                if not ok:
                    raise ValueError(f"{d.claim_id}: cannot mark {d.verdict} - {reason}")

        for d in decisions:
            c = by_id[d.claim_id]
            case.review.decisions[c.claim_id] = d.action
            if d.action == "edit":
                before = {"verdict": c.verdict, "evidence": c.evidence.model_dump() if c.evidence else None}
                c.verdict = d.verdict  # type: ignore[assignment]
                if d.verdict == "Not covered":
                    c.evidence = None
                else:
                    doc = self.pack.docs[d.source_id]  # type: ignore[index]
                    c.evidence = Evidence(source_id=doc.id, quote=clean_quote(d.quote), source_date=doc.date,
                                          quote_verified=True)
                c.flags.append(f"human_override by {reviewer}: {before['verdict']} -> {c.verdict}")
                edit = {"claim_id": c.claim_id, "time": now_iso(), "reviewer": reviewer, "before": before,
                        "after": {"verdict": c.verdict, "evidence": c.evidence.model_dump() if c.evidence else None}}
                case.review.edits.append(edit)
                self.db.audit(case.case_id, reviewer, "verdict_edited", edit)
            else:
                self.db.audit(case.case_id, reviewer, f"verdict_{d.action}d", {"claim_id": c.claim_id,
                                                                               "verdict": c.verdict})

        case.review.status, case.review.reviewer, case.review.note = "approved", reviewer, note
        case.status = "approved"
        return self._save(case)

    # ------------------------------------------------------------------ 8 reply + voice note

    def _reply_items(self, case: Case) -> list[dict]:
        items = []
        for c in case.claims:
            if case.review.decisions.get(c.claim_id) not in ("approve", "edit"):
                continue
            doc = self.pack.docs.get(c.evidence.source_id) if c.evidence else None
            items.append({
                "claim": c.claim, "verdict": c.verdict,
                "source_title": doc.title if doc else None, "source_date": doc.date if doc else None,
                "quote": c.evidence.quote if c.evidence else None,
                "category": doc.category if doc else ("health" if re.search(
                    r"(?i)\b(cure|disease|dengue|covid|vaccine|medicine|doctor|fever|cancer|virus)\b",
                    c.claim_en) else None),
                "out_of_scope": c.scope == "out_of_scope",
            })
        return items

    def reply(self, case: Case, model_choice: str | None = None, text_override: str | None = None,
              actor: str = "volunteer") -> Case:
        if case.status not in ("approved", "replied"):
            raise StateError("Every verdict must be reviewed and approved before a reply is generated")
        items = self._reply_items(case)
        if not items:
            raise ValueError("No approved verdicts to reply about")
        model = self.s.resolve_model(model_choice)
        flags: list[str] = []

        if text_override is not None:
            if not text_override.strip():
                raise ValueError("Reply text is empty")
            candidates = [(strip_links(text_override.strip()), None)]
            self.db.audit(case.case_id, actor, "reply_text_edited", {})
        else:
            candidates = [(None, self.s.reply_max_words), (None, self.s.reply_retry_words)]

        text, secs, wav = "", None, b""
        for i, (fixed, limit) in enumerate(candidates):
            if fixed is None:
                out = self._llm_json(case.case_id, "reply",
                                     prompts.reply_messages(items, case.input_language or "en", limit),
                                     ReplyOutput, model)
                if out is None:
                    raise NexusError("The model did not return a valid reply twice; write the reply manually")
                text = strip_links(out.text)
            else:
                text = fixed
            wav = self.client.tts(text)
            secs = audio_seconds(wav)
            if secs is not None and secs <= self.s.max_audio_seconds:
                break
            if i < len(candidates) - 1:
                self.db.audit(case.case_id, "system", "reply_too_long_retry", {"seconds": secs})

        if secs is None:
            flags.append("audio_duration_unknown: could not measure the voice note; please listen before sending")
        elif secs > self.s.max_audio_seconds:
            flags.append(f"audio_too_long: {secs:.1f}s is over {self.s.max_audio_seconds:.0f}s. "
                         "Shorten the reply text and regenerate the voice note.")

        audio_dir = self.s.data_dir / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        audio_name = f"{case.case_id}.wav"
        (audio_dir / audio_name).write_bytes(wav)

        case.reply.text, case.reply.audio_file, case.reply.audio_seconds, case.reply.flags = \
            text, audio_name, secs, flags
        case.status = "replied"
        self.db.audit(case.case_id, actor, "reply_generated",
                      {"words": len(text.split()), "audio_seconds": secs, "flags": flags})
        return self._save(case)

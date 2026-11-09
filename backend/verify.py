"""Code-side evidence verification. This is the safety core: no verified quote, no verdict."""
from __future__ import annotations

import re

from .schemas import Claim, Evidence, VerdictOutput
from .source_pack import SourcePack
from .textutil import normalize

MIN_QUOTE_WORDS = 4
_EDGE_CHARS = " \"'.,;:!?…-()[]"


def clean_quote(quote: str | None) -> str:
    return normalize(quote or "").strip(_EDGE_CHARS)


def verify_quote(pack: SourcePack, source_id: str | None, quote: str | None,
                 allowed_ids: set[str] | None = None) -> tuple[bool, str]:
    """True only if `quote` appears word for word (after normalisation) in source `source_id`."""
    if not source_id:
        return False, "no source was cited"
    doc = pack.docs.get(source_id)
    if doc is None:
        return False, "the cited source is not in the source pack"
    if allowed_ids is not None and source_id not in allowed_ids:
        return False, "the cited source was not among the retrieved passages"
    q = clean_quote(quote)
    if not q:
        return False, "no quote was given"
    if len(q.split()) < MIN_QUOTE_WORDS:
        return False, f"the quote is shorter than {MIN_QUOTE_WORDS} words"
    pattern = r"(?<!\w)" + re.escape(q.casefold()) + r"(?!\w)"
    if not re.search(pattern, doc.body_norm.casefold()):
        return False, "the quote does not appear word for word in the source"
    return True, "verified"


def apply_model_verdict(claim: Claim, v: VerdictOutput, pack: SourcePack, allowed_ids: set[str]) -> None:
    """Set claim verdict from model output, forcing Not covered when evidence fails verification."""
    if v.verdict == "Not covered":
        claim.verdict, claim.evidence = "Not covered", None
        return
    ok, reason = verify_quote(pack, v.source_id, v.quote, allowed_ids)
    if ok:
        doc = pack.docs[v.source_id]  # type: ignore[index]
        claim.verdict = v.verdict
        claim.evidence = Evidence(source_id=doc.id, quote=clean_quote(v.quote), source_date=doc.date,
                                  quote_verified=True)
    else:
        claim.verdict, claim.evidence = "Not covered", None
        claim.flags.append(
            f"evidence_rejected: model said {v.verdict} citing {v.source_id!r}, but {reason}. Forced to Not covered."
        )

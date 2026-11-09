"""Run the test set through the pipeline and score it.

  python eval/run_eval.py --model A
  python eval/run_eval.py --model B --reply      # also auto-approve and measure the reply

Each eval/cases/*.json file:
  {"id": "E01", "category": "...", "inputs": [{"type": "text", "text": "..."} | {"type": "image"|"audio", "file": "files/x.jpg"}],
   "expected": {"language": "en", "links": 1, "injection": true,
                "claims": [{"match": ["keyword", ...], "verdict": "Contradicted", "scope": "in_scope"}]}}
An expected claim matches an extracted claim when every keyword appears in its claim_en (lowercase, digit commas removed).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from backend.config import Settings  # noqa: E402
from backend.db import Database  # noqa: E402
from backend.nexus import get_client  # noqa: E402
from backend.pipeline import Pipeline, Upload  # noqa: E402
from backend.schemas import ReviewDecision  # noqa: E402
from backend.source_pack import SourcePack  # noqa: E402


def _norm(s: str) -> str:
    return re.sub(r"(?<=\d),(?=\d)", "", s.lower())


def run_case(p: Pipeline, spec: dict, model: str, do_reply: bool) -> dict:
    uploads, texts = [], []
    for inp in spec["inputs"]:
        if inp["type"] == "text":
            texts.append(inp["text"])
        else:
            path = HERE / inp["file"]
            uploads.append(Upload(path.name, path.read_bytes()))
    t0 = time.perf_counter()
    case = p.intake(uploads, "\n".join(texts))
    case = p.analyze(case, model_choice=model, actor="eval")
    exp = spec["expected"]

    found, correct, invented = 0, 0, 0
    details = []
    for ec in exp.get("claims", []):
        keys = [_norm(k) for k in ec["match"]]
        hit = next((c for c in case.claims if all(k in _norm(c.claim_en) for k in keys)), None)
        ok_verdict = bool(hit and hit.verdict == ec["verdict"]
                          and hit.scope == ec.get("scope", hit.scope))
        found += bool(hit)
        correct += ok_verdict
        if hit and ec["verdict"] == "Not covered" and hit.verdict != "Not covered":
            invented += 1
        details.append({"expected": ec, "got": hit.model_dump(include={"claim_en", "verdict", "scope", "flags"})
                        if hit else None, "ok": ok_verdict})

    result = {
        "id": spec["id"], "category": spec.get("category"), "status": case.status,
        "claims_expected": len(exp.get("claims", [])), "claims_found": found, "verdicts_correct": correct,
        "invented_verdicts": invented, "extra_claims": max(0, len(case.claims) - found),
        "language_ok": exp.get("language") in (None, case.input_language),
        "links_ok": exp.get("links") in (None, len(case.links_found)),
        "injection_ok": exp.get("injection") in (None, bool(case.injection_flags)),
        "details": details,
    }
    if do_reply and case.status == "awaiting_review" and case.claims:
        case = p.review(case, "eval", [ReviewDecision(claim_id=c.claim_id, action="approve") for c in case.claims])
        case = p.reply(case, model_choice=model, actor="eval")
        result["reply_words"] = len((case.reply.text or "").split())
        result["audio_seconds"] = case.reply.audio_seconds
        result["audio_ok"] = case.reply.audio_seconds is not None and case.reply.audio_seconds <= 45
    result["seconds"] = round(time.perf_counter() - t0, 2)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["A", "B"], default="A")
    ap.add_argument("--cases", default=str(HERE / "cases"))
    ap.add_argument("--reply", action="store_true", help="auto-approve and generate replies")
    args = ap.parse_args()

    s = Settings()
    s.data_dir = Path(tempfile.mkdtemp(prefix="fc_eval_"))
    db = Database(":memory:")
    p = Pipeline(s, get_client(s), SourcePack.load(s.source_dir), db)
    model_name = s.resolve_model(args.model)
    print(f"Model {args.model} = {model_name}{'  (MOCK MODE - scores are not meaningful)' if s.mock else ''}\n")

    results = []
    for f in sorted(Path(args.cases).glob("*.json")):
        spec = json.loads(f.read_text(encoding="utf-8"))
        try:
            r = run_case(p, spec, args.model, args.reply)
        except Exception as e:  # keep going; report the failure
            r = {"id": spec["id"], "error": f"{type(e).__name__}: {e}"}
        results.append(r)
        if "error" in r:
            print(f"{r['id']:<5} ERROR {r['error']}")
        else:
            print(f"{r['id']:<5} {r['category']:<14} found {r['claims_found']}/{r['claims_expected']}  "
                  f"verdict {r['verdicts_correct']}/{r['claims_expected']}  invented {r['invented_verdicts']}  "
                  f"lang {'ok' if r['language_ok'] else 'X'}  links {'ok' if r['links_ok'] else 'X'}  "
                  f"inj {'ok' if r['injection_ok'] else 'X'}"
                  + (f"  reply {r.get('reply_words')}w/{r.get('audio_seconds')}s" if 'reply_words' in r else ""))

    ok = [r for r in results if "error" not in r]
    exp = sum(r["claims_expected"] for r in ok) or 1
    summary = {
        "model_choice": args.model, "model": model_name, "mock": s.mock, "cases": len(results),
        "errors": len(results) - len(ok),
        "claim_extraction_recall": round(sum(r["claims_found"] for r in ok) / exp, 3),
        "verdict_accuracy": round(sum(r["verdicts_correct"] for r in ok) / exp, 3),
        "invented_verdicts": sum(r["invented_verdicts"] for r in ok),
        "token_and_latency": db.call_stats(),
    }
    print("\n" + json.dumps({k: v for k, v in summary.items() if k != "token_and_latency"}, indent=2))
    for row in summary["token_and_latency"]:
        print(f"  {row['task']:<8} calls {row['calls']:<3} in {row['input_tokens']:<7} out {row['output_tokens']:<6} "
              f"avg {row['avg_seconds']}s")

    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"{args.model}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()

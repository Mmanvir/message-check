"""Volunteer dashboard. Talks only to the ForwardCheck backend; no API keys here.

Run:  streamlit run frontend/app.py
"""
from __future__ import annotations

import os

import httpx
import streamlit as st

BACKEND = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")
VERDICT_COLOURS = {"Supported": "green", "Contradicted": "red", "Not covered": "gray"}

st.set_page_config(page_title="ForwardCheck", layout="wide")


def api(method: str, path: str, **kw):
    try:
        r = httpx.request(method, BACKEND + path, timeout=180, **kw)
    except httpx.HTTPError as e:
        st.error(f"Cannot reach the backend at {BACKEND} ({type(e).__name__}). Is it running?")
        st.stop()
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail")
        except ValueError:
            detail = r.text
        st.error(f"{r.status_code}: {detail}")
        return None
    return r


def load_case(case_id: str) -> dict | None:
    r = api("GET", f"/cases/{case_id}")
    return r.json() if r else None


# ---------------- sidebar ----------------
with st.sidebar:
    st.title("ForwardCheck")
    health = api("GET", "/health")
    if health:
        h = health.json()
        if h["mock"]:
            st.warning("MOCK MODE: no Nexus key set. Results are placeholders.")
        st.caption(f"{h['sources']} source documents · Model A: {h['models']['A']} · Model B: {h['models']['B']}")
    reviewer = st.text_input("Your name (reviewer)", key="reviewer")
    model = st.radio("LLM", ["A", "B"], horizontal=True, key="model")
    st.divider()
    if st.button("➕ New case", width="stretch"):
        st.session_state.pop("case_id", None)
        st.rerun()
    st.subheader("Recent cases")
    recent = api("GET", "/cases", params={"limit": 20})
    for row in (recent.json() if recent else []):
        if st.button(f"{row['case_id']} · {row['status']}", key=f"open_{row['case_id']}", width="stretch"):
            st.session_state.case_id = row["case_id"]
            st.rerun()


# ---------------- intake ----------------
case_id = st.session_state.get("case_id")
if not case_id:
    st.header("New forward")
    with st.form("intake", clear_on_submit=True):
        files = st.file_uploader("Screenshot(s) and/or voice note(s)", accept_multiple_files=True,
                                 type=["jpg", "jpeg", "png", "webp", "mp3", "wav", "ogg", "opus", "m4a", "aac", "webm"])
        text = st.text_area("Or paste the forwarded text", height=150)
        if st.form_submit_button("Submit", type="primary"):
            payload = [("files", (f.name, f.getvalue(), f.type or "application/octet-stream")) for f in files or []]
            r = api("POST", "/cases", files=payload or None, data={"text": text})
            if r:
                st.session_state.case_id = r.json()["case_id"]
                st.rerun()
    st.stop()

case = load_case(case_id)
if case is None:
    st.stop()

st.header(f"{case['case_id']}")
st.caption(f"Status: **{case['status']}** · created {case['created_at']}"
           + (f" · language: {case['input_language']}" if case.get("input_language") else ""))

# ---------------- step 1: confirm extracted text ----------------
editable = case["status"] in ("awaiting_text_confirmation", "needs_input", "needs_manual_review", "awaiting_review")
with st.expander("1 · Input text (check OCR / transcript)", expanded=case["status"] in
                 ("awaiting_text_confirmation", "needs_input")):
    corrected = {}
    for i, src in enumerate(case["input_sources"]):
        st.markdown(f"**#{i} · {src['type']}**" + (f" · `{src['file']}`" if src.get("file") else ""))
        for f in src["flags"]:
            st.warning(f)
        corrected[i] = st.text_area("Text", src["extracted_text"], key=f"src_{case_id}_{i}",
                                    disabled=not editable, label_visibility="collapsed", height=120)
    if editable:
        label = "Re-analyse" if case["claims"] else "Confirm text and analyse"
        if st.button(label, type="primary"):
            with st.spinner("Extracting claims and checking the source pack…"):
                r = api("POST", f"/cases/{case_id}/analyze",
                        json={"corrected_texts": {str(k): v for k, v in corrected.items()}, "model": model,
                              "actor": reviewer or "volunteer"})
            if r:
                st.rerun()

if case["status"] == "needs_manual_review" and not case["claims"]:
    st.error("The model's output was invalid twice. Correct the text and re-analyse, or reject the case.")

# ---------------- step 2: verdict review ----------------
if case["claims"] or case["status"] in ("awaiting_review", "needs_manual_review"):
    st.subheader("2 · Verdicts")
    if case["links_found"]:
        st.markdown("**Links found (never opened):**")
        st.code("\n".join(case["links_found"]), language=None)
    for f in case["injection_flags"]:
        st.error(f"⚠ {f}")

    sources = {s["id"]: s for s in (api("GET", "/sources").json() or [])}
    reviewing = case["status"] in ("awaiting_review", "needs_manual_review")
    decisions = []
    for c in case["claims"]:
        with st.container(border=True):
            colour = VERDICT_COLOURS[c["verdict"]]
            st.markdown(f"**{c['claim_id']}** · :{colour}[**{c['verdict']}**]"
                        + (" · *out of scope*" if c["scope"] == "out_of_scope" else ""))
            st.markdown(f"> {c['claim']}")
            if c["claim_en"] != c["claim"]:
                st.caption(f"English: {c['claim_en']}")
            st.caption(f"From `{c['source_of_claim']['file']}`: “{c['source_of_claim']['excerpt']}”")
            if c["evidence"]:
                ev = c["evidence"]
                title = sources.get(ev["source_id"], {}).get("title", ev["source_id"])
                st.success(f"“{ev['quote']}”\n\n— {title} ({ev['source_id']}, {ev['source_date']})"
                           + ("  ✔ quote verified" if ev["quote_verified"] else ""))
            for f in c["flags"]:
                st.info(f)
            if reviewing:
                action = st.radio("Decision", ["approve", "edit", "reject"], horizontal=True,
                                  key=f"act_{case_id}_{c['claim_id']}")
                d = {"claim_id": c["claim_id"], "action": action}
                if action == "edit":
                    d["verdict"] = st.selectbox("New verdict", ["Not covered", "Supported", "Contradicted"],
                                                key=f"v_{case_id}_{c['claim_id']}")
                    if d["verdict"] != "Not covered":
                        d["source_id"] = st.selectbox("Source", list(sources),
                                                      format_func=lambda s: f"{s} · {sources[s]['title']}",
                                                      key=f"s_{case_id}_{c['claim_id']}")
                        d["quote"] = st.text_area("Exact quote from that source", key=f"q_{case_id}_{c['claim_id']}")
                        with st.expander("Show source text"):
                            src = api("GET", f"/sources/{d['source_id']}")
                            if src:
                                st.text(src.json()["body"])
                decisions.append(d)
            else:
                st.caption(f"Volunteer decision: {case['review']['decisions'].get(c['claim_id'], '-')}")

    if reviewing:
        note = st.text_input("Review note (optional)")
        col1, col2 = st.columns(2)
        if col1.button("Submit review", type="primary", disabled=not reviewer or not case["claims"]):
            if api("POST", f"/cases/{case_id}/review",
                   json={"reviewer": reviewer, "decisions": decisions, "note": note or None}):
                st.rerun()
        if col2.button("Reject whole case", disabled=not reviewer):
            if api("POST", f"/cases/{case_id}/review",
                   json={"reviewer": reviewer, "reject_case": True, "note": note or None}):
                st.rerun()
        if not reviewer:
            st.caption("Enter your name in the sidebar to submit.")

# ---------------- step 3: reply ----------------
if case["status"] in ("approved", "replied"):
    st.subheader("3 · Shareable reply")
    if not case["reply"]["text"]:
        if st.button("Generate reply + voice note", type="primary"):
            with st.spinner("Writing reply and recording voice note…"):
                if api("POST", f"/cases/{case_id}/reply", json={"model": model, "actor": reviewer or "volunteer"}):
                    st.rerun()
    else:
        for f in case["reply"]["flags"]:
            st.warning(f)
        new_text = st.text_area("Reply text (edit and regenerate the voice note if needed)",
                                case["reply"]["text"], height=150, key=f"reply_{case_id}")
        st.caption(f"{len(new_text.split())} words · voice note {case['reply']['audio_seconds']} s (limit 45 s)")
        audio = api("GET", f"/cases/{case_id}/audio")
        if audio:
            st.audio(audio.content, format="audio/wav")
        if st.button("Regenerate voice note from this text"):
            if api("POST", f"/cases/{case_id}/reply",
                   json={"text_override": new_text, "actor": reviewer or "volunteer"}):
                st.rerun()

if case["status"] == "rejected":
    st.error(f"Case rejected by {case['review']['reviewer']}. {case['review'].get('note') or ''}")

with st.expander("Audit log"):
    log = api("GET", f"/cases/{case_id}/audit")
    if log:
        st.dataframe(log.json(), width="stretch")

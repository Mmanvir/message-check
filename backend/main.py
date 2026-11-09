"""FastAPI backend. All Nexus calls happen here; the browser never sees the key.

Run:  uvicorn backend.main:app --reload
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from .config import Settings
from .db import Database
from .nexus import NexusError, get_client
from .pipeline import Pipeline, StateError, Upload
from .schemas import Case, ReviewDecision
from .source_pack import SourcePack


@lru_cache(maxsize=1)
def get_pipeline() -> Pipeline:
    s = Settings()
    s.data_dir.mkdir(parents=True, exist_ok=True)
    return Pipeline(s, get_client(s), SourcePack.load(s.source_dir), Database(s.db_path))


app = FastAPI(title="ForwardCheck API", version="0.1.0")


@app.exception_handler(ValueError)
async def _value_error(_: Request, e: ValueError):
    return JSONResponse(status_code=400, content={"detail": str(e)})


@app.exception_handler(StateError)
async def _state_error(_: Request, e: StateError):
    return JSONResponse(status_code=409, content={"detail": str(e)})


@app.exception_handler(NexusError)
async def _nexus_error(_: Request, e: NexusError):
    return JSONResponse(status_code=502, content={"detail": str(e)})


def _case(case_id: str) -> Case:
    case = get_pipeline().db.get_case(case_id)
    if case is None:
        raise HTTPException(404, f"Case {case_id} not found")
    return case


class AnalyzeRequest(BaseModel):
    corrected_texts: dict[int, str] = Field(default_factory=dict)
    model: Literal["A", "B"] | None = None
    actor: str = "volunteer"


class ReviewRequest(BaseModel):
    reviewer: str
    decisions: list[ReviewDecision] = Field(default_factory=list)
    reject_case: bool = False
    note: str | None = None


class ReplyRequest(BaseModel):
    model: Literal["A", "B"] | None = None
    text_override: str | None = None
    actor: str = "volunteer"


@app.get("/health")
def health():
    p = get_pipeline()
    return {"status": "ok", "mock": p.s.mock, "sources": len(p.pack.docs),
            "models": {"A": p.s.llm_model, "B": p.s.llm_model_b}}


@app.get("/sources")
def list_sources():
    return [{"id": d.id, "title": d.title, "date": d.date, "category": d.category, "url": d.url}
            for d in get_pipeline().pack.docs.values()]


@app.get("/sources/{source_id}")
def get_source(source_id: str):
    d = get_pipeline().pack.docs.get(source_id)
    if d is None:
        raise HTTPException(404, "Source not found")
    return {"id": d.id, "title": d.title, "date": d.date, "category": d.category, "url": d.url,
            "licence": d.licence, "body": d.body}


@app.post("/cases", response_model=Case)
async def create_case(files: list[UploadFile] = File(default=[]), text: str = Form(default="")):
    uploads = [Upload(f.filename or "upload", await f.read()) for f in files if f.filename]
    return get_pipeline().intake(uploads, text)


@app.get("/cases")
def list_cases(limit: int = 50):
    return get_pipeline().db.list_cases(limit)


@app.get("/cases/{case_id}", response_model=Case)
def get_case(case_id: str):
    return _case(case_id)


@app.get("/cases/{case_id}/audit")
def get_audit(case_id: str):
    _case(case_id)
    return get_pipeline().db.get_audit(case_id)


@app.post("/cases/{case_id}/analyze", response_model=Case)
def analyze(case_id: str, req: AnalyzeRequest):
    return get_pipeline().analyze(_case(case_id), req.corrected_texts, req.model, req.actor)


@app.post("/cases/{case_id}/review", response_model=Case)
def review(case_id: str, req: ReviewRequest):
    return get_pipeline().review(_case(case_id), req.reviewer, req.decisions, req.reject_case, req.note)


@app.post("/cases/{case_id}/reply", response_model=Case)
def reply(case_id: str, req: ReplyRequest):
    return get_pipeline().reply(_case(case_id), req.model, req.text_override, req.actor)


@app.get("/cases/{case_id}/audio")
def get_audio(case_id: str):
    case = _case(case_id)
    if not case.reply.audio_file:
        raise HTTPException(404, "No voice note yet")
    path = get_pipeline().s.data_dir / "audio" / case.reply.audio_file
    if not path.exists():
        raise HTTPException(404, "Voice note file missing")
    return FileResponse(path, media_type="audio/wav", filename=case.reply.audio_file)


@app.get("/metrics")
def metrics():
    return get_pipeline().db.call_stats()

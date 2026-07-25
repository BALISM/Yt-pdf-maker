"""
Phase 0/2/6 — FastAPI app and routes.

Endpoints:
  POST /summarize          Kick off a job for a YouTube URL. Returns a job_id
                            immediately (Phase 6: async background processing).
  GET  /status/{job_id}    Poll job status/progress.
  GET  /download/{job_id}  Download the finished PDF once status == "done".
  GET  /                   Serve the simple demo frontend (Phase 7).

Phase 2 originally asked for a synchronous POST /summarize that returns JSON
directly - that version is kept too, at POST /summarize/sync, since it's a
genuinely useful "does the core loop work" smoke-test endpoint for short
videos, separate from the production async flow.
"""
from __future__ import annotations

import logging

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.jobs import create_job, get_job, run_pipeline
from app.models import JobStatus, SummarizeRequest, SummarizeResponse, VideoSummary
from app.summarizer import SummarizationError, summarize_single
from app.transcript import (
    InvalidYouTubeURL,
    NoTranscriptAvailable,
    TranscriptError,
    VideoTooLong,
    get_transcript,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="YouTube -> PDF Summary Generator",
    description="Paste a YouTube URL, get back a structured PDF summary.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # demo project - lock this down for real deployments
    allow_methods=["*"],
    allow_headers=["*"],
)


def _map_transcript_error(e: Exception) -> HTTPException:
    if isinstance(e, InvalidYouTubeURL):
        return HTTPException(status_code=400, detail=f"Invalid YouTube URL: {e}")
    if isinstance(e, VideoTooLong):
        return HTTPException(status_code=413, detail=str(e))
    if isinstance(e, NoTranscriptAvailable):
        return HTTPException(status_code=422, detail=str(e))
    if isinstance(e, TranscriptError):
        return HTTPException(status_code=502, detail=str(e))
    return HTTPException(status_code=500, detail=f"Unexpected error: {e}")


# ---------------------------------------------------------------------------
# Phase 2 — synchronous smoke-test endpoint (short videos only)
# ---------------------------------------------------------------------------

@app.post("/summarize/sync", response_model=VideoSummary, tags=["demo"])
def summarize_sync(payload: SummarizeRequest) -> VideoSummary:
    """
    Synchronous version: blocks until done and returns the structured JSON
    summary directly (no PDF). Only sensible for short videos - long ones
    will just make the HTTP request hang until the whole pipeline finishes.
    Kept around as the Phase 2 checkpoint endpoint.
    """
    try:
        transcript = get_transcript(payload.url)
        return summarize_single(transcript.full_text, video_title=None)
    except TranscriptError as e:
        raise _map_transcript_error(e) from e
    except SummarizationError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e


# ---------------------------------------------------------------------------
# Phase 6 — async job-based endpoints (production path)
# ---------------------------------------------------------------------------

@app.post("/summarize", response_model=SummarizeResponse, tags=["pipeline"])
def summarize(payload: SummarizeRequest, background_tasks: BackgroundTasks) -> SummarizeResponse:
    """Validate the URL synchronously (fast, cheap) so obviously-bad input
    fails immediately instead of silently going into a background job, then
    hand the real work off to a BackgroundTask."""
    try:
        from app.transcript import extract_video_id

        extract_video_id(payload.url)
    except InvalidYouTubeURL as e:
        raise HTTPException(status_code=400, detail=f"Invalid YouTube URL: {e}") from e

    job = create_job(payload.url, category=payload.category or "auto")
    background_tasks.add_task(run_pipeline, job.job_id)
    return SummarizeResponse(job_id=job.job_id, status=job.status)


@app.get("/status/{job_id}", tags=["pipeline"])
def status(job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job_id")
    return job


@app.get("/download/{job_id}", tags=["pipeline"])
def download(job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job_id")
    if job.status != JobStatus.DONE or not job.pdf_filename:
        raise HTTPException(status_code=409, detail=f"Job is not finished yet (status={job.status})")

    pdf_path = settings.output_path / job.pdf_filename
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="PDF file not found on disk")

    download_name = f"{job.video_title or 'summary'}.pdf".replace("/", "-")
    return FileResponse(pdf_path, media_type="application/pdf", filename=download_name)


@app.get("/health", tags=["meta"])
def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Phase 7 — serve the simple demo frontend
# ---------------------------------------------------------------------------

app.mount("/", StaticFiles(directory="static", html=True), name="static")

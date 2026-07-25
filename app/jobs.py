"""
Phase 6 — background job handling.

Keeps an in-memory job store (a dict is enough for a single-process demo
app — see the README for how this would map onto Celery+Redis if you
needed multiple worker processes or durability across restarts) and runs
the full pipeline end to end for a job:

  fetch transcript -> decide if chunking is needed -> chunk if so ->
  summarize (single-call or map-reduce) -> render PDF -> mark done

Every stage updates the job's status so /status/{job_id} always reflects
what's currently happening, and every stage is wrapped so a failure anywhere
lands the job in a clean ERROR state with a human-readable message instead
of crashing the background task silently.
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import Dict

from app.chunking import chunk_transcript, needs_chunking
from app.config import settings
from app.models import JobRecord, JobStatus
from app.pdf_generator import generate_pdf
from app.summarizer import SummarizationError, summarize_long_transcript, summarize_single
from app.transcript import (
    InvalidYouTubeURL,
    NoTranscriptAvailable,
    TranscriptError,
    VideoTooLong,
    get_transcript,
)

logger = logging.getLogger(__name__)

# Simple in-memory store. Fine for a single-process demo; swap for
# Redis/a database if you need this to survive restarts or run across
# multiple worker processes.
_JOBS: Dict[str, JobRecord] = {}


def create_job(url: str, category: str = "auto") -> JobRecord:
    job_id = uuid.uuid4().hex[:12]
    now = time.time()
    job = JobRecord(job_id=job_id, url=url, category=category, created_at=now, updated_at=now)
    _JOBS[job_id] = job
    return job


def get_job(job_id: str) -> JobRecord | None:
    return _JOBS.get(job_id)


def _update(job: JobRecord, **fields) -> None:
    for k, v in fields.items():
        setattr(job, k, v)
    job.updated_at = time.time()
    _JOBS[job.job_id] = job


def run_pipeline(job_id: str) -> None:
    """The full Phase 1-5 pipeline for one job. Meant to be called from a
    FastAPI BackgroundTask, so it takes only the job_id (BackgroundTasks
    run after the response is sent, in the same process)."""
    job = _JOBS.get(job_id)
    if job is None:
        logger.error("run_pipeline called for unknown job_id=%s", job_id)
        return

    try:
        # --- Phase 1: transcript -----------------------------------------
        _update(job, status=JobStatus.FETCHING_TRANSCRIPT, progress_message="Fetching transcript...")
        transcript = get_transcript(job.url)
        _update(
            job,
            video_id=transcript.video_id,
            used_whisper_fallback=(transcript.source == "whisper"),
        )

        full_text = transcript.full_text

        # --- Phase 3: chunk if needed -------------------------------------
        if needs_chunking(full_text):
            _update(job, status=JobStatus.CHUNKING, progress_message="Splitting long transcript into chunks...")
            chunks = chunk_transcript(transcript.segments, strategy="chapter_aware", chapters=transcript.chapters)
            _update(job, num_chunks=len(chunks))


            # --- Phase 4: map-reduce summarization -------------------------
            _update(
                job,
                status=JobStatus.SUMMARIZING_CHUNKS,
                progress_message=f"Summarizing chunk 0/{len(chunks)}...",
            )

            def on_chunk_done(done: int, total: int) -> None:
                _update(job, progress_message=f"Summarizing chunk {done}/{total}...")

            summary = summarize_long_transcript(
                chunks, video_title=job.video_title, category=job.category, on_chunk_done=on_chunk_done
            )
            _update(job, status=JobStatus.SYNTHESIZING, progress_message="Synthesizing final document...")
        else:
            # --- Phase 2: single-call summarization ---------------------
            _update(job, status=JobStatus.SUMMARIZING_CHUNKS, progress_message="Summarizing transcript...")
            summary = summarize_single(full_text, video_title=job.video_title, category=job.category)

        _update(job, video_title=summary.title, summary=summary)

        # --- Phase 5: render PDF -------------------------------------------
        _update(job, status=JobStatus.GENERATING_PDF, progress_message="Rendering PDF report...")
        pdf_filename = f"{job.job_id}.pdf"
        pdf_path = settings.output_path / pdf_filename
        generate_pdf(summary, pdf_path, source_url=job.url)

        _update(
            job,
            status=JobStatus.DONE,
            progress_message="Done",
            pdf_filename=pdf_filename,
            summary=summary,
        )
        logger.info("Job %s completed successfully -> %s", job.job_id, pdf_filename)

    except InvalidYouTubeURL as e:
        _update(job, status=JobStatus.ERROR, progress_message="Invalid URL", error=str(e))
    except VideoTooLong as e:
        _update(job, status=JobStatus.ERROR, progress_message="Video too long", error=str(e))
    except NoTranscriptAvailable as e:
        _update(job, status=JobStatus.ERROR, progress_message="No transcript available", error=str(e))
    except TranscriptError as e:
        _update(job, status=JobStatus.ERROR, progress_message="Transcript error", error=str(e))
    except SummarizationError as e:
        _update(job, status=JobStatus.ERROR, progress_message="Summarization failed", error=str(e))
    except Exception as e:  # noqa: BLE001 - last-resort catch so the job never hangs silently
        logger.exception("Unexpected error in job %s", job_id)
        _update(job, status=JobStatus.ERROR, progress_message="Unexpected error", error=str(e))

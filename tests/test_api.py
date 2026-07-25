"""
API-level tests using FastAPI's TestClient. These test routing, request
validation, and job lifecycle plumbing WITHOUT needing real network access
to YouTube or Gemini - anything that requires those is monkeypatched.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
from app.models import JobStatus, TranscriptResult, TranscriptSegment, VideoSummary  # noqa: E402

client = TestClient(app)


def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_summarize_rejects_invalid_url():
    res = client.post("/summarize", json={"url": "https://example.com/not-youtube"})
    assert res.status_code == 400
    assert "Invalid YouTube URL" in res.json()["detail"]


def test_status_unknown_job_404():
    res = client.get("/status/does-not-exist")
    assert res.status_code == 404


def test_download_unknown_job_404():
    res = client.get("/download/does-not-exist")
    assert res.status_code == 404


def test_full_pipeline_with_mocked_transcript_and_llm(monkeypatch):
    """
    Exercises the whole /summarize -> background job -> /status -> /download
    flow with transcript fetching and LLM calls monkeypatched out, so it
    runs fully offline and deterministically.
    """
    fake_transcript = TranscriptResult(
        video_id="dQw4w9WgXcQ",
        language_code="en",
        is_generated=False,
        source="captions",
        segments=[
            TranscriptSegment(text="hello and welcome to this video", start=0.0, duration=2.0),
            TranscriptSegment(text="today we're covering an important topic", start=2.0, duration=2.5),
        ],
    )
    fake_summary = VideoSummary(
        title="Mocked Summary Title",
        overview="A mocked overview for testing purposes.",
        key_takeaways=["First takeaway", "Second takeaway"],
        sections=[],
    )

    monkeypatch.setattr("app.jobs.get_transcript", lambda url: fake_transcript)
    monkeypatch.setattr("app.jobs.needs_chunking", lambda text: False)
    monkeypatch.setattr("app.jobs.summarize_single", lambda text, video_title=None: fake_summary)

    res = client.post("/summarize", json={"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"})
    assert res.status_code == 200
    job_id = res.json()["job_id"]
    assert res.json()["status"] == JobStatus.PENDING.value

    # TestClient runs BackgroundTasks synchronously before returning control
    # here in most versions, but poll briefly just in case.
    deadline = time.time() + 5
    job = None
    while time.time() < deadline:
        status_res = client.get(f"/status/{job_id}")
        assert status_res.status_code == 200
        job = status_res.json()
        if job["status"] in (JobStatus.DONE.value, JobStatus.ERROR.value):
            break
        time.sleep(0.1)

    assert job is not None
    assert job["status"] == JobStatus.DONE.value, job
    assert job["video_title"] == "Mocked Summary Title"

    download_res = client.get(f"/download/{job_id}")
    assert download_res.status_code == 200
    assert download_res.headers["content-type"] == "application/pdf"
    assert len(download_res.content) > 500


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

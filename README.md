# YouTube → PDF Summary Generator

Paste a YouTube URL, get back a structured, well-formatted PDF summary of the
video. Built in the 7 phases below, each one runnable/testable on its own
before the next was wired in.

```
POST /summarize {"url": "..."}  ->  {"job_id": "..."}
GET  /status/{job_id}           ->  progress + status
GET  /download/{job_id}         ->  the finished PDF
```

A simple browser demo is served at `/` (paste a link, watch it progress, download the PDF).

---

## Project structure

```
app/
  config.py         Settings, loaded from .env
  models.py         All Pydantic schemas (transcript, chunks, LLM output, jobs)
  transcript.py      Phase 1 — URL parsing, captions, Whisper fallback
  chunking.py        Phase 3 — naive + timestamp-gap chunking strategies
  summarizer.py       Phase 2/4 — single-call + map-reduce Gemini calls
  pdf_generator.py    Phase 5 — structured JSON -> PDF (reportlab)
  jobs.py             Phase 6 — in-memory job store + pipeline orchestration
  main.py             FastAPI routes
static/index.html      Phase 7 — demo frontend
scripts/test_transcript.py   Phase 1 checkpoint script (no FastAPI, no LLM)
tests/                  Offline pytest suite (chunking, PDF, API routing)
```

---

## Setup

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and set GEMINI_API_KEY (get one at https://aistudio.google.com/apikey)

uvicorn app.main:app --reload
# open http://127.0.0.1:8000
```

For the test suite you'll also need `pip install -r requirements-dev.txt`.

---

## ⚠️ How this was actually tested (read this)

I built this in a sandboxed environment whose network access is locked to
package registries (PyPI, npm, GitHub) — it **cannot reach `youtube.com` or
Google's Gemini API**. So the testing split into two categories:

**Verified by me, with a real automated test suite (19 tests, all passing,
including a from-scratch clean-venv install of the exact pinned
`requirements.txt`):**
- URL parsing for every common YouTube link shape (`scripts/test_transcript.py --parse-only`)
- Both chunking strategies (`tests/test_chunking.py`) — contiguity, full
  coverage, target-size adherence, gap-preference, and the hard-cap safety
  valve, all checked against synthetic transcripts
- PDF rendering (`tests/test_pdf_generator.py`) — file creation, the
  TOC/no-TOC branch, all content actually appearing in the extracted text,
  special-character escaping, empty-input edge cases. I also rendered real
  PDFs and visually inspected them.
- The FastAPI app itself (`tests/test_api.py`) — routing, validation, and a
  full mocked pipeline run (`/summarize` → background job → `/status` →
  `/download`), plus a live `uvicorn` boot hitting real HTTP routes
- The `google-genai` SDK's config objects construct correctly against the
  installed version (verified without a real API key/network call)
- **A real end-to-end run against an actual `youtube.com` URL** — this
  didn't get a transcript back (network's blocked, as expected), but it
  proved something more useful: the pipeline correctly tried captions
  first, fell back to the Whisper/yt-dlp path automatically when that
  failed, and landed the job in a clean `error` state with a readable
  message instead of hanging or crashing.

**NOT verified by me — needs to be checked on your machine with your own
API key and real network access:**
- Actually fetching a real transcript from `youtube_transcript_api`
- Actually calling the Gemini API and getting back structured JSON
- The Whisper fallback actually transcribing real audio (faster-whisper
  imports fine and the model download logic is straightforward, but I
  never ran it against real audio)
- Whether YouTube blocks your IP outright (very common on cloud hosts —
  see the note in `transcript.py` and the "Known limitations" section below)

Run `python scripts/test_transcript.py` on your own machine first — it's
built for exactly this: a quick "does the core loop even work" check before
you rely on anything downstream.

---

## Phase-by-phase notes

### Phase 1 — Transcript retrieval
`extract_video_id()` handles `youtube.com/watch?v=`, `youtu.be/`,
`/embed/`, `/shorts/`, `/live/`, `m.youtube.com`, extra query params, and
bare video IDs. Captions come from `youtube-transcript-api` (v1.2+, which
uses the newer instance-based `YouTubeTranscriptApi().fetch()` API, not the
older static-method one you'll see in a lot of tutorials online). It prefers
manually-created transcripts over auto-generated ones, then falls back to
whatever's available in any language.

If no captions exist at all, it downloads audio with `yt-dlp` and
transcribes locally with `faster-whisper` (CPU by default — set
`WHISPER_DEVICE=cuda` in `.env` if you have a GPU). Both paths converge on
the same `TranscriptResult` shape so nothing downstream needs to know which
one ran.

**Known real-world gotcha:** YouTube blocks a lot of cloud-provider IP
ranges (AWS, GCP, Azure) from `youtube_transcript_api` and from `yt-dlp`.
If you deploy this and start seeing `RequestBlocked`/`IpBlocked` errors,
that's not a bug — see the residential-proxy note in `transcript.py`
(`youtube_transcript_api.proxies.WebshareProxyConfig`) or expect to run
this locally / from a residential IP for now.

### Phase 2 — One-shot summarization
`summarize_single()` sends the whole transcript in one Gemini call with a
Pydantic `response_schema` (`VideoSummary`), so you get back parsed,
validated structured data — not markdown you then have to parse yourself.
Exposed directly at `POST /summarize/sync` as a quick smoke-test endpoint
(blocks until done, returns JSON, no PDF — good for testing the model/prompt
without waiting on the full pipeline).

### Phase 3 — Chunking
Two strategies, both in `chunking.py`:
- `naive_chunk_by_words` — fixed word-count chunks. Simple, will cut mid-sentence.
- `chunk_by_timestamp_gaps` (**the default**) — same target size, but looks
  for a real pause between caption segments (configurable via
  `CHUNK_GAP_SECONDS`) near the target and cuts there instead, which tends
  to land on actual topic boundaries. A hard cap
  (`hard_max_multiplier`) guarantees it still splits even for a video with
  zero silence.

**Worth knowing:** current Gemini Flash models have ~1M-token context
windows, so in raw "does it fit" terms almost no single video's transcript
actually *requires* chunking anymore. `chunking.py` has a longer comment on
why it's still worth doing — mainly output quality (a focused per-segment
summary beats asking one call to hold a 2-hour transcript in its head) and
because it's the map half of the Phase 4 map-reduce pipeline.

### Phase 4 — Synthesis across chunks
Map step: `summarize_chunk()` runs once per chunk, producing a heading +
bullets for just that segment (explicitly told not to invent a conclusion
it hasn't earned yet). Reduce step: `synthesize_summaries()` gets all the
chunk summaries as one JSON blob and is prompted specifically to merge
duplicate points across chunks, resolve references that only make sense
with earlier context, and even out uneven granularity — producing one
coherent `VideoSummary`, not a concatenation.

### Phase 5 — Output structure
`VideoSummary` (title, overview, key takeaways, sections with
heading/timestamp/bullets/detail) is the single structured contract between
the LLM layer and the PDF layer — the LLM never touches markdown or layout
concerns. `pdf_generator.py` renders it with `reportlab`'s Platypus API:
title page, overview, key takeaways, an auto-generated table of contents
(only for documents with 5+ sections — short ones skip it), then one
subsection per section. A two-pass `multiBuild` is used so TOC page numbers
are accurate.

### Phase 6 — Async job API
`POST /summarize` validates the URL synchronously (fast/cheap — fails
immediately on garbage input) then hands the real work to a FastAPI
`BackgroundTasks` call. Jobs live in an in-memory dict (`jobs.py`) with a
status enum that mirrors the actual pipeline stages
(`fetching_transcript` → `chunking` → `summarizing_chunks` → `synthesizing`
→ `generating_pdf` → `done`/`error`), so `/status/{job_id}` always reflects
what's really happening, not just "processing."

This is intentionally a single-process, in-memory job store — good enough
for a demo/portfolio project or light personal use. If you needed this to
survive restarts or scale across multiple worker processes, the natural
next step is Celery + Redis: swap `BackgroundTasks.add_task(run_pipeline,
job.job_id)` for `run_pipeline.delay(job.job_id)`, back `_JOBS` with Redis
instead of a dict, and run workers with `celery -A app.jobs worker`. The
pipeline function itself (`run_pipeline`) wouldn't need to change much —
it's already a single function that takes a job_id and does its own status
updates.

### Phase 7 — Polish
- `static/index.html` — plain HTML/JS demo (paste URL → poll status → download link), no build step.
- Error handling: invalid URLs (400), no transcript available at all (422),
  video too long (413, configurable via `MAX_VIDEO_DURATION_SECONDS`),
  transcript-layer failures (502), unexpected errors (500) — all mapped to
  clean HTTP responses in `main.py`'s `_map_transcript_error`, and every
  stage of the background pipeline is wrapped so a failure anywhere lands
  the job in a readable `error` state instead of hanging silently.
- **Not implemented, left as a clear next step:** clickable timestamp links
  from summary points back to the exact moment in the video. The
  groundwork is there — every `Section` already carries a `timestamp`
  string and every chunk knows its `start_time`/`end_time` in seconds — it
  just isn't rendered as a link in the PDF yet. The natural way to add it:
  turn `Section.timestamp` into a real `youtube.com/watch?v={id}&t={seconds}s`
  URL and render it as a reportlab hyperlink (`<link href="...">`) instead
  of plain text in `pdf_generator.py`.

---

## Configuration reference (`.env`)

| Variable | Default | Notes |
|---|---|---|
| `GEMINI_API_KEY` | *(required)* | from https://aistudio.google.com/apikey |
| `GEMINI_MODEL` | `gemini-2.5-flash` | any current Gemini model works; see note below |
| `CHUNK_TARGET_TOKENS` | `3000` | soft target chunk size |
| `CHUNK_GAP_SECONDS` | `2.5` | pause length treated as a likely topic break |
| `MAX_VIDEO_DURATION_SECONDS` | `14400` (4h) | reject anything longer |
| `WHISPER_MODEL_SIZE` | `base` | tiny/base/small/medium/large-v3 |
| `WHISPER_DEVICE` | `cpu` | `cuda` if you have a GPU |

**On the Gemini model choice:** as of mid-2026 Google's Flash-tier lineup
(3, 3.1, 3.5, 3.6) all ship with ~1M-token context windows, so basically
any of them works fine here. I defaulted to `gemini-2.5-flash` as the
safest, longest-established option, but `gemini-3.5-flash` or the
newer `gemini-3.6-flash` (released July 21, 2026) are worth trying —
just change `GEMINI_MODEL` in `.env`. Check
[ai.google.dev/gemini-api/docs/models](https://ai.google.dev/gemini-api/docs/models)
for the current list, since this moves fast.

---

## Running the tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

19 tests, all offline (no network/API key needed) — chunking logic, PDF
rendering, and API routing with the transcript/LLM layers mocked out.

For the parts that genuinely need network + a real API key:

```bash
# Phase 1 checkpoint - no FastAPI, no LLM, just transcript retrieval
python scripts/test_transcript.py                    # a few default videos
python scripts/test_transcript.py "https://youtu.be/..." "..."  # your own

# Phase 2 checkpoint - one real Gemini call, short video
curl -X POST http://127.0.0.1:8000/summarize/sync \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.youtube.com/watch?v=jNQXAC9IVRw"}'

# Full pipeline
curl -X POST http://127.0.0.1:8000/summarize \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.youtube.com/watch?v=..."}'
# then poll /status/{job_id}, then GET /download/{job_id}
```

---

## Known limitations / honest gaps

- Job store is in-memory — restarting the server loses in-flight/completed
  job records (the PDF files themselves stay on disk in `outputs/` though).
- No auth/rate-limiting — fine for local/personal use, add something before
  exposing this publicly.
- Whisper fallback hasn't been tested against real audio by me (see testing
  section above) — the happy path should work, but budget time for
  debugging `yt-dlp` format selection quirks on real videos.
- No semantic (embedding-based) chunking — the README for Phase 3 explains
  why timestamp-gap chunking was the practical choice, but embedding-based
  topic detection is the natural "harder" upgrade if you want it for
  portfolio value.
- Timestamp deep-links into the PDF aren't wired up yet (see Phase 7 notes above).

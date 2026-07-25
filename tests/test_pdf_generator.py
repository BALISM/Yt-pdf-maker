"""
Offline tests for pdf_generator.py. No network, no LLM calls - renders
synthetic VideoSummary objects and checks the resulting PDF's extracted
text, which is a reasonable proxy for "did this actually render right"
without needing a human to look at every page.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402
from pypdf import PdfReader  # noqa: E402

from app.models import Section, VideoSummary  # noqa: E402
from app.pdf_generator import TOC_MIN_SECTIONS, generate_pdf  # noqa: E402


def _extract_all_text(pdf_path) -> str:
    reader = PdfReader(str(pdf_path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def make_summary(num_sections: int) -> VideoSummary:
    sections = [
        Section(
            heading=f"Section {i + 1} heading",
            timestamp=f"{i}:00",
            bullets=[f"Bullet point {i + 1}.A", f"Bullet point {i + 1}.B"],
            detail=f"Detail sentence for section {i + 1}." if i % 2 == 0 else None,
        )
        for i in range(num_sections)
    ]
    return VideoSummary(
        title="Test Video Title With Some Detail",
        overview="This is a synthetic overview paragraph used purely for testing PDF rendering.",
        key_takeaways=["Takeaway one", "Takeaway two", "Takeaway three"],
        sections=sections,
    )


def test_pdf_file_is_created(tmp_path):
    summary = make_summary(3)
    out = generate_pdf(summary, tmp_path / "out.pdf", source_url="https://youtu.be/abc123")
    assert out.exists()
    assert out.stat().st_size > 500  # not an empty/corrupt file


def test_short_summary_skips_toc(tmp_path):
    assert TOC_MIN_SECTIONS >= 2
    summary = make_summary(TOC_MIN_SECTIONS - 1)
    out = generate_pdf(summary, tmp_path / "short.pdf")
    text = _extract_all_text(out)
    assert "Contents" not in text


def test_long_summary_includes_toc(tmp_path):
    summary = make_summary(TOC_MIN_SECTIONS + 2)
    out = generate_pdf(summary, tmp_path / "long.pdf")
    text = _extract_all_text(out)
    assert "Contents" in text


def test_all_content_appears_in_output(tmp_path):
    summary = make_summary(4)
    out = generate_pdf(summary, tmp_path / "content.pdf", source_url="https://youtu.be/xyz789")
    text = _extract_all_text(out)

    assert summary.title.replace("\n", "") in text.replace("\n", "")
    assert "youtu.be/xyz789" in text
    for takeaway in summary.key_takeaways:
        assert takeaway in text
    for section in summary.sections:
        assert section.heading in text
        for bullet in section.bullets:
            assert bullet in text


def test_special_characters_are_escaped_not_dropped(tmp_path):
    summary = VideoSummary(
        title="A & B: <Testing> Special Characters",
        overview="Contains an ampersand & angle brackets <like this> which reportlab treats as markup.",
        key_takeaways=["100% > 50%, always", "Q&A session covered edge cases"],
        sections=[
            Section(heading="Edge cases: <script> & such", bullets=["x < y & y < z"]),
        ],
    )
    out = generate_pdf(summary, tmp_path / "special.pdf")
    text = _extract_all_text(out)
    # The literal characters should survive in the rendered text layer,
    # proving they were escaped (not interpreted as XML markup and eaten).
    assert "&" in text
    assert "100%" in text


def test_empty_optional_fields_dont_crash(tmp_path):
    summary = VideoSummary(
        title="Minimal Summary",
        overview="Just an overview, no takeaways, no sections.",
        key_takeaways=[],
        sections=[],
    )
    out = generate_pdf(summary, tmp_path / "minimal.pdf")
    assert out.exists()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

"""
Phase 5 — Structuring the output.

Takes a VideoSummary (structured JSON — see models.py) and renders it into a
clean PDF: title page, overview, key takeaways, a table of contents (for
documents with enough sections to be worth one), then one subsection per
section with headings, bullets, and optional timestamp + detail text.

We deliberately render from structured JSON rather than asking the LLM for
markdown — it gives us full, predictable control over layout (bullet
styling, spacing, TOC, page breaks) instead of hoping a markdown-to-PDF
conversion handles the model's formatting choices gracefully.
"""
from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    ListFlowable,
    ListItem,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
)
from reportlab.platypus.tableofcontents import TableOfContents

from app.models import VideoSummary

# Sections beyond this count get a table of contents; short documents don't
# need one and it just adds a page.
TOC_MIN_SECTIONS = 5


def _build_styles():
    styles = getSampleStyleSheet()

    styles.add(
        ParagraphStyle(
            "DocTitle",
            parent=styles["Title"],
            fontSize=24,
            leading=29,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            "SourceLine",
            parent=styles["Normal"],
            fontSize=9,
            textColor=colors.HexColor("#666666"),
            spaceAfter=18,
        )
    )
    styles.add(
        ParagraphStyle(
            "OverviewHeading",
            parent=styles["Heading2"],
            spaceBefore=6,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            "Overview",
            parent=styles["Normal"],
            fontSize=11,
            leading=15,
            spaceAfter=16,
        )
    )
    styles.add(
        ParagraphStyle(
            "Takeaway",
            parent=styles["Normal"],
            fontSize=11,
            leading=15,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            "SectionHeading",
            parent=styles["Heading2"],
            spaceBefore=18,
            spaceAfter=4,
            textColor=colors.HexColor("#1a1a1a"),
        )
    )
    styles.add(
        ParagraphStyle(
            "Timestamp",
            parent=styles["Normal"],
            fontSize=9,
            textColor=colors.HexColor("#4a6fa5"),
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            "BulletText",
            parent=styles["Normal"],
            fontSize=10.5,
            leading=14,
            spaceAfter=3,
        )
    )
    styles.add(
        ParagraphStyle(
            "Detail",
            parent=styles["Normal"],
            fontSize=10.5,
            leading=14,
            textColor=colors.HexColor("#333333"),
            spaceBefore=4,
            spaceAfter=4,
            leftIndent=0,
        )
    )
    styles.add(
        ParagraphStyle(
            "TOCHeading",
            parent=styles["Heading1"],
            spaceAfter=12,
        )
    )
    return styles


def _toc_entry_style(styles):
    from reportlab.lib.styles import ParagraphStyle

    return ParagraphStyle(
        "TOCEntry",
        parent=styles["Normal"],
        fontSize=11,
        leading=16,
    )


def _escape(text: str) -> str:
    """reportlab Paragraphs interpret a small XML-like markup language, so
    raw ampersands/angle brackets from the transcript or LLM output need
    escaping or they'll break rendering (or silently vanish)."""
    if not text:
        return ""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def generate_pdf(
    summary: VideoSummary,
    output_path: Path | str,
    source_url: str | None = None,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    styles = _build_styles()
    story = []

    include_toc = len(summary.sections) >= TOC_MIN_SECTIONS

    # --- Title block -------------------------------------------------
    story.append(Paragraph(_escape(summary.title), styles["DocTitle"]))
    if source_url:
        story.append(Paragraph(_escape(source_url), styles["SourceLine"]))
    else:
        story.append(Spacer(1, 12))

    # --- Overview ------------------------------------------------------
    story.append(Paragraph("Overview", styles["OverviewHeading"]))
    story.append(Paragraph(_escape(summary.overview), styles["Overview"]))

    # --- Key takeaways ---------------------------------------------------
    if summary.key_takeaways:
        story.append(Paragraph("Key Takeaways", styles["OverviewHeading"]))
        items = [
            ListItem(Paragraph(_escape(point), styles["Takeaway"]), leftIndent=14)
            for point in summary.key_takeaways
        ]
        story.append(
            ListFlowable(items, bulletType="bullet", start="circle", leftIndent=10)
        )
        story.append(Spacer(1, 10))

    # --- Table of contents (long documents only) ------------------------
    toc = None
    if include_toc:
        story.append(PageBreak())
        story.append(Paragraph("Contents", styles["TOCHeading"]))
        toc = TableOfContents()
        toc.levelStyles = [_toc_entry_style(styles)]
        story.append(toc)

    # --- Sections --------------------------------------------------------
    story.append(PageBreak())
    for i, section in enumerate(summary.sections):
        heading_text = _escape(section.heading)
        # Bookmark + TOC notification: reportlab's TableOfContents listens
        # for a special paragraph "outline" event, which is triggered by
        # giving the Paragraph a `bookmarkName`... but the standard/simple
        # approach is a custom flowable-level afterFlowable hook on the doc
        # template (see _NumberedCanvas/DocTemplate below via afterFlowable).
        para = Paragraph(f'<a name="section-{i}"/>{heading_text}', styles["SectionHeading"])
        para._toc_text = heading_text  # consumed by the doc template's afterFlowable
        para._toc_level = 0
        story.append(para)

        if section.timestamp:
            story.append(Paragraph(f"~ {_escape(section.timestamp)}", styles["Timestamp"]))

        if section.bullets:
            items = [
                ListItem(Paragraph(_escape(b), styles["BulletText"]), leftIndent=14)
                for b in section.bullets
            ]
            story.append(
                ListFlowable(items, bulletType="bullet", start="disc", leftIndent=10)
            )

        if section.detail:
            story.append(Paragraph(_escape(section.detail), styles["Detail"]))

        story.append(Spacer(1, 6))

    doc = _TOCDocTemplate(str(output_path), pagesize=LETTER,
                           leftMargin=0.85 * inch, rightMargin=0.85 * inch,
                           topMargin=0.85 * inch, bottomMargin=0.85 * inch,
                           title=summary.title)

    if include_toc:
        # Two passes are required for an accurate TOC (reportlab quirk):
        # first pass collects heading positions, second pass renders page
        # numbers correctly. multiBuild handles this automatically.
        doc.multiBuild(story)
    else:
        doc.build(story)

    return output_path


class _TOCDocTemplate(BaseDocTemplate):
    """A BaseDocTemplate that feeds section headings into the TableOfContents
    automatically, based on the `_toc_text`/`_toc_level` attributes we stash
    on heading Paragraphs above."""

    def __init__(self, filename, **kwargs):
        super().__init__(filename, **kwargs)
        frame = Frame(
            self.leftMargin, self.bottomMargin,
            self.width, self.height,
            id="normal",
        )
        self.addPageTemplates([PageTemplate(id="main", frames=frame)])

    def afterFlowable(self, flowable):
        if hasattr(flowable, "_toc_text"):
            self.notify(
                "TOCEntry",
                (flowable._toc_level, flowable._toc_text, self.page),
            )

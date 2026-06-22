"""
Document parser — converts PDF / HTML / TXT files into structured text sections
while preserving source metadata (page numbers, section headers).

Per spec §3 Stage 1: "Don't just dump text. Preserve structure."
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# Headers tend to be short lines; we use this heuristic when a doc has no
# explicit heading markup (plain text / some PDFs).
_HEADER_HEURISTIC = re.compile(r"^(#{1,6}\s+|[A-Z0-9][A-Za-z0-9 \-:&]{3,60})$")


@dataclass
class Section:
    """A logical section of a document (header + body text + page)."""

    text: str
    header: str = ""
    page_num: Optional[int] = None
    source_doc: str = ""

    @property
    def full_text(self) -> str:
        return f"{self.header}\n{self.text}".strip() if self.header else self.text


@dataclass
class ParsedDocument:
    """The result of parsing one file."""

    source_doc: str
    sections: List[Section] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        return "\n\n".join(s.full_text for s in self.sections)


def parse_file(path: str | Path) -> ParsedDocument:
    """Parse a file by extension. Raises ValueError on unsupported types."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    suffix = path.suffix.lower()
    source = path.name

    if suffix == ".pdf":
        return _parse_pdf(path, source)
    if suffix in (".html", ".htm"):
        return _parse_html(path, source)
    if suffix in (".txt", ".md"):
        return _parse_text(path, source)
    raise ValueError(f"Unsupported file type: {suffix} ({path})")


# ---------------------------------------------------------------------------
# PDF — pymupdf (preserves reading order)
# ---------------------------------------------------------------------------
def _parse_pdf(path: Path, source: str) -> ParsedDocument:
    import fitz  # pymupdf

    doc = fitz.open(path)
    sections: List[Section] = []
    try:
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            if not text:
                continue
            # Detect headers: first non-empty line that looks like a heading.
            lines = text.split("\n")
            header = lines[0].strip() if lines and len(lines[0].strip()) < 80 else ""
            sections.append(Section(text=text, header=header, page_num=page_num, source_doc=source))
    finally:
        doc.close()

    logger.info("Parsed PDF %s: %d pages with text", source, len(sections))
    return ParsedDocument(source_doc=source, sections=sections)


# ---------------------------------------------------------------------------
# HTML — beautifulsoup (extracts headers from h1-h6)
# ---------------------------------------------------------------------------
def _parse_html(path: Path, source: str) -> ParsedDocument:
    from bs4 import BeautifulSoup

    html = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "lxml")

    # Strip non-content noise.
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    sections: List[Section] = []
    # Walk headings; each heading starts a section.
    headings = soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])
    if headings:
        for head in headings:
            header_text = head.get_text(strip=True)
            # Collect text of siblings until the next heading.
            body_parts: List[str] = []
            for sib in head.find_all_next(["p", "li", "table"]):
                if sib.name in ("h1", "h2", "h3", "h4", "h5", "h6"):
                    break
                body_parts.append(sib.get_text(" ", strip=True))
            body = "\n".join(p for p in body_parts if p)
            if body:
                sections.append(Section(text=body, header=header_text, source_doc=source))
    else:
        # No headings: treat whole body as one section.
        body_text = soup.get_text("\n", strip=True)
        if body_text:
            sections.append(Section(text=body_text, source_doc=source))

    logger.info("Parsed HTML %s: %d sections", source, len(sections))
    return ParsedDocument(source_doc=source, sections=sections)


# ---------------------------------------------------------------------------
# Plain text / markdown
# ---------------------------------------------------------------------------
def _parse_text(path: Path, source: str) -> ParsedDocument:
    text = path.read_text(encoding="utf-8", errors="ignore")
    sections: List[Section] = []
    current_header = ""
    current_lines: List[str] = []

    def flush() -> None:
        body = "\n".join(current_lines).strip()
        if body:
            sections.append(
                Section(text=body, header=current_header, source_doc=source)
            )
        current_lines.clear()

    for line in text.splitlines():
        stripped = line.strip()
        # Markdown-style headers.
        if stripped.startswith("#"):
            flush()
            current_header = stripped.lstrip("#").strip()
            continue
        # Heuristic header (short, title-case line).
        if _HEADER_HEURISTIC.match(stripped) and len(current_lines) == 0 and stripped:
            current_header = stripped
            continue
        current_lines.append(line)
    flush()

    if not sections:
        sections.append(Section(text=text, source_doc=source))

    logger.info("Parsed text %s: %d sections", source, len(sections))
    return ParsedDocument(source_doc=source, sections=sections)

#!/usr/bin/env python3
"""
PDF Paper Parser

Extracts structured content from a paper PDF using PyMuPDF:
  - Full text organized by detected sections
  - All figures/tables rendered as high-resolution images
  - Figure-to-caption binding and in-text reference tracking

Usage:
    python pdf_parser.py <pdf_path> [--output-dir <dir>] [--dpi 200]

Output: JSON to stdout with the following structure:
    {
        "success": true,
        "pages": 12,
        "sections": [{"heading": "...", "level": 1, "text": "...", "page": 1}, ...],
        "figures": [{"id": "fig1", "image_path": "...", "caption": "...", "page": 3, "references": ["..."]}, ...],
        "tables": [...],
        "full_text": "...",
        "metadata": {"title": "...", "author": "..."}
    }
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional


try:
    import fitz  # PyMuPDF
except ImportError:
    print(json.dumps({
        "success": False,
        "error": "PyMuPDF (fitz) is required. Install with: pip install PyMuPDF"
    }, ensure_ascii=False, indent=2))
    sys.exit(1)


# ── Figure & Table detection patterns ──────────────────────────────────────

FIGURE_CAPTION_PATTERN = re.compile(
    r"^(Fig(?:ure)?\.?\s*\d+|Figure\s+\d+)", re.IGNORECASE
)

TABLE_CAPTION_PATTERN = re.compile(
    r"^Table\.?\s*\d+", re.IGNORECASE
)

SECTION_PATTERNS = [
    # Numbered sections: "1. Introduction", "3.1.2 Method Details"
    (re.compile(r"^(\d+(?:\.\d+)*)\s+(.+)$"), 1),
    # Unnumbered common headings
    (re.compile(r"^(Abstract|Introduction|Related Work|Method|Experiments?"
                 r"|Results?|Discussion|Conclusion|References?|Appendix"
                 r"|Acknowledgments?|Supplementary\s+Material)$", re.IGNORECASE), 1),
    # Chinese section headings
    (re.compile(r"^(摘要|引言|相关工作|方法|实验|结果|讨论|结论|参考文献|附录)$"), 1),
]

INLINE_REF_PATTERN = re.compile(
    r"(?:Figure|Fig\.?)\s*(\d+)", re.IGNORECASE
)


def _is_bold(flags: int) -> bool:
    """Check if a text span has bold font flags (bit 4 in fitz font flags)."""
    return bool(flags & 2 ** 3)


def _font_size(spans: list) -> float:
    """Return the maximum font size from a list of text spans."""
    sizes = [s.get("size", 0) for s in spans]
    return max(sizes) if sizes else 0


def _clean_text(text: str) -> str:
    """Collapse whitespace and remove PDF artifacts."""
    return re.sub(r"\s+", " ", text).strip()


def _extract_spans(block: dict) -> list:
    """
    Extract text spans from a PyMuPDF block dict, handling both
    block→spans and block→lines→spans structures.
    """
    spans = list(block.get("spans", []))
    if spans:
        return spans
    for line in block.get("lines", []):
        spans.extend(line.get("spans", []))
    return spans


# ── Section detection ──────────────────────────────────────────────────────

def _detect_sections(blocks: list, global_font_sizes: list) -> list:
    """
    Partition text blocks into sections based on font size and heading patterns.

    Returns a list of dicts: {"heading": str, "level": int, "text": str, "page": int}
    """
    if not global_font_sizes:
        return []

    body_size = max(set(global_font_sizes), key=global_font_sizes.count)
    threshold = body_size * 1.15  # 15% larger than body = heading candidate

    sections = []
    current_section = {"heading": "Abstract", "level": 1, "text": "", "page": 0}
    heading_stack = [{"size": 999, "level": 0}]

    for block in blocks:
        spans = _extract_spans(block)
        if not spans:
            continue

        text = _clean_text(" ".join(s.get("text", "") for s in spans))
        if not text or len(text) < 2:
            continue

        size = _font_size(spans)
        page = block.get("number", 0)
        is_bold = any(_is_bold(s.get("flags", 0)) for s in spans)

        # Heuristic: heading if font significantly larger than body, or bold + slightly larger
        is_heading_candidate = (
            size >= threshold or (is_bold and size > body_size)
        )

        # Check against section patterns
        matched_pattern = False
        if is_heading_candidate:
            for pattern, level in SECTION_PATTERNS:
                m = pattern.match(text)
                if m:
                    matched_pattern = True
                    # Pop headings of equal or deeper level
                    while heading_stack and heading_stack[-1]["level"] >= level:
                        heading_stack.pop()

                    if current_section["text"].strip() or current_section["heading"] != "Abstract":
                        sections.append(dict(current_section))

                    current_section = {"heading": text, "level": level, "text": "", "page": page}
                    heading_stack.append({"size": size, "level": level})
                    break

        if not matched_pattern:
            current_section["text"] += text + "\n"
            if current_section["page"] == 0:
                current_section["page"] = page

    # Don't forget the last section
    if current_section["text"].strip():
        sections.append(dict(current_section))

    return sections


# ── Figure extraction ──────────────────────────────────────────────────────

def _extract_figures(doc: fitz.Document, output_dir: str, dpi: int = 200) -> list:
    """
    Extract figures from the PDF.

    Strategy:
    1. Render each page as a high-res image
    2. Find figure/table caption text blocks (with their bounding boxes)
    3. Crop the region above each caption as the figure image
    4. Also extract embedded raster images from the page

    Returns a list of figure dicts.
    """
    figures = []
    fig_counter = 0

    for page_num in range(len(doc)):
        page = doc[page_num]
        text_blocks = page.get_text("dict").get("blocks", [])

        # Collect all caption blocks
        caption_blocks = []
        for block in text_blocks:
            if block.get("type") != 0:  # text block
                continue
            spans = _extract_spans(block)
            text = _clean_text(" ".join(s.get("text", "") for s in spans))
            if FIGURE_CAPTION_PATTERN.match(text):
                caption_blocks.append({"text": text, "bbox": block["bbox"], "page": page_num})

        if not caption_blocks:
            continue

        # Render the page for figure cropping
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        page_img_path = os.path.join(output_dir, f"page_{page_num + 1}.png")
        pix.save(page_img_path)

        for cap in caption_blocks:
            fig_counter += 1
            fig_id = f"fig_{fig_counter}"

            # Calculate crop region: the area above the caption
            x0, y0, x1, y1 = cap["bbox"]
            scale = dpi / 72

            # Figure typically sits between top of page (or previous caption) and this caption
            # We crop from top-of-page to the caption top, with margins
            figure_bbox = fitz.Rect(
                0,
                0,
                page.rect.width,
                max(0, y0 - 5)  # 5pt margin above caption
            )

            # If there's a previous caption on the same page, adjust lower bound
            for other in caption_blocks:
                if other is cap:
                    continue
                oy = other["bbox"][3]  # bottom of other caption
                if oy < y0 and oy > figure_bbox.y0:
                    figure_bbox.y0 = oy + 10

            # Crop and save figure image
            fig_pix = page.get_pixmap(
                matrix=mat,
                clip=figure_bbox,
            )
            fig_img_path = os.path.join(output_dir, f"{fig_id}.png")
            fig_pix.save(fig_img_path)

            # Extract figure number from caption
            fig_num_match = FIGURE_CAPTION_PATTERN.search(cap["text"])
            fig_number = fig_num_match.group(0) if fig_num_match else str(fig_counter)

            figures.append({
                "id": fig_id,
                "number": fig_number,
                "image_path": fig_img_path,
                "page_image_path": page_img_path,
                "caption": cap["text"],
                "page": page_num + 1,
                "references": [],  # populated later by _bind_references
            })

    return figures


def _bind_references(figures: list, full_text: str) -> list:
    """
    Find in-text references to each figure (e.g., "as shown in Figure 3").
    Enriches each figure dict with a 'references' list of citing sentences.
    """
    # Build a map: normalized figure number → list of figures
    fig_map = {}
    for fig in figures:
        num = fig["number"].lower().replace(".", "").replace("figure", "").replace("fig", "").strip()
        if num not in fig_map:
            fig_map[num] = []
        fig_map[num].append(fig)

    # Split text into sentences and search for references
    sentences = re.split(r"(?<=[.!?])\s+", full_text)

    for sentence in sentences:
        for match in INLINE_REF_PATTERN.finditer(sentence):
            ref_num = match.group(1)
            if ref_num in fig_map:
                for fig in fig_map[ref_num]:
                    clean_sentence = _clean_text(sentence)
                    if clean_sentence not in fig["references"]:
                        fig["references"].append(clean_sentence)

    return figures


# ── Table extraction ───────────────────────────────────────────────────────

def _extract_tables(doc: fitz.Document) -> list:
    """
    Extract table captions and their associated content.
    For actual cell-level extraction, tools like camelot or tabula would be better;
    here we capture table regions and captions for identification.
    """
    tables = []
    table_counter = 0

    for page_num in range(len(doc)):
        page = doc[page_num]
        # PyMuPDF can detect tables in some PDFs
        found_tables = page.find_tables()
        if found_tables:
            for tab in found_tables:
                table_counter += 1
                tables.append({
                    "id": f"table_{table_counter}",
                    "page": page_num + 1,
                    "rows": len(tab.row_count) if hasattr(tab, "row_count") else "unknown",
                    "data": tab.extract() if hasattr(tab, "extract") else [],
                })

    return tables


# ── Metadata extraction ────────────────────────────────────────────────────

def _extract_metadata(doc: fitz.Document) -> dict:
    """Extract PDF metadata if available."""
    meta = doc.metadata or {}
    return {
        "title": meta.get("title", ""),
        "author": meta.get("author", ""),
        "subject": meta.get("subject", ""),
        "creation_date": meta.get("creationDate", ""),
    }


# ── Main ────────────────────────────────────────────────────────────────────

def parse_pdf(pdf_path: str, output_dir: str, dpi: int = 200) -> dict:
    """
    Parse a paper PDF and return structured content.

    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory for extracted images and artifacts
        dpi: Rendering resolution for figure extraction (higher = better quality)

    Returns:
        A dict with sections, figures, tables, and metadata.
    """
    if not os.path.exists(pdf_path):
        return {"success": False, "error": f"PDF not found: {pdf_path}"}

    os.makedirs(output_dir, exist_ok=True)

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        return {"success": False, "error": f"Failed to open PDF: {e}"}

    if doc.page_count == 0:
        return {"success": False, "error": "PDF has no pages"}

    # ── Build full text and blocks ──
    all_blocks = []
    full_text_parts = []
    font_sizes = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        full_text_parts.append(text)

        blocks = page.get_text("dict").get("blocks", [])
        for block in blocks:
            if block.get("type") == 0:
                block["number"] = page_num
                all_blocks.append(block)
                for span in _extract_spans(block):
                    font_sizes.append(span.get("size", 0))

    full_text = "\n".join(full_text_parts)

    # ── Detect sections ──
    sections = _detect_sections(all_blocks, font_sizes)

    # ── Extract figures ──
    figures = _extract_figures(doc, output_dir, dpi)

    # ── Bind in-text references ──
    figures = _bind_references(figures, full_text)

    # ── Extract tables ──
    tables = _extract_tables(doc)

    # ── Metadata ──
    metadata = _extract_metadata(doc)
    total_pages = len(doc)

    doc.close()

    return {
        "success": True,
        "pages": total_pages,
        "sections": sections,
        "figures": figures,
        "tables": tables,
        "full_text": full_text,
        "metadata": metadata,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract structured content from a paper PDF"
    )
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument(
        "--output-dir", "-o",
        default=None,
        help="Directory for extracted images (default: <pdf_name>_parsed/)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="DPI for figure rendering (default: 200, higher = better quality)",
    )
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help="Only extract figures, skip text parsing",
    )
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Only extract text sections, skip figure extraction",
    )

    args = parser.parse_args()

    pdf_path = Path(args.pdf_path)
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = str(pdf_path.parent / f"{pdf_path.stem}_parsed")

    result = parse_pdf(str(pdf_path), output_dir, args.dpi)

    # Remove full_text from output if figures-only
    if args.figures_only:
        result.pop("full_text", None)
        result.pop("sections", None)
    if args.text_only:
        result.pop("figures", None)
        result.pop("tables", None)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result.get("success"):
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Multi-Modal Cross Validator

Prepares cross-validation records for Claude to verify consistency between
paper text and figures. This tool bundles each figure with its surrounding
context, enabling Claude (multimodal) to:
  1. Describe what it sees in each figure
  2. Compare with the text-based architecture/training descriptions
  3. Flag any inconsistencies for user review

Usage:
    # Generate cross-validation records from pdf_parser output
    python cross_validator.py <pdf_parser_output.json> [--output report.json]

    # Also accepts reproduction_state.yaml as input (uses parsed_ fields)
    python cross_validator.py <reproduction_state.yaml>

Output:
    JSON array of cross-validation records, each containing:
    - figure image path, caption, surrounding text
    - claims extracted from nearby text (what the paper says)
    - a blank "visual_observation" and "consistency" field for Claude to fill
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional


# ── Claim extraction patterns ──────────────────────────────────────────────

# Patterns to find architecture-related claims in text
CLAIM_PATTERNS = [
    # Layer/dimension claims
    (re.compile(r"(\d+)\s*(?:-)?\s*layers?", re.IGNORECASE), "num_layers"),
    (re.compile(r"(?:hidden|embedding|feature)\s*(?:dim(?:ension)?|size)\s*(?:is|=|of)?\s*(\d+)", re.IGNORECASE), "hidden_dim"),
    (re.compile(r"(?:head|attention\s*head)s?\s*(?:is|=|of|:)?\s*(\d+)", re.IGNORECASE), "num_heads"),
    # Architecture patterns
    (re.compile(r"(ResNet|ViT|BERT|GPT|Swin|ConvNeXt|EfficientNet|DETR|YOLO|UNet|CLIP)(?:[-\s](\w+))?", re.IGNORECASE), "backbone"),
    (re.compile(r"(ReLU|GELU|SiLU|Swish|LeakyReLU|Sigmoid|Tanh)\s*(?:activation|nonlinearity)?", re.IGNORECASE), "activation"),
    (re.compile(r"(BatchNorm|LayerNorm|GroupNorm|InstanceNorm|RMSNorm)", re.IGNORECASE), "normalization"),
    # Training patterns
    (re.compile(r"(?:learning\s*rate|lr)\s*(?:is|=|of|:)?\s*([\d.e+-]+)", re.IGNORECASE), "learning_rate"),
    (re.compile(r"(?:batch\s*size)\s*(?:is|=|of|:)?\s*(\d+)", re.IGNORECASE), "batch_size"),
    (re.compile(r"(?:weight\s*decay)\s*(?:is|=|of|:)?\s*([\d.e+-]+)", re.IGNORECASE), "weight_decay"),
    # Optimizer
    (re.compile(r"(SGD|AdamW?|RMSprop|LAMB|LARS)(?:\s*optimizer)?", re.IGNORECASE), "optimizer"),
    # Loss
    (re.compile(r"(Cross(?:-|\s*)Entropy|MSE|L1\s*Loss|Focal\s*Loss|BCE|Contrastive\s*Loss|Triplet\s*Loss)", re.IGNORECASE), "loss_function"),
]


def extract_claims(text: str) -> dict:
    """
    Extract architecture and training claims from a block of text.
    Returns a dict mapping claim_type to the matched value.
    """
    claims = {}
    for pattern, claim_type in CLAIM_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            # Take the first match
            value = matches[0]
            if isinstance(value, tuple):
                value = " ".join(v for v in value if v)
            claims[claim_type] = value
    return claims


def _get_section_text_for_figure(figure: dict, sections: list) -> str:
    """
    Find the section text that contains or surrounds a given figure.
    Matches by page number proximity.
    """
    fig_page = figure.get("page", 0)

    # First, try to find the section on the same page
    for section in sections:
        if section.get("page", 0) == fig_page:
            return section.get("text", "")

    # Fallback: find the nearest section
    best_section = None
    best_distance = 999
    for section in sections:
        dist = abs(section.get("page", 0) - fig_page)
        if dist < best_distance:
            best_distance = dist
            best_section = section

    return best_section.get("text", "") if best_section else ""


def build_cross_validation_records(pdf_parser_output: dict) -> list:
    """
    Build cross-validation records from pdf_parser output.

    Each record bundles a figure with all context needed for visual-text comparison.
    """
    figures = pdf_parser_output.get("figures", [])
    sections = pdf_parser_output.get("sections", [])
    full_text = pdf_parser_output.get("full_text", "")

    records = []
    for fig in figures:
        # Find surrounding context
        caption = fig.get("caption", "")
        references = fig.get("references", [])
        section_text = _get_section_text_for_figure(fig, sections)

        # Extract claims from caption + references + section text
        combined_text = f"{caption}\n{' '.join(references)}\n{section_text}"
        claims = extract_claims(combined_text)

        record = {
            "figure_id": fig.get("id", ""),
            "figure_number": fig.get("number", ""),
            "figure_image_path": fig.get("image_path", ""),
            "page_image_path": fig.get("page_image_path", ""),
            "page": fig.get("page", 0),
            "caption": caption,
            "in_text_references": references,
            "surrounding_section_text": section_text[:2000],  # Truncate for context limits
            "claims_from_text": claims,
            # Fields for Claude to fill in:
            "visual_observation": "",
            "consistency_assessment": "",  # "consistent", "minor_discrepancy", "major_discrepancy", "cannot_verify"
            "discrepancies": [],
        }
        records.append(record)

    return records


def generate_validation_report(records: list) -> str:
    """
    Generate a human-readable validation report from filled-in records.
    """
    lines = ["# Multi-Modal Cross Validation Report\n"]

    consistent = 0
    minor = 0
    major = 0
    unverified = 0

    for rec in records:
        assessment = rec.get("consistency_assessment", "")
        if assessment == "consistent":
            consistent += 1
        elif assessment == "minor_discrepancy":
            minor += 1
        elif assessment == "major_discrepancy":
            major += 1
        else:
            unverified += 1

    lines.append("## Summary\n")
    lines.append(f"- Total figures analyzed: {len(records)}")
    lines.append(f"- Consistent: {consistent}")
    lines.append(f"- Minor discrepancies: {minor}")
    lines.append(f"- Major discrepancies: {major}")
    lines.append(f"- Cannot verify: {unverified}")
    lines.append("")

    if major > 0:
        lines.append("## Major Discrepancies (Require Investigation)\n")
        for rec in records:
            if rec.get("consistency_assessment") == "major_discrepancy":
                lines.append(f"### {rec['figure_number']}: {rec['caption'][:100]}")
                lines.append(f"**Visual observation**: {rec.get('visual_observation', 'N/A')}")
                lines.append(f"**Discrepancies**:")
                for d in rec.get("discrepancies", []):
                    lines.append(f"  - {d}")
                lines.append("")

    if minor > 0:
        lines.append("## Minor Discrepancies\n")
        for rec in records:
            if rec.get("consistency_assessment") == "minor_discrepancy":
                lines.append(f"- **{rec['figure_number']}**: {rec.get('discrepancies', ['See record'])[0]}")
        lines.append("")

    return "\n".join(lines)


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Prepare figure-text cross-validation records from parsed paper content"
    )
    parser.add_argument(
        "input_file",
        help="Path to pdf_parser output JSON or reproduction_state.yaml",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output path for cross-validation records JSON",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Generate a markdown report from filled-in records",
    )

    args = parser.parse_args()

    with open(args.input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if args.report:
        report = generate_validation_report(data)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(report)
            print(f"Report written to {args.output}")
        else:
            print(report)
        return

    records = build_cross_validation_records(data)
    output = {
        "total_figures": len(records),
        "records": records,
        "instruction": (
            "For each record, view the figure image and describe what you see in "
            "'visual_observation'. Then compare with 'claims_from_text' and set "
            "'consistency_assessment' to one of: consistent, minor_discrepancy, "
            "major_discrepancy, cannot_verify. List any specific discrepancies found."
        ),
    }

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"Cross-validation records written to {args.output}")
    else:
        print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

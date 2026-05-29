#!/usr/bin/env python3
"""
arXiv Paper Fetcher

Downloads a paper PDF from arXiv and retrieves its metadata (title, authors,
abstract, categories, etc.) via the arXiv API.

Usage:
    python arxiv_fetcher.py <arxiv_identifier> [--output-dir <dir>] [--no-download]

Input formats supported:
    - Full URL:  https://arxiv.org/abs/2410.12345
    - PDF URL:   https://arxiv.org/pdf/2410.12345
    - Bare ID:   2410.12345
    - ID prefix: arxiv:2410.12345

Output (JSON to stdout):
    {
        "success": true,
        "arxiv_id": "2410.12345",
        "title": "...",
        "authors": ["Author One", "Author Two"],
        "abstract": "...",
        "categories": ["cs.CV", "cs.AI"],
        "published": "2024-10-15",
        "pdf_path": "/tmp/arxiv_2410.12345/paper.pdf",
        "source_url": "https://arxiv.org/abs/2410.12345"
    }
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional


ARXIV_API_URL = "https://export.arxiv.org/api/query"
ARXIV_ID_PATTERN = re.compile(
    r"(?:arxiv[:\s]*)?(\d{4}\.\d{4,5})(?:v\d+)?"
)


def parse_arxiv_id(raw: str) -> Optional[str]:
    """Extract a bare arXiv ID from various input formats."""
    match = ARXIV_ID_PATTERN.search(raw)
    if match:
        return match.group(1)
    return None


def extract_arxiv_id_from_pdf_url(url: str) -> Optional[str]:
    """Extract arXiv ID from a direct PDF URL like arxiv.org/pdf/2410.12345.pdf"""
    # Strip query parameters and fragment
    base = url.split("?")[0].split("#")[0]
    match = re.search(r"/(\d{4}\.\d{4,5})(?:\.pdf)?", base)
    if match:
        return match.group(1)
    return None


def _http_request_with_retry(url: str, timeout: int = 30, max_retries: int = 3,
                             initial_delay: float = 0) -> bytes:
    """
    Make an HTTP request with exponential backoff for 429/503 responses.
    arXiv rate-limits aggressively; this retries with increasing delays.

    Args:
        url: The URL to request
        timeout: Connection timeout in seconds
        max_retries: Maximum number of retry attempts
        initial_delay: Delay before the first request (seconds), useful for
                       manual rate-limiting across multiple invocations
    """
    if initial_delay > 0:
        time.sleep(initial_delay)

    last_error = None
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Paper-Reproduce-Skill/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            last_error = e
            if e.code in (429, 503):
                delay = 3 * (2 ** attempt)
                time.sleep(delay)
            else:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionResetError, OSError) as e:
            last_error = e
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)

    raise urllib.error.URLError(
        f"Request failed after {max_retries} retries: {last_error}"
    )


def fetch_metadata(arxiv_id: str, initial_delay: float = 0) -> dict:
    """
    Query the arXiv API for paper metadata.

    Returns a dict with keys: arxiv_id, title, authors, abstract, categories,
    published, comment, source_url.
    """
    url = f"{ARXIV_API_URL}?id_list={arxiv_id}&max_results=1"

    try:
        xml_data = _http_request_with_retry(url, initial_delay=initial_delay).decode("utf-8")
    except urllib.error.URLError as e:
        return {"success": False, "error": f"arXiv API request failed: {e}"}

    root = ET.fromstring(xml_data)
    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }

    entry = root.find("atom:entry", ns)
    if entry is None:
        return {"success": False, "error": f"No entry found for arXiv ID: {arxiv_id}"}

    title = entry.find("atom:title", ns)
    summary = entry.find("atom:summary", ns)
    published = entry.find("atom:published", ns)

    authors = [
        author.find("atom:name", ns).text.strip()
        for author in entry.findall("atom:author", ns)
        if author.find("atom:name", ns) is not None
    ]

    categories = [
        cat.get("term")
        for cat in entry.findall("atom:category", ns)
        if cat.get("term")
    ]

    comment_elem = entry.find("arxiv:comment", ns)
    comment = comment_elem.text.strip() if comment_elem is not None and comment_elem.text else ""

    result = {
        "success": True,
        "arxiv_id": arxiv_id,
        "title": title.text.strip().replace("\n", " ") if title is not None and title.text else "",
        "authors": authors,
        "abstract": summary.text.strip().replace("\n", " ") if summary is not None and summary.text else "",
        "categories": categories,
        "published": published.text.strip()[:10] if published is not None and published.text else "",
        "comment": comment,
        "source_url": f"https://arxiv.org/abs/{arxiv_id}",
    }

    return result


def download_pdf(arxiv_id: str, output_dir: str, initial_delay: float = 0) -> dict:
    """
    Download the PDF for a given arXiv ID.

    Returns a dict with pdf_path on success, or error info on failure.
    """
    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, "paper.pdf")

    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

    try:
        pdf_data = _http_request_with_retry(pdf_url, timeout=60, initial_delay=initial_delay)
        with open(pdf_path, "wb") as f:
            f.write(pdf_data)

        file_size = os.path.getsize(pdf_path)
        if file_size < 1024:
            os.remove(pdf_path)
            return {"success": False, "error": "Downloaded file too small (< 1KB), likely not a valid PDF"}

        return {"success": True, "pdf_path": pdf_path, "file_size": file_size}

    except urllib.error.HTTPError as e:
        return {"success": False, "error": f"HTTP {e.code}: {e.reason}"}
    except urllib.error.URLError as e:
        return {"success": False, "error": f"Download failed: {e.reason}"}


def main():
    parser = argparse.ArgumentParser(
        description="Download a paper PDF and fetch metadata from arXiv"
    )
    parser.add_argument(
        "identifier",
        help="arXiv URL, ID (e.g., 2410.12345), or DOI",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=None,
        help="Directory to save the PDF (default: /tmp/arxiv_<id>/)",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Only fetch metadata, do not download PDF",
    )
    parser.add_argument(
        "--wait", "-w",
        type=float,
        default=0,
        help="Delay in seconds before each request to avoid rate limiting (default: 0)",
    )

    args = parser.parse_args()
    initial_delay = args.wait

    # Handle PDF URL separately (different extraction pattern)
    if "/pdf/" in args.identifier or args.identifier.endswith(".pdf"):
        arxiv_id = extract_arxiv_id_from_pdf_url(args.identifier)
    else:
        arxiv_id = parse_arxiv_id(args.identifier)

    if not arxiv_id:
        print(json.dumps({
            "success": False,
            "error": f"Cannot extract arXiv ID from: {args.identifier}"
        }, ensure_ascii=False, indent=2))
        sys.exit(1)

    # Step 1: Fetch metadata
    metadata = fetch_metadata(arxiv_id, initial_delay=initial_delay)
    if not metadata.get("success"):
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
        sys.exit(1)

    # Step 2: Download PDF (if requested)
    if not args.no_download:
        output_dir = args.output_dir or f"/tmp/arxiv_{arxiv_id}"
        download_result = download_pdf(arxiv_id, output_dir, initial_delay=initial_delay)
        metadata.update(download_result)
    else:
        metadata["pdf_path"] = None

    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    if not metadata.get("success", True):
        sys.exit(1)


if __name__ == "__main__":
    main()

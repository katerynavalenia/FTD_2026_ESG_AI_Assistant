
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import fitz


MARGIN_TOP_RATIO = 0.08
MARGIN_BOTTOM_RATIO = 0.92
HEADER_FOOTER_MIN_PAGES = 3
HEADING_MIN_FONT_SIZE = 12.0


@dataclass
class PageBlock:
    text: str
    normalized_text: str
    y0: float
    y1: float
    avg_font_size: float
    max_font_size: float
    is_margin_candidate: bool


def clean_whitespace(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_for_repeat_detection(text: str) -> str:
    text = clean_whitespace(text).lower()
    text = re.sub(r"\d+", "<num>", text)
    text = re.sub(r"[^a-z0-9<>%&/().,\- ]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def looks_like_heading(text: str, avg_font_size: float, max_font_size: float, baseline_font_size: float) -> bool:
    if not text:
        return False
    if len(text) > 160:
        return False
    if re.fullmatch(r"[\d\W_]+", text):
        return False
    if text.endswith(".") and len(text.split()) > 10:
        return False

    uppercase_ratio = sum(1 for char in text if char.isupper()) / max(1, sum(1 for char in text if char.isalpha()))
    title_like = text == text.title() and len(text.split()) <= 12
    short_label = len(text.split()) <= 10
    strong_font = max_font_size >= max(HEADING_MIN_FONT_SIZE, baseline_font_size * 1.2)

    return strong_font and (short_label or title_like or uppercase_ratio > 0.55)


def extract_page_blocks(page: fitz.Page) -> list[PageBlock]:
    page_dict = page.get_text("dict", sort=True)
    page_height = page.rect.height
    blocks: list[PageBlock] = []

    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:
            continue

        lines: list[str] = []
        font_sizes: list[float] = []
        for line in block.get("lines", []):
            span_texts: list[str] = []
            for span in line.get("spans", []):
                text = clean_whitespace(span.get("text", ""))
                if not text:
                    continue
                span_texts.append(text)
                font_sizes.append(float(span.get("size", 0.0)))
            if span_texts:
                lines.append(" ".join(span_texts))

        raw_text = "\n".join(lines).strip()
        if not raw_text:
            continue

        normalized_text = normalize_for_repeat_detection(raw_text)
        if not normalized_text:
            continue

        y0 = float(block.get("bbox", [0, 0, 0, 0])[1])
        y1 = float(block.get("bbox", [0, 0, 0, 0])[3])
        is_margin_candidate = y0 <= page_height * MARGIN_TOP_RATIO or y1 >= page_height * MARGIN_BOTTOM_RATIO

        blocks.append(
            PageBlock(
                text=raw_text,
                normalized_text=normalized_text,
                y0=y0,
                y1=y1,
                avg_font_size=(sum(font_sizes) / len(font_sizes)) if font_sizes else 0.0,
                max_font_size=max(font_sizes) if font_sizes else 0.0,
                is_margin_candidate=is_margin_candidate,
            )
        )

    return blocks


def detect_repeated_margin_blocks(document_blocks: list[list[PageBlock]]) -> set[str]:
    counts: Counter[str] = Counter()
    page_count = len(document_blocks)

    for page_blocks in document_blocks:
        seen_on_page = set()
        for block in page_blocks:
            if block.is_margin_candidate and len(block.normalized_text) >= 4:
                seen_on_page.add(block.normalized_text)
        counts.update(seen_on_page)

    repeated: set[str] = set()
    for normalized_text, count in counts.items():
        if count >= HEADER_FOOTER_MIN_PAGES and count >= max(3, int(page_count * 0.2)):
            repeated.add(normalized_text)
    return repeated


def merge_block_texts(blocks: list[PageBlock], repeated_margin_texts: set[str]) -> tuple[str, list[str], float]:
    kept_blocks: list[PageBlock] = []
    font_sizes = [block.avg_font_size for block in blocks if block.avg_font_size > 0]
    baseline_font_size = sorted(font_sizes)[len(font_sizes) // 2] if font_sizes else HEADING_MIN_FONT_SIZE

    headings: list[str] = []
    paragraphs: list[str] = []

    for block in blocks:
        if block.normalized_text in repeated_margin_texts and block.is_margin_candidate:
            continue

        cleaned = clean_whitespace(block.text)
        if not cleaned:
            continue

        if looks_like_heading(cleaned, block.avg_font_size, block.max_font_size, baseline_font_size):
            headings.append(cleaned)

        kept_blocks.append(block)
        paragraphs.append(cleaned.replace("\n", " "))

    page_text = "\n\n".join(paragraphs)
    page_text = re.sub(r"(\w)-\s+(\w)", r"\1\2", page_text)
    page_text = re.sub(r"[ \t]+", " ", page_text)
    page_text = re.sub(r"\n{3,}", "\n\n", page_text).strip()

    deduped_headings: list[str] = []
    seen = set()
    for heading in headings:
        key = heading.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped_headings.append(heading)

    return page_text, deduped_headings, baseline_font_size


def load_metadata(metadata_path: Path) -> list[dict[str, str]]:
    with metadata_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def resolve_pdf_path(metadata_path: Path, row: dict[str, str]) -> Path:
    relative_path = row["relative_path"]
    return metadata_path.parent / relative_path


def parse_pdf(metadata_row: dict[str, str], metadata_path: Path, output_dir: Path) -> list[dict[str, object]]:
    pdf_path = resolve_pdf_path(metadata_path, metadata_row)
    try:
        document = fitz.open(pdf_path)
    except Exception as exc:
        logging.warning("Could not open PDF %s: %s — skipping.", pdf_path, exc)
        return []

    try:
        page_blocks = [extract_page_blocks(page) for page in document]
        repeated_margin_texts = detect_repeated_margin_blocks(page_blocks)

        page_records: list[dict[str, object]] = []
        current_section = ""

        for page_number, blocks in enumerate(page_blocks, start=1):
            page_text, headings, baseline_font_size = merge_block_texts(blocks, repeated_margin_texts)
            if headings:
                current_section = headings[0]

            if not page_text:
                continue

            record = {
                "company": metadata_row["company"],
                "report_type": metadata_row["report_type"],
                "year": metadata_row["year"],
                "source_page": metadata_row["source_page"],
                "candidate_url": metadata_row["candidate_url"],
                "final_url": metadata_row["final_url"],
                "pdf_path": str(pdf_path),
                "pdf_filename": metadata_row["filename"],
                "page_number": page_number,
                "page_label": document[page_number - 1].get_label() or str(page_number),
                "section_title": current_section,
                "headings": headings,
                "word_count": len(page_text.split()),
                "char_count": len(page_text),
                "baseline_font_size": round(baseline_font_size, 2),
                "text": page_text,
            }
            page_records.append(record)

        company_slug = slugify(metadata_row["company"])
        document_stem = Path(metadata_row["filename"]).stem
        company_dir = output_dir / company_slug
        company_dir.mkdir(parents=True, exist_ok=True)

        jsonl_path = company_dir / f"{document_stem}.jsonl"
        txt_path = company_dir / f"{document_stem}.txt"

        with jsonl_path.open("w", encoding="utf-8") as handle:
            for record in page_records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        full_text = "\n\n".join(
            f"[Page {record['page_number']}] {record['text']}" for record in page_records
        ).strip()
        txt_path.write_text(full_text, encoding="utf-8")

        logging.info("Parsed %s -> %s pages", metadata_row["filename"], len(page_records))
        return page_records
    finally:
        document.close()


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def write_manifest(records: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "company",
        "report_type",
        "year",
        "pdf_filename",
        "page_number",
        "page_label",
        "section_title",
        "word_count",
        "char_count",
        "pdf_path",
        "final_url",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key, "") for key in fieldnames})


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse ESG PDFs into page-level clean text.")
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=Path("data/pdf_metadata.csv"),
        help="CSV file generated by the scraper.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/parsed_pages"),
        help="Directory where parsed page JSONL/TXT files will be stored.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("data/parsed_pages_manifest.csv"),
        help="CSV summary with one row per parsed page.",
    )
    parser.add_argument(
        "--companies",
        nargs="*",
        default=[],
        help="Optional list of company names to parse.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional maximum number of PDFs to parse.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    rows = load_metadata(args.metadata_path)
    if args.companies:
        wanted = {name.lower() for name in args.companies}
        rows = [row for row in rows if row["company"].lower() in wanted]
    if args.limit > 0:
        rows = rows[: args.limit]

    if not rows:
        logging.warning("No PDF metadata rows selected for parsing.")
        return 0

    all_records: list[dict[str, object]] = []
    for row in rows:
        pdf_path = resolve_pdf_path(args.metadata_path, row)
        if not pdf_path.exists():
            logging.warning("Skipping missing PDF: %s", pdf_path)
            continue
        all_records.extend(parse_pdf(row, args.metadata_path, args.output_dir))

    if all_records:
        write_manifest(all_records, args.manifest_path)
        logging.info("Manifest written to %s", args.manifest_path)
    else:
        logging.warning("No pages were parsed.")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

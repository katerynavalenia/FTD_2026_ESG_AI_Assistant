
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path


TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\"\'])")
PARAGRAPH_SPLIT_PATTERN = re.compile(r"\n\s*\n+")


@dataclass
class Segment:
    text: str
    token_count: int
    page_number: int
    page_label: str
    section_title: str


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def normalize_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def estimate_tokens(text: str) -> int:
    return len(TOKEN_PATTERN.findall(text))


def split_sentences(text: str) -> list[str]:
    text = normalize_text(text)
    if not text:
        return []
    parts = [part.strip() for part in SENTENCE_SPLIT_PATTERN.split(text) if part.strip()]
    return parts if parts else [text]


def split_text_by_token_window(text: str, max_tokens: int) -> list[str]:
    rough_tokens = TOKEN_PATTERN.findall(text)
    if not rough_tokens:
        return []

    windows: list[str] = []
    for start in range(0, len(rough_tokens), max_tokens):
        window = rough_tokens[start : start + max_tokens]
        rebuilt = " ".join(window)
        rebuilt = re.sub(r"\s+([,.;:!?%)\]])", r"\1", rebuilt)
        rebuilt = re.sub(r"([(\[])\s+", r"\1", rebuilt)
        windows.append(rebuilt.strip())
    return [window for window in windows if window]


def split_paragraph(text: str, max_tokens: int) -> list[str]:
    text = normalize_text(text)
    if not text:
        return []
    if estimate_tokens(text) <= max_tokens:
        return [text]

    sentences = split_sentences(text)
    if len(sentences) <= 1:
        return split_text_by_token_window(text, max_tokens)

    parts: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = estimate_tokens(sentence)
        if sentence_tokens > max_tokens:
            if current:
                parts.append(" ".join(current).strip())
                current = []
                current_tokens = 0
            parts.extend(split_text_by_token_window(sentence, max_tokens))
            continue

        if current and current_tokens + sentence_tokens > max_tokens:
            parts.append(" ".join(current).strip())
            current = [sentence]
            current_tokens = sentence_tokens
        else:
            current.append(sentence)
            current_tokens += sentence_tokens

    if current:
        parts.append(" ".join(current).strip())

    return [part for part in parts if part]


def load_page_records(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def build_segments(page_records: list[dict[str, object]], max_tokens: int) -> list[Segment]:
    segments: list[Segment] = []
    for record in page_records:
        text = str(record.get("text", "")).strip()
        if not text:
            continue
        page_number = int(record["page_number"])
        page_label = str(record.get("page_label", page_number))
        section_title = str(record.get("section_title", "")).strip()

        for paragraph in PARAGRAPH_SPLIT_PATTERN.split(text):
            cleaned = normalize_text(paragraph)
            if not cleaned:
                continue
            for part in split_paragraph(cleaned, max_tokens=max_tokens):
                segments.append(
                    Segment(
                        text=part,
                        token_count=estimate_tokens(part),
                        page_number=page_number,
                        page_label=page_label,
                        section_title=section_title,
                    )
                )
    return segments


def unique_preserve_order(values: list[str]) -> list[str]:
    seen = set()
    result: list[str] = []
    for value in values:
        if not value:
            continue
        key = value.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(value.strip())
    return result


def build_chunks(
    segments: list[Segment],
    target_tokens: int,
    max_tokens: int,
    overlap_tokens: int,
    min_tokens: int,
) -> list[list[Segment]]:
    if not segments:
        return []

    chunks: list[list[Segment]] = []
    start = 0

    while start < len(segments):
        total = 0
        end = start

        while end < len(segments):
            next_tokens = segments[end].token_count
            would_exceed = total > 0 and total + next_tokens > max_tokens
            if would_exceed and total > 0:
                break

            total += next_tokens
            end += 1

            if total >= target_tokens:
                if end >= len(segments):
                    break
                if total + segments[end].token_count > max_tokens:
                    break

        if end <= start:
            end = start + 1

        chunk_segments = segments[start:end]
        chunks.append(chunk_segments)

        if end >= len(segments):
            break

        if overlap_tokens <= 0:
            start = end
            continue

        next_start = end
        overlap = 0
        while next_start > start + 1 and overlap < overlap_tokens:
            next_start -= 1
            overlap += segments[next_start].token_count

        if next_start <= start:
            start = end
        else:
            start = next_start

    return chunks


def chunk_record_from_segments(
    chunk_segments: list[Segment],
    page_records: list[dict[str, object]],
    chunk_index: int,
) -> dict[str, object]:
    first_page = page_records[0]
    page_numbers = list(dict.fromkeys(segment.page_number for segment in chunk_segments))
    page_labels = unique_preserve_order([segment.page_label for segment in chunk_segments])
    section_titles = unique_preserve_order([segment.section_title for segment in chunk_segments])
    text = "\n\n".join(segment.text for segment in chunk_segments).strip()
    token_count = sum(segment.token_count for segment in chunk_segments)

    pdf_filename = str(first_page["pdf_filename"])
    pdf_stem = Path(pdf_filename).stem
    company_slug = slugify(str(first_page["company"]))
    chunk_id = f"{company_slug}::{pdf_stem}::{chunk_index:04d}"

    page_start = min(page_numbers)
    page_end = max(page_numbers)
    citation = (
        f"{first_page['company']} | {pdf_filename} | page {page_start}"
        if page_start == page_end
        else f"{first_page['company']} | {pdf_filename} | pages {page_start}-{page_end}"
    )

    return {
        "chunk_id": chunk_id,
        "company": first_page["company"],
        "report_type": first_page["report_type"],
        "year": first_page["year"],
        "pdf_filename": pdf_filename,
        "pdf_path": first_page["pdf_path"],
        "final_url": first_page["final_url"],
        "chunk_index": chunk_index,
        "page_start": page_start,
        "page_end": page_end,
        "page_numbers": page_numbers,
        "page_labels": page_labels,
        "section_title": section_titles[0] if section_titles else "",
        "section_titles": section_titles,
        "token_count_estimate": token_count,
        "word_count": len(text.split()),
        "char_count": len(text),
        "citation": citation,
        "text": text,
    }


def write_jsonl(records: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_manifest(records: list[dict[str, object]], path: Path) -> None:
    fieldnames = [
        "chunk_id",
        "company",
        "report_type",
        "year",
        "pdf_filename",
        "chunk_index",
        "page_start",
        "page_end",
        "section_title",
        "token_count_estimate",
        "word_count",
        "char_count",
        "citation",
        "pdf_path",
        "final_url",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field, "") for field in fieldnames})


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chunk parsed ESG reports into RAG-ready segments.")
    parser.add_argument(
        "--parsed-dir",
        type=Path,
        default=Path("data/parsed_pages"),
        help="Directory containing parsed page JSONL files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/chunks"),
        help="Directory where chunk JSONL files will be written.",
    )
    parser.add_argument(
        "--all-chunks-path",
        type=Path,
        default=Path("data/chunks/all_chunks.jsonl"),
        help="Global JSONL file containing all chunks.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("data/chunks_manifest.csv"),
        help="CSV summary with one row per chunk.",
    )
    parser.add_argument(
        "--target-tokens",
        type=int,
        default=400,
        help="Target chunk size.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=500,
        help="Hard chunk size ceiling.",
    )
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=280,
        help="Preferred minimum chunk size before stopping a chunk.",
    )
    parser.add_argument(
        "--overlap-tokens",
        type=int,
        default=80,
        help="Approximate overlap between consecutive chunks.",
    )
    parser.add_argument(
        "--companies",
        nargs="*",
        default=[],
        help="Optional company filters, using company folder names like totalenergies or engie.",
    )
    parser.add_argument(
        "--limit-files",
        type=int,
        default=0,
        help="Optional maximum number of parsed JSONL files to chunk.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    jsonl_files = sorted(path for path in args.parsed_dir.glob("*/*.jsonl"))
    if args.companies:
        company_filter = {slugify(company) for company in args.companies}
        jsonl_files = [path for path in jsonl_files if path.parent.name in company_filter]
    if args.limit_files > 0:
        jsonl_files = jsonl_files[: args.limit_files]

    if not jsonl_files:
        logging.warning("No parsed JSONL files selected for chunking.")
        return 0

    all_chunk_records: list[dict[str, object]] = []

    for jsonl_path in jsonl_files:
        page_records = load_page_records(jsonl_path)
        if not page_records:
            logging.warning("Skipping empty parsed file: %s", jsonl_path)
            continue

        segments = build_segments(page_records, max_tokens=args.max_tokens)
        chunk_segments = build_chunks(
            segments,
            target_tokens=args.target_tokens,
            max_tokens=args.max_tokens,
            overlap_tokens=args.overlap_tokens,
            min_tokens=args.min_tokens,
        )

        chunk_records = [
            chunk_record_from_segments(chunk, page_records, index)
            for index, chunk in enumerate(chunk_segments, start=1)
        ]
        all_chunk_records.extend(chunk_records)

        output_company_dir = args.output_dir / jsonl_path.parent.name
        output_company_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_company_dir / jsonl_path.name
        write_jsonl(chunk_records, output_path)
        logging.info("Chunked %s -> %s chunks", jsonl_path.name, len(chunk_records))

    write_jsonl(all_chunk_records, args.all_chunks_path)
    write_manifest(all_chunk_records, args.manifest_path)
    logging.info("Global chunk corpus written to %s", args.all_chunks_path)
    logging.info("Chunk manifest written to %s", args.manifest_path)
    logging.info("Total chunks: %s", len(all_chunk_records))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

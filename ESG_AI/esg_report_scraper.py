
from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import parse_qsl, unquote, urlencode, urljoin, urlparse, urlunparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup, Tag
from requests import Response, Session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


CURRENT_YEAR = 2026
DEFAULT_DELAY_SECONDS = 1.5
DEFAULT_TIMEOUT = 30
PDF_SIGNATURE = b"%PDF-"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/135.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/pdf,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,fr;q=0.8",
    "Upgrade-Insecure-Requests": "1",
}

GENERIC_ANCHOR_TEXTS = {
    "",
    "download",
    "download pdf",
    "pdf",
    "open",
    "read more",
    "view",
}

POSITIVE_PATTERNS: tuple[tuple[str, int], ...] = (
    ("sustainability & climate progress report", 125),
    ("sustainability and climate progress report", 125),
    ("progress report", 60),
    ("sustainability statement", 118),
    ("climate report", 108),
    ("integrated report", 98),
    ("extra-financial report", 102),
    ("universal registration document", 78),
    ("registration document", 70),
    ("sustainability report", 102),
    ("esg report", 96),
    ("annual report", 88),
    ("csr report", 82),
    ("databook", 35),
)

NEGATIVE_PATTERNS: tuple[tuple[str, int], ...] = (
    ("slide deck", -150),
    ("presentation", -145),
    ("webcast", -140),
    ("xhtml", -135),
    ("xbrl", -135),
    ("esef", -125),
    ("financial statements", -120),
    ("board of directors", -115),
    ("statutory auditors", -110),
    ("press release", -105),
    ("code of conduct", -95),
    ("policy", -90),
    ("factsheet", -85),
    ("brochure", -80),
    ("summary", -25),
)

REPORT_TYPES = {
    "sustainability_statement": (
        "sustainability statement",
        "sustainability & climate progress report",
        "sustainability and climate progress report",
        "climate report",
        "sustainability report",
        "esg report",
        "extra-financial report",
        "progress report",
    ),
    "integrated_report": (
        "integrated report",
        "integrated annual report",
        "iar",
    ),
    "universal_registration_document": (
        "universal registration document",
        "registration document",
        "document d'enregistrement universel",
        "urd",
        "deu",
    ),
    "annual_report": (
        "annual report",
    ),
    "esg_databook": (
        "databook",
    ),
}

REPORT_PRIORITY = (
    "sustainability_statement",
    "integrated_report",
    "annual_report",
    "universal_registration_document",
    "esg_databook",
    "other",
)

REPORT_PRIORITY_BONUS = {
    "sustainability_statement": 35,
    "integrated_report": 30,
    "annual_report": 24,
    "universal_registration_document": 18,
    "esg_databook": 4,
    "other": 0,
}

DOCUMENT_SIGNAL_TERMS = (
    "report",
    "statement",
    "registration",
    "document",
    "annual",
    "integrated",
    "sustainability",
    "climate",
    "esg",
    "extra-financial",
    "databook",
    "urd",
    "deu",
    "iar",
)


@dataclass(frozen=True)
class CompanyConfig:
    name: str
    start_urls: tuple[str, ...]
    allowed_domains: tuple[str, ...]
    required_terms: tuple[str, ...]
    preferred_terms: tuple[str, ...] = ()
    excluded_terms: tuple[str, ...] = ()


@dataclass
class CandidateLink:
    company: str
    source_page: str
    url: str
    context: str
    score: int
    report_type: str


COMPANIES: dict[str, CompanyConfig] = {
    "totalenergies": CompanyConfig(
        name="TotalEnergies",
        start_urls=(
            "https://totalenergies.com/sustainability",
            "https://totalenergies.com/sustainability/our-approach/esg-documentation",
        ),
        allowed_domains=("totalenergies.com",),
        required_terms=("report", "document", "statement"),
        preferred_terms=("sustainability", "climate", "progress report"),
        excluded_terms=("presentation", "policy", "press release"),
    ),
    "bnp_paribas": CompanyConfig(
        name="BNP Paribas",
        start_urls=(
            "https://group.bnpparibas/en/group/about-us/company-purpose",
            "https://group.bnpparibas/en/news/discover-the-2024-edition-of-the-groups-integrated-report",
            "https://invest.bnpparibas/en/search/reports/documents/financial-reports",
        ),
        allowed_domains=(
            "group.bnpparibas",
            "group.bnpparibas.com",
            "cdn-group.bnpparibas.com",
            "invest.bnpparibas",
            "reports.invest.bnpparibas",
        ),
        required_terms=("report", "document", "statement"),
        preferred_terms=("integrated", "sustainability", "climate", "annual report"),
        excluded_terms=("presentation", "policy", "press release", "anti-corruption", "fatca"),
    ),
    "airbus": CompanyConfig(
        name="Airbus",
        start_urls=(
            "https://www.airbus.com/en/investors/annual-reports",
            "https://www.airbus.com/en/sustainability/sustainability-standards-and-performance",
        ),
        allowed_domains=("airbus.com", "www.airbus.com"),
        required_terms=("report", "document", "statement"),
        preferred_terms=("annual report", "sustainability", "integrated"),
        excluded_terms=("board of directors", "financial statements", "presentation", "esef", "overview"),
    ),
    "danone": CompanyConfig(
        name="Danone",
        start_urls=(
            "https://www.danone.com/investors/publications-and-events/financial-and-extra-financial-reports.html",
            "https://www.danone.com/investors.html",
        ),
        allowed_domains=("danone.com", "www.danone.com"),
        required_terms=("report", "document", "statement"),
        preferred_terms=("extra-financial", "annual report", "integrated", "sustainability", "iar"),
        excluded_terms=("presentation", "press release", "half-year", "results", "summary"),
    ),
    "engie": CompanyConfig(
        name="ENGIE",
        start_urls=(
            "https://www.engie.com/en/investors/ESG",
            "https://www.engie.com/en/group/social-responsibility/csr-publications",
        ),
        allowed_domains=("engie.com", "www.engie.com"),
        required_terms=("report", "document", "statement"),
        preferred_terms=("sustainability statement", "integrated report", "climate"),
        excluded_terms=("slide deck", "presentation", "press release"),
    ),
    "schneider_electric": CompanyConfig(
        name="Schneider Electric",
        start_urls=(
            "https://www.se.com/ww/en/about-us/sustainability/sustainability-reports.jsp",
            "https://www.se.com/ww/en/about-us/investor-relations/regulated-information.jsp",
        ),
        allowed_domains=("se.com", "www.se.com", "download.schneider-electric.com"),
        required_terms=("report", "document", "statement", "impact"),
        preferred_terms=("sustainability impact", "csrd", "climate", "esg"),
        excluded_terms=("slide deck", "presentation", "press release", "q1", "q2", "q3", "q4"),
    ),
    "loreal": CompanyConfig(
        name="L'Oreal",
        start_urls=(
            "https://www.loreal-finance.com/en/annual-report",
            "https://www.loreal-finance.com/en/investor-relations/publications",
        ),
        allowed_domains=("loreal-finance.com", "www.loreal-finance.com"),
        required_terms=("report", "document", "statement"),
        preferred_terms=("universal registration document", "sustainability", "annual report"),
        excluded_terms=("presentation", "press release", "half-year", "results", "summary"),
    ),
    "volkswagen": CompanyConfig(
        name="Volkswagen",
        start_urls=(
            "https://annualreport2024.volkswagen-group.com/en/",
            "https://www.volkswagen-group.com/en/publications/more/annual-and-sustainability-reports-1813",
        ),
        allowed_domains=(
            "annualreport2024.volkswagen-group.com",
            "www.volkswagen-group.com",
            "annualreport2023.volkswagen-group.com",
        ),
        required_terms=("report", "document", "statement"),
        preferred_terms=("sustainability report", "esrs", "annual report", "esg"),
        excluded_terms=("presentation", "press release", "interim", "half-year"),
    ),
    "siemens": CompanyConfig(
        name="Siemens",
        start_urls=(
            "https://www.siemens.com/global/en/company/investor-relations/reports-financials/annual-report.html",
            "https://www.siemens.com/global/en/company/sustainability.html",
        ),
        allowed_domains=(
            "siemens.com",
            "www.siemens.com",
            "assets.new.siemens.com",
        ),
        required_terms=("report", "document", "statement"),
        preferred_terms=("sustainability statement", "annual report", "esg"),
        excluded_terms=("presentation", "press release", "quarterly", "interim"),
    ),
    "iberdrola": CompanyConfig(
        name="Iberdrola",
        start_urls=(
            "https://www.iberdrola.com/shareholders-investors/annual-report",
            "https://www.iberdrola.com/sustainability/sustainability-report",
        ),
        allowed_domains=("iberdrola.com", "www.iberdrola.com"),
        required_terms=("report", "document", "statement"),
        preferred_terms=("sustainability report", "annual report", "esg", "climate"),
        excluded_terms=("presentation", "press release", "quarterly", "interim"),
    ),
}


def build_session() -> Session:
    session = requests.Session()
    retries = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=0.8,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update(HEADERS)
    return session


class RobotsCache:
    def __init__(self, session: Session) -> None:
        self.session = session
        self.parsers: dict[str, RobotFileParser | None] = {}

    def can_fetch(self, url: str) -> bool | None:
        parsed = urlparse(url)
        host_key = f"{parsed.scheme}://{parsed.netloc}"

        if host_key not in self.parsers:
            robots_url = f"{host_key}/robots.txt"
            try:
                response = self.session.get(robots_url, timeout=15)
            except requests.RequestException as exc:
                logging.warning("Robots fetch failed for %s: %s", robots_url, exc)
                self.parsers[host_key] = None
            else:
                if response.status_code >= 400:
                    logging.warning("Robots unavailable for %s (%s)", robots_url, response.status_code)
                    self.parsers[host_key] = None
                else:
                    parser = RobotFileParser()
                    parser.set_url(robots_url)
                    parser.parse(response.text.splitlines())
                    self.parsers[host_key] = parser

        parser = self.parsers[host_key]
        if parser is None:
            return None
        return parser.can_fetch(HEADERS["User-Agent"], url)


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def strip_tracking_params(url: str) -> str:
    parsed = urlparse(url)
    if not parsed.query:
        return urlunparse(parsed._replace(fragment=""))

    kept_query = []
    for key, value in parse_qsl(parsed.query, keep_blank_values=True):
        lowered = key.lower()
        if lowered.startswith("utm_"):
            continue
        if lowered in {"cmpid", "mc_cid", "mc_eid"}:
            continue
        kept_query.append((key, value))

    return urlunparse(parsed._replace(query=urlencode(kept_query), fragment=""))


def is_allowed_domain(url: str, allowed_domains: Iterable[str]) -> bool:
    host = urlparse(url).netloc.lower()
    return any(host == domain or host.endswith(f".{domain}") for domain in allowed_domains)


def infer_report_type(text: str) -> str:
    lowered = text.lower()
    for report_type, patterns in REPORT_TYPES.items():
        if any(pattern in lowered for pattern in patterns):
            return report_type
    return "other"


def extract_year(text: str) -> int | None:
    years = [int(match) for match in re.findall(r"\b(20\d{2})\b", text)]
    if not years:
        return None
    return max(years)


def extract_document_year(*texts: str) -> int | None:
    combined = " ".join(texts)
    normalized = clean_text(combined).lower()
    keyword_group = (
        r"(?:integrated|annual|sustainability|climate|extra-financial|esg|"
        r"registration|progress|report|statement|document|urd|deu|iar)"
    )
    patterns = (
        rf"\b(20\d{{2}})\b[\w\s\-]{{0,24}}{keyword_group}",
        rf"{keyword_group}[\w\s\-]{{0,24}}\b(20\d{{2}})\b",
    )
    for pattern in patterns:
        matches = re.findall(pattern, normalized)
        if matches:
            return int(matches[0])
    return extract_year(normalized)


def candidate_year(candidate: CandidateLink) -> int:
    return extract_document_year(candidate.url, candidate.context) or 0


def score_candidate(url: str, signal_text: str, context: str, company: CompanyConfig) -> tuple[int, str]:
    signal = f"{url} {signal_text}".lower()
    full_text = f"{signal} {context}".lower()
    score = 0

    if ".pdf" in signal:
        score += 25

    report_type = infer_report_type(signal)
    if report_type == "other":
        report_type = infer_report_type(full_text)

    if report_type == "sustainability_statement":
        score += 80
    elif report_type == "integrated_report":
        score += 70
    elif report_type == "annual_report":
        score += 55
    elif report_type == "universal_registration_document":
        score += 45
    elif report_type == "esg_databook":
        score += 8

    for pattern, weight in POSITIVE_PATTERNS:
        if pattern in signal:
            score += weight
        elif pattern in full_text and report_type != "other":
            score += max(4, int(weight * 0.35))

    for pattern, weight in NEGATIVE_PATTERNS:
        if pattern in signal:
            score += weight
        elif pattern in full_text:
            score += int(weight * 0.5)

    if any(term in signal for term in company.required_terms):
        score += 15
    elif any(term in full_text for term in company.required_terms):
        score += 4
    else:
        score -= 40

    for term in company.preferred_terms:
        if term in signal:
            score += 18
        elif term in full_text:
            score += 6

    for term in company.excluded_terms:
        if term in signal:
            score -= 85
        elif term in full_text:
            score -= 35

    year = extract_document_year(signal, context)
    if year is not None:
        score += max(-80, 45 - abs(CURRENT_YEAR - year) * 12)
        if year < CURRENT_YEAR - 6:
            score -= 70

    if "english" in signal or re.search(r"\ben\b", signal):
        score += 4
    if "fran" in signal or "french" in signal:
        score -= 2

    return score, report_type


def context_from_anchor(anchor: Tag) -> str:
    parts: list[str] = []
    own_text = clean_text(anchor.get_text(" ", strip=True))
    if own_text:
        parts.append(own_text)

    for attr in ("title", "aria-label", "download"):
        attr_value = clean_text(str(anchor.get(attr, "")))
        if attr_value:
            parts.append(attr_value)

    current: Tag | None = anchor
    depth = 0
    while current is not None and depth < 3:
        parent = current.parent
        if not isinstance(parent, Tag):
            break
        parent_text = clean_text(parent.get_text(" ", strip=True))
        if 20 <= len(parent_text) <= 320:
            parts.append(parent_text)
        current = parent
        depth += 1

    unique_parts: list[str] = []
    seen = set()
    for part in parts:
        if not part:
            continue
        normalized = part.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        unique_parts.append(part)

    if own_text.lower() in GENERIC_ANCHOR_TEXTS and len(unique_parts) > 1:
        return " | ".join(unique_parts[:3])
    return " | ".join(unique_parts[:2])


def signal_from_anchor(anchor: Tag, absolute_url: str) -> str:
    parts = [Path(urlparse(absolute_url).path).name]
    own_text = clean_text(anchor.get_text(" ", strip=True))
    if own_text:
        parts.append(own_text)
    for attr in ("title", "aria-label", "download"):
        attr_value = clean_text(str(anchor.get(attr, "")))
        if attr_value:
            parts.append(attr_value)
    return " ".join(parts)


def has_document_signal(text: str) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in DOCUMENT_SIGNAL_TERMS)


def extract_candidates_from_html(html: str, source_url: str, company: CompanyConfig) -> list[CandidateLink]:
    soup = BeautifulSoup(html, "html.parser")
    candidates: list[CandidateLink] = []

    for anchor in soup.find_all("a", href=True):
        href = clean_text(str(anchor.get("href", "")))
        if not href:
            continue
        absolute_url = strip_tracking_params(urljoin(source_url, href))
        if absolute_url.startswith("mailto:") or absolute_url.startswith("javascript:"):
            continue
        if not is_allowed_domain(absolute_url, company.allowed_domains):
            continue
        signal_text = signal_from_anchor(anchor, absolute_url)
        if not has_document_signal(signal_text):
            continue

        context = context_from_anchor(anchor)
        score, report_type = score_candidate(absolute_url, signal_text, context, company)
        if score < 35:
            continue

        candidates.append(
            CandidateLink(
                company=company.name,
                source_page=source_url,
                url=absolute_url,
                context=context,
                score=score,
                report_type=report_type,
            )
        )

    raw_pdf_urls = sorted(
        set(
            re.findall(
                r"https?://[^\"' <>()]+?\.pdf(?:\?[^\"' <>()]*)?",
                html,
                flags=re.IGNORECASE,
            )
        )
    )
    for raw_url in raw_pdf_urls:
        normalized_url = strip_tracking_params(raw_url)
        if not is_allowed_domain(normalized_url, company.allowed_domains):
            continue
        signal_text = Path(urlparse(normalized_url).path).name
        if not has_document_signal(signal_text):
            continue
        score, report_type = score_candidate(normalized_url, signal_text, normalized_url, company)
        if score < 35:
            continue
        candidates.append(
            CandidateLink(
                company=company.name,
                source_page=source_url,
                url=normalized_url,
                context=normalized_url,
                score=score,
                report_type=report_type,
            )
        )

    deduped: dict[str, CandidateLink] = {}
    for candidate in candidates:
        previous = deduped.get(candidate.url)
        if previous is None or candidate.score > previous.score:
            deduped[candidate.url] = candidate

    return sorted(deduped.values(), key=lambda item: (-item.score, item.url))


def fetch_html(session: Session, url: str, timeout: int) -> str:
    response = session.get(url, timeout=timeout)
    response.raise_for_status()
    return response.text


def prioritize_candidates(candidates: list[CandidateLink], limit: int) -> list[CandidateLink]:
    if limit <= 0:
        return []

    ranked = sorted(
        candidates,
        key=lambda item: (
            item.score + REPORT_PRIORITY_BONUS.get(item.report_type, 0),
            candidate_year(item),
            item.score,
        ),
        reverse=True,
    )

    selected: list[CandidateLink] = []
    seen_urls: set[str] = set()
    seen_types: set[str] = set()

    for candidate in ranked:
        if candidate.url in seen_urls or candidate.report_type in seen_types:
            continue
        selected.append(candidate)
        seen_urls.add(candidate.url)
        seen_types.add(candidate.report_type)
        if len(selected) >= limit:
            return selected

    for candidate in ranked:
        if candidate.url in seen_urls:
            continue
        selected.append(candidate)
        seen_urls.add(candidate.url)
        if len(selected) >= limit:
            break

    return selected


def make_filename(company_slug: str, candidate: CandidateLink, sha256: str, final_url: str) -> str:
    path_name = unquote(Path(urlparse(final_url).path).name)
    if path_name.lower().endswith(".pdf") and len(path_name) < 150:
        filename = path_name
    else:
        year = extract_document_year(candidate.url, candidate.context) or "unknown"
        filename = f"{company_slug}_{year}_{candidate.report_type}_{sha256[:10]}.pdf"

    if not filename.lower().endswith(".pdf"):
        filename = f"{filename}.pdf"

    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", filename).strip("._")
    if not cleaned:
        cleaned = f"{company_slug}_{sha256[:10]}.pdf"
    if not cleaned.lower().startswith(f"{company_slug}_"):
        cleaned = f"{company_slug}_{cleaned}"
    return cleaned


def looks_like_pdf(response: Response, content: bytes) -> bool:
    content_type = response.headers.get("Content-Type", "").lower()
    if "application/pdf" in content_type:
        return True
    return content.startswith(PDF_SIGNATURE)


def download_candidate(
    session: Session,
    candidate: CandidateLink,
    company_slug: str,
    save_dir: Path,
    timeout: int,
) -> dict[str, str | int]:
    response = session.get(candidate.url, timeout=timeout)
    response.raise_for_status()

    content = response.content
    if not looks_like_pdf(response, content):
        raise ValueError(
            f"URL did not return a PDF (content-type={response.headers.get('Content-Type', 'unknown')})"
        )

    sha256 = hashlib.sha256(content).hexdigest()
    filename = make_filename(company_slug, candidate, sha256, response.url)
    company_dir = save_dir / company_slug
    company_dir.mkdir(parents=True, exist_ok=True)
    file_path = company_dir / filename

    if not file_path.exists():
        file_path.write_bytes(content)

    return {
        "company": candidate.company,
        "report_type": candidate.report_type,
        "year": extract_document_year(candidate.url, candidate.context) or "",
        "score": candidate.score,
        "source_page": candidate.source_page,
        "candidate_url": candidate.url,
        "final_url": response.url,
        "filename": filename,
        "relative_path": str(file_path.relative_to(save_dir.parent)),
        "sha256": sha256,
        "bytes": len(content),
        "context": candidate.context,
    }


def write_metadata(rows: list[dict[str, str | int]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "company",
        "report_type",
        "year",
        "score",
        "source_page",
        "candidate_url",
        "final_url",
        "filename",
        "relative_path",
        "sha256",
        "bytes",
        "context",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def discover_company_candidates(
    session: Session,
    robots: RobotsCache,
    company: CompanyConfig,
    timeout: int,
    delay_seconds: float,
) -> list[CandidateLink]:
    all_candidates: dict[str, CandidateLink] = {}

    for source_url in company.start_urls:
        robots_status = robots.can_fetch(source_url)
        if robots_status is False:
            logging.warning("Skipping %s because robots.txt disallows it", source_url)
            continue

        logging.info("Scanning %s", source_url)
        try:
            html = fetch_html(session, source_url, timeout)
        except requests.RequestException as exc:
            logging.warning("Source page failed for %s: %s", source_url, exc)
            continue

        candidates = extract_candidates_from_html(html, source_url, company)
        logging.info("Found %s relevant candidates on %s", len(candidates), source_url)

        for candidate in candidates:
            existing = all_candidates.get(candidate.url)
            if existing is None or candidate.score > existing.score:
                all_candidates[candidate.url] = candidate

        time.sleep(delay_seconds)

    return sorted(all_candidates.values(), key=lambda item: (-item.score, item.url))


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Robust ESG report scraper for the FTD ESG project.")
    parser.add_argument(
        "--companies",
        nargs="+",
        choices=sorted(COMPANIES.keys()),
        default=list(COMPANIES.keys()),
        help="Companies to scrape.",
    )
    parser.add_argument(
        "--max-per-company",
        type=int,
        default=1,
        help="Maximum number of PDFs to download per company.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path("data/raw_pdfs"),
        help="Directory where PDFs will be stored.",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=Path("data/pdf_metadata.csv"),
        help="CSV file used to store download metadata.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only discover candidates without downloading files.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help="HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_DELAY_SECONDS,
        help="Delay between requests.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    session = build_session()
    robots = RobotsCache(session)
    metadata_rows: list[dict[str, str | int]] = []

    for company_key in args.companies:
        company = COMPANIES[company_key]
        company_slug = slugify(company.name)

        try:
            candidates = discover_company_candidates(
                session=session,
                robots=robots,
                company=company,
                timeout=args.timeout,
                delay_seconds=args.delay,
            )
        except requests.RequestException as exc:
            logging.error("Discovery failed for %s: %s", company.name, exc)
            continue

        if not candidates:
            logging.warning("No relevant candidates found for %s", company.name)
            continue

        preview_limit = max(args.max_per_company * 3, args.max_per_company)
        prioritized = prioritize_candidates(candidates, preview_limit)
        logging.info("Prioritized candidates for %s:", company.name)
        for candidate in prioritized[:5]:
            logging.info(
                "  year=%s score=%s type=%s url=%s",
                candidate_year(candidate) or "n/a",
                candidate.score,
                candidate.report_type,
                candidate.url,
            )

        if args.dry_run:
            continue

        downloaded_hashes: set[str] = set()
        downloaded_for_company = 0

        for candidate in prioritized:
            if downloaded_for_company >= args.max_per_company:
                break

            robots_status = robots.can_fetch(candidate.url)
            if robots_status is False:
                logging.warning("Skipping %s because robots.txt disallows it", candidate.url)
                continue

            try:
                row = download_candidate(
                    session=session,
                    candidate=candidate,
                    company_slug=company_slug,
                    save_dir=args.save_dir,
                    timeout=args.timeout,
                )
            except (requests.RequestException, ValueError) as exc:
                logging.warning("Skipping candidate %s: %s", candidate.url, exc)
                continue

            sha256 = str(row["sha256"])
            if sha256 in downloaded_hashes:
                logging.info("Duplicate ignored for %s: %s", company.name, row["filename"])
                continue

            downloaded_hashes.add(sha256)
            metadata_rows.append(row)
            downloaded_for_company += 1
            logging.info("Downloaded %s -> %s", company.name, row["filename"])
            time.sleep(args.delay)

    if not args.dry_run and metadata_rows:
        write_metadata(metadata_rows, args.metadata_path)
        logging.info("Metadata written to %s", args.metadata_path)
    elif not args.dry_run:
        logging.warning("No documents were downloaded, metadata file was not updated.")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

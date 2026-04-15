#!/usr/bin/env python3
"""Scrape Metacritic critic reviews for Best Picture nominees.

This script intentionally focuses on raw collection only. It exports one row
per critic review so later preprocessing can handle date windows, text cleanup,
and normalization.
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator, Sequence
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup


BASE_URL = "https://www.metacritic.com/movie/"
BACKEND_BASE_URL = "https://backend.metacritic.com/reviews/metacritic/critic/movies/"
DATE_PATTERN = re.compile(r"^[A-Z][a-z]{2} \d{1,2}, \d{4}$")
SCORE_PUBLICATION_PATTERN = re.compile(r"^(?P<score>\d{1,3}|tbd)\s+(?P<publication>.+)$")
SCORE_ONLY_PATTERN = re.compile(r"^(?:\d{1,3}|tbd)$")
NOISE_LINES = {
    "All Reviews",
    "All Reviews Positive Reviews Mixed Reviews Negative Reviews",
    "Positive Reviews",
    "Mixed Reviews",
    "Negative Reviews",
    "Metascore",
    "Metascore Recently Added Publication (A-Z)",
    "Recently Added",
    "Publication (A-Z)",
    "Score",
    "Score Recently Added",
    "Select",
}
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/135.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}


@dataclass
class ReviewRecord:
    ceremony_year: int
    release_year: int
    film_title: str
    winner: int
    metacritic_slug: str
    metacritic_url: str
    critic_review_page: str
    review_date: str
    critic_score: str
    publication: str
    author: str
    quote: str
    full_review_url: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/nominees.csv"),
        help="CSV containing ceremony_year, film_title, release_year, winner, metacritic_slug",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/metacritic_reviews.csv"),
        help="Where to write the flattened review dataset",
    )
    parser.add_argument(
        "--failures-output",
        type=Path,
        default=Path("data/raw/metacritic_failures.csv"),
        help="Where to write films that could not be scraped",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for quick smoke tests",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=1.5,
        help="Delay between film requests to reduce rate-limit risk",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Per-request timeout in seconds",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def slugify_title(title: str) -> str:
    normalized = title.lower()
    normalized = normalized.replace("&", "and")
    normalized = normalized.replace("'", "")
    normalized = normalized.replace(":", "")
    normalized = normalized.replace(",", "")
    normalized = normalized.replace(".", "")
    normalized = normalized.replace("(", "")
    normalized = normalized.replace(")", "")
    normalized = re.sub(r"\s+", "-", normalized.strip())
    normalized = re.sub(r"-{2,}", "-", normalized)
    return normalized


def build_critic_review_url(slug: str) -> str:
    return urljoin(BASE_URL, f"{slug}/critic-reviews/")


def build_backend_review_url(slug: str, offset: int) -> str:
    return (
        f"{BACKEND_BASE_URL}{slug}/web?"
        f"offset={offset}&limit=10&filterBySentiment=all&sort=date"
        "&componentName=critic-reviews"
        "&componentDisplayName=critic+Reviews"
        "&componentType=ReviewList"
    )


def request_html(session: requests.Session, url: str, timeout: float) -> str:
    response = session.get(url, timeout=timeout)
    response.raise_for_status()
    return response.text


def request_json(session: requests.Session, url: str, timeout: float) -> dict:
    response = session.get(url, timeout=timeout)
    response.raise_for_status()
    return response.json()


def extract_full_review_links(soup: BeautifulSoup, critic_page_url: str) -> list[str]:
    links: list[str] = []
    for anchor in soup.find_all("a", href=True):
        text = anchor.get_text(" ", strip=True)
        href = anchor["href"].strip()
        if "FULL REVIEW" not in text.upper():
            continue
        if not href:
            continue
        links.append(urljoin(critic_page_url, href))
    return links


def cleaned_strings(soup: BeautifulSoup) -> list[str]:
    values: list[str] = []
    for entry in soup.stripped_strings:
        text = entry.strip()
        if not text:
            continue
        if text == "Advertisement":
            continue
        values.append(text)
    return values


def slice_review_section(lines: Sequence[str]) -> list[str]:
    start_index = None
    end_index = None

    for index, value in enumerate(lines):
        if value.startswith("Showing ") and value.endswith("Critic Reviews"):
            start_index = index + 1
            break

    if start_index is None:
        raise ValueError("Could not find the start of the critic review section")

    for index in range(start_index, len(lines)):
        if lines[index] in {"## Overview", "Overview", "## Details", "Details"}:
            end_index = index
            break

    if end_index is None:
        end_index = len(lines)

    return list(lines[start_index:end_index])


def normalize_date(raw_value: str) -> str:
    try:
        return datetime.strptime(raw_value, "%b %d, %Y").date().isoformat()
    except ValueError:
        return raw_value


def is_review_start(value: str) -> bool:
    return bool(DATE_PATTERN.match(value) or SCORE_PUBLICATION_PATTERN.match(value) or SCORE_ONLY_PATTERN.match(value))


def parse_score_and_publication(review_lines: Sequence[str], index: int) -> tuple[str, str, int] | None:
    if index >= len(review_lines):
        return None

    current = review_lines[index]
    combined_match = SCORE_PUBLICATION_PATTERN.match(current)
    if combined_match:
        return (
            combined_match.group("score"),
            combined_match.group("publication"),
            index + 1,
        )

    if SCORE_ONLY_PATTERN.match(current):
        next_index = index + 1
        while next_index < len(review_lines) and review_lines[next_index] in NOISE_LINES:
            next_index += 1
        if next_index < len(review_lines):
            return current, review_lines[next_index], next_index + 1

    return None


def parse_review_lines(review_lines: Sequence[str]) -> list[dict[str, str]]:
    parsed: list[dict[str, str]] = []
    index = 0

    while index < len(review_lines):
        current = review_lines[index]

        if current in NOISE_LINES:
            index += 1
            continue

        review_date = ""
        if DATE_PATTERN.match(current):
            review_date = normalize_date(current)
            index += 1
            while index < len(review_lines) and review_lines[index] in NOISE_LINES:
                index += 1
            if index >= len(review_lines):
                break
            current = review_lines[index]

        if not is_review_start(current):
            index += 1
            continue

        score_publication = parse_score_and_publication(review_lines, index)
        if score_publication is None:
            logging.debug("Skipping malformed review start at line: %s", current)
            index += 1
            continue

        critic_score, publication, index = score_publication

        quote_parts: list[str] = []
        author = ""

        while index < len(review_lines):
            value = review_lines[index]

            if value in NOISE_LINES:
                index += 1
                continue

            if is_review_start(value):
                break

            if value.startswith("By "):
                author = value.removeprefix("By ").strip()
                index += 1
                if index < len(review_lines) and "FULL REVIEW" in review_lines[index].upper():
                    index += 1
                if index < len(review_lines) and review_lines[index] == "open-full-review":
                    index += 1
                break

            if "FULL REVIEW" in value.upper():
                index += 1
                if index < len(review_lines) and review_lines[index] == "open-full-review":
                    index += 1
                break

            if value == "open-full-review":
                index += 1
                break

            quote_parts.append(value)
            index += 1

        parsed.append(
            {
                "review_date": review_date,
                "critic_score": critic_score,
                "publication": publication,
                "author": author,
                "quote": " ".join(quote_parts).strip(),
            }
        )

    return parsed


def scrape_review_page(
    session: requests.Session,
    critic_page_url: str,
    timeout: float,
) -> list[dict[str, str]]:
    html = request_html(session, critic_page_url, timeout=timeout)
    soup = BeautifulSoup(html, "html.parser")
    lines = cleaned_strings(soup)
    review_lines = slice_review_section(lines)
    reviews = parse_review_lines(review_lines)
    full_review_links = extract_full_review_links(soup, critic_page_url)

    for review, link in zip(reviews, full_review_links):
        review["full_review_url"] = link

    for review in reviews:
        review.setdefault("full_review_url", "")

    if not reviews:
        raise ValueError("No critic reviews were parsed from the page")

    return reviews


def normalize_api_date(raw_value: str | None) -> str:
    if not raw_value:
        return ""

    try:
        return datetime.fromisoformat(raw_value.replace("Z", "+00:00")).date().isoformat()
    except ValueError:
        return raw_value


def scrape_review_api(
    session: requests.Session,
    slug: str,
    timeout: float,
) -> list[dict[str, str]]:
    reviews: list[dict[str, str]] = []
    offset = 0
    total_results = None

    while total_results is None or offset < total_results:
        payload = request_json(session, build_backend_review_url(slug, offset), timeout=timeout)
        data = payload.get("data", {})
        items = data.get("items", [])

        if total_results is None:
            total_results = int(data.get("totalResults", 0))

        if not items:
            break

        for item in items:
            reviews.append(
                {
                    "review_date": normalize_api_date(item.get("date")),
                    "critic_score": str(item.get("score", "")),
                    "publication": item.get("publicationName", "") or "",
                    "author": item.get("author", "") or "",
                    "quote": item.get("quote", "") or "",
                    "full_review_url": item.get("url", "") or "",
                }
            )

        offset += len(items)

    if not reviews:
        raise ValueError("No critic reviews were returned by the backend endpoint")

    return reviews


def iter_review_records(
    manifest_rows: Iterable[dict[str, str]],
    session: requests.Session,
    timeout: float,
    sleep_seconds: float,
) -> tuple[list[ReviewRecord], list[dict[str, str]]]:
    records: list[ReviewRecord] = []
    failures: list[dict[str, str]] = []

    for row in manifest_rows:
        slug = (row.get("metacritic_slug") or "").strip() or slugify_title(row["film_title"])
        critic_page_url = build_critic_review_url(slug)
        logging.info("Scraping %s (%s)", row["film_title"], critic_page_url)

        try:
            reviews = scrape_review_api(session, slug, timeout=timeout)
        except Exception as exc:  # noqa: BLE001
            logging.warning("Backend scrape failed for %s: %s", row["film_title"], exc)
            try:
                reviews = scrape_review_page(session, critic_page_url, timeout=timeout)
            except Exception as fallback_exc:  # noqa: BLE001
                logging.warning("Fallback HTML scrape failed for %s: %s", row["film_title"], fallback_exc)
                failures.append(
                    {
                        "ceremony_year": row["ceremony_year"],
                        "film_title": row["film_title"],
                        "metacritic_slug": slug,
                        "critic_review_page": critic_page_url,
                        "error": f"backend: {exc}; fallback: {fallback_exc}",
                    }
                )
                time.sleep(sleep_seconds)
                continue

        for review in reviews:
            records.append(
                ReviewRecord(
                    ceremony_year=int(row["ceremony_year"]),
                    release_year=int(row["release_year"]),
                    film_title=row["film_title"],
                    winner=int(row["winner"]),
                    metacritic_slug=slug,
                    metacritic_url=urljoin(BASE_URL, f"{slug}/"),
                    critic_review_page=critic_page_url,
                    review_date=review["review_date"],
                    critic_score=review["critic_score"],
                    publication=review["publication"],
                    author=review["author"],
                    quote=review["quote"],
                    full_review_url=review["full_review_url"],
                )
            )

        time.sleep(sleep_seconds)

    return records, failures


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_reviews(path: Path, records: Sequence[ReviewRecord]) -> None:
    ensure_parent(path)
    fieldnames = list(asdict(records[0]).keys()) if records else list(ReviewRecord.__dataclass_fields__.keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def write_failures(path: Path, failures: Sequence[dict[str, str]]) -> None:
    ensure_parent(path)
    fieldnames = ["ceremony_year", "film_title", "metacritic_slug", "critic_review_page", "error"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for failure in failures:
            writer.writerow(failure)


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)

    manifest_rows = load_manifest(args.manifest)
    if args.limit is not None:
        manifest_rows = manifest_rows[: args.limit]

    session = requests.Session()
    session.headers.update(DEFAULT_HEADERS)

    records, failures = iter_review_records(
        manifest_rows=manifest_rows,
        session=session,
        timeout=args.timeout,
        sleep_seconds=args.sleep_seconds,
    )

    if records:
        write_reviews(args.output, records)
        logging.info("Wrote %s review rows to %s", len(records), args.output)
    else:
        logging.warning("No review rows were written")

    write_failures(args.failures_output, failures)
    logging.info("Logged %s scrape failures to %s", len(failures), args.failures_output)

    return 0 if records else 1


if __name__ == "__main__":
    sys.exit(main())

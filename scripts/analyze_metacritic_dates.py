#!/usr/bin/env python3
"""Summarize Metacritic review-date coverage."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reviews",
        type=Path,
        default=Path("data/raw/metacritic_reviews.csv"),
        help="Scraped Metacritic review CSV",
    )
    parser.add_argument(
        "--top-missing-films",
        type=int,
        default=20,
        help="Number of films with highest missing-date counts to print",
    )
    return parser.parse_args()


def pct(part: int, total: int) -> float:
    return round((part / total) * 100, 2) if total else 0.0


def print_group_summary(rows: list[dict[str, str]], key: str) -> None:
    totals = Counter(row[key] for row in rows)
    missing = Counter(row[key] for row in rows if not row["review_date"].strip())

    print(f"\nMissing review_date by {key}")
    print(f"{key},total_reviews,missing_reviews,missing_percent")
    for value in sorted(totals, key=lambda item: int(item) if item.isdigit() else item):
        print(f"{value},{totals[value]},{missing[value]},{pct(missing[value], totals[value])}")


def main() -> int:
    args = parse_args()

    with args.reviews.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    total = len(rows)
    missing_count = sum(not row["review_date"].strip() for row in rows)
    film_keys = {(row["ceremony_year"], row["film_title"]) for row in rows}
    missing_film_keys = {
        (row["ceremony_year"], row["film_title"])
        for row in rows
        if not row["review_date"].strip()
    }

    print("Metacritic review-date coverage")
    print(f"total_reviews,{total}")
    print(f"missing_review_dates,{missing_count}")
    print(f"missing_review_date_percent,{pct(missing_count, total)}")
    print(f"unique_films,{len(film_keys)}")
    print(f"films_with_any_missing_date,{len(missing_film_keys)}")
    print(f"films_with_any_missing_date_percent,{pct(len(missing_film_keys), len(film_keys))}")

    print_group_summary(rows, "release_year")
    print_group_summary(rows, "ceremony_year")

    by_film: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: {"total": 0, "missing": 0})
    for row in rows:
        key = (row["ceremony_year"], row["film_title"])
        by_film[key]["total"] += 1
        if not row["review_date"].strip():
            by_film[key]["missing"] += 1

    ranked = sorted(
        by_film.items(),
        key=lambda item: (-item[1]["missing"], -pct(item[1]["missing"], item[1]["total"]), item[0]),
    )

    print("\nFilms with missing review_date")
    print("ceremony_year,film_title,total_reviews,missing_reviews,missing_percent")
    printed = 0
    for (ceremony_year, film_title), counts in ranked:
        if counts["missing"] == 0:
            continue
        print(
            f"{ceremony_year},{film_title},{counts['total']},"
            f"{counts['missing']},{pct(counts['missing'], counts['total'])}"
        )
        printed += 1
        if printed >= args.top_missing_films:
            break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

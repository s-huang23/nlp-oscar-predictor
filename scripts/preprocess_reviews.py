#!/usr/bin/env python3
"""Preprocess Oscar nominee review datasets.

The core window is:
    review_date < ceremony_date

The ceremony date is excluded because the scraped data is date-only. A review
on Oscar day could be published after the Best Picture winner is announced.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


DEFAULT_START_YEAR = 2012
DEFAULT_END_YEAR = 2020
MIN_TEXT_CHARS = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        choices=["metacritic", "imdb"],
        default="metacritic",
        help="Input schema to preprocess",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/metacritic_reviews.csv"),
        help="Raw review CSV",
    )
    parser.add_argument(
        "--windows",
        type=Path,
        default=Path("data/oscar_windows.csv"),
        help="CSV with ceremony_year, nomination_date, ceremony_date",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/metacritic_reviews_2012_2020_window.csv"),
        help="Processed review CSV",
    )
    parser.add_argument("--start-year", type=int, default=DEFAULT_START_YEAR)
    parser.add_argument("--end-year", type=int, default=DEFAULT_END_YEAR)
    parser.add_argument(
        "--max-reviews-per-film",
        type=int,
        default=None,
        help="Optional deterministic cap after window filtering",
    )
    return parser.parse_args()


def clean_text(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_review_dates(df: pd.DataFrame, source: str) -> pd.Series:
    if source == "metacritic":
        return pd.to_datetime(df["review_date"], errors="coerce")

    return pd.to_datetime(df["review_date"], errors="coerce", dayfirst=False)


def text_column_for_source(source: str) -> str:
    return "quote" if source == "metacritic" else "review_detail"


def filter_by_ceremony_year(df: pd.DataFrame, start_year: int, end_year: int) -> pd.DataFrame:
    filtered = df[(df["ceremony_year"] >= start_year) & (df["ceremony_year"] <= end_year)].copy()
    print_summary("After ceremony-year filter", filtered)
    return filtered


def filter_by_awards_window(df: pd.DataFrame, windows: pd.DataFrame, source: str) -> pd.DataFrame:
    working = df.merge(windows, on="ceremony_year", how="left", validate="many_to_one")
    working["review_date_parsed"] = parse_review_dates(working, source)

    before_missing = working["review_date_parsed"].isna().sum()
    if before_missing:
        print(f"Rows dropped because review_date could not be parsed: {before_missing}")

    in_window = (
        working["review_date_parsed"].notna()
        & (working["review_date_parsed"] < working["ceremony_date"])
    )
    filtered = working[in_window].copy()
    print_summary("After ceremony-date filter", filtered)
    return filtered


def remove_noise_and_duplicates(df: pd.DataFrame, source: str) -> pd.DataFrame:
    text_col = text_column_for_source(source)
    working = df.copy()
    working["clean_text"] = working[text_col].map(clean_text)

    before = len(working)
    working = working[working["clean_text"].str.len() >= MIN_TEXT_CHARS].copy()
    print(f"Rows dropped for short/empty text: {before - len(working)}")

    dedupe_cols = ["ceremony_year", "film_title", "clean_text"]
    if source == "imdb" and "reviewer" in working.columns:
        dedupe_cols.insert(2, "reviewer")
    if source == "metacritic" and "publication" in working.columns:
        dedupe_cols.insert(2, "publication")

    before = len(working)
    working = working.drop_duplicates(subset=dedupe_cols).copy()
    print(f"Rows dropped as duplicates: {before - len(working)}")
    print_summary("After noise and duplicate removal", working)
    return working


def add_volume_normalization(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()
    group_cols = ["ceremony_year", "film_title"]
    working["film_review_count"] = working.groupby(group_cols)["clean_text"].transform("size")
    working["film_review_weight"] = 1.0 / working["film_review_count"]
    return working


def cap_reviews_per_film(df: pd.DataFrame, max_reviews_per_film: int | None) -> pd.DataFrame:
    if max_reviews_per_film is None:
        return df

    sort_cols = ["ceremony_year", "film_title", "review_date_parsed"]
    if "critic_score" in df.columns:
        sort_cols.append("critic_score")

    capped = (
        df.sort_values(sort_cols)
        .groupby(["ceremony_year", "film_title"], group_keys=False)
        .head(max_reviews_per_film)
        .copy()
    )
    print_summary(f"After cap of {max_reviews_per_film} reviews per film", capped)
    return add_volume_normalization(capped)


def print_summary(label: str, df: pd.DataFrame) -> None:
    print(f"\n{label}")
    print(f"Rows         : {len(df)}")
    print(f"Unique films : {df[['ceremony_year', 'film_title']].drop_duplicates().shape[0]}")
    if len(df):
        years = sorted(df["ceremony_year"].dropna().astype(int).unique().tolist())
        print(f"Years        : {years}")


def main() -> int:
    args = parse_args()

    reviews = pd.read_csv(args.input)
    windows = pd.read_csv(args.windows, parse_dates=["nomination_date", "ceremony_date"])

    reviews["ceremony_year"] = reviews["ceremony_year"].astype(int)
    windows["ceremony_year"] = windows["ceremony_year"].astype(int)

    processed = filter_by_ceremony_year(reviews, args.start_year, args.end_year)
    processed = filter_by_awards_window(processed, windows, args.source)
    processed = remove_noise_and_duplicates(processed, args.source)
    processed = add_volume_normalization(processed)
    processed = cap_reviews_per_film(processed, args.max_reviews_per_film)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(args.output, index=False)
    print(f"\nWrote {len(processed)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

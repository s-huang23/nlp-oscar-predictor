# nlp-oscar-predictor

## Overview
A transformer-based model to estimate the probability that an Oscar-nominated film wins Best Picture by analyzing textual discourse surrounding each nominee. For each nominee within its annual pool, the model learns linguistic signals of prestige, sentiment, and narrative momentum and outputs a win probability ∈ [0, 1] via a sigmoid activation.

## Dataset
The dataset combines Metacritic critic reviews and Twitter discourse during the two-month window between Oscar nominations (mid-January) and the awards ceremony (mid-March). The current Metacritic manifest covers Best Picture nominees for film release years 2009–2023, which map to Oscar ceremony years 2010–2024.

Metacritic collection is now set up as a standalone raw-data step:

- Nominee manifest: `data/nominees.csv`
- Scraper: `scripts/scrape_metacritic.py`
- Output: `data/raw/metacritic_reviews.csv`
- Failures log: `data/raw/metacritic_failures.csv`
- Oscar campaign windows: `data/oscar_windows.csv`

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the scraper:

```bash
python scripts/scrape_metacritic.py
```

Run a small smoke test first:

```bash
python scripts/scrape_metacritic.py --limit 3 --verbose
```

Analyze review-date coverage:

```bash
python scripts/analyze_metacritic_dates.py
```

Preprocess reviews for the 2012-2020 Oscar campaign windows:

```bash
python scripts/preprocess_reviews.py
```

The campaign window is `nomination_date <= review_date < ceremony_date`.

## Model
Pipeline:
BERT  → [CLS] per review  → WeightedAggregator → review film vector <br>
BERTweet → [CLS] per tweet → WeightedAggregator → tweet film vector <br>
Concatenate → Classification Head → P(win)

Weighted Aggregator: aggregates review embeddings from BERT and BERTweet into a single film vector

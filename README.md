# nlp-oscar-predictor

## Overview
A transformer-based model to estimate the probability that an Oscar-nominated film wins Best Picture by analyzing textual discourse surrounding each nominee. For each nominee within its annual pool, the model learns linguistic signals of prestige, sentiment, and narrative momentum and outputs a win probability ∈ [0, 1] via a sigmoid activation.

## Dataset
The dataset combines Metacritic critic reviews and iMDb reviews of Oscar Best Picture nominees and winners between 2012-2020. 

Metacritic collection is now set up as a standalone raw-data step:

- Nominee manifest: `data/nominees.csv`
- Scraper: `scripts/scrape_metacritic.py`
- Output: `data/raw/metacritic_reviews.csv`
- Failures log: `data/raw/metacritic_failures.csv`
- Oscar campaign windows: `data/oscar_windows.csv`

## Quick Start

### Google Collab (Recommended)
1. Click [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/s-huang23/nlp-oscar-predictor/blob/main/oscar_predictor.ipynb) or open [oscar_predictor.ipynb](oscar_predictor.ipynb) in GitHub, then click **Open in Colab** (or go to `File → Open notebook → GitHub` in Colab and paste the repo URL).
2. In Colab, go to **Runtime → Run all**.
   - The notebook clones this repo automatically so `data/` is available — no Drive upload needed.
   - The IMDb dataset is fetched via `kagglehub` on first run (uses Colab's cache on subsequent runs).
3. If you want to use a Kaggle API key for faster downloads, add it as a Colab secret named `KAGGLE_KEY` before running.

**GPU recommended.** In Colab, go to **Runtime → Change runtime type → T4 GPU** before running the BERT cells.

### Run Locally
1. **Clone the repo:**
   - Go to the repository on GitHub.
   - Click the green **Code** button → **SSH** → copy the URL.
   - In your terminal:
     ```bash
     git clone <paste SSH URL here>
     ```
     ```bash
     cd nlp-oscar-predictor
     ```
     
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118  # GPU (CUDA 11.8)
   # or for CPU only:
   pip install torch torchvision
   pip install transformers pandas numpy scikit-learn scipy kagglehub jupyter
   ```

3. **Set up Kaggle credentials** (required for the IMDb dataset):
   - Go to [kaggle.com/settings](https://www.kaggle.com/settings) → **API → Create New Token** — this downloads `kaggle.json`.
   - Place it at `~/.kaggle/kaggle.json` (Linux/Mac) or `%USERPROFILE%\.kaggle\kaggle.json` (Windows).

4. **Launch the notebook:**
   ```bash
   jupyter notebook oscar_predictor.ipynb
   ```
   - All data files (`nominees.csv`, `metacritic_reviews.csv`, `oscar_windows.csv`) are already in `data/` — no manual uploads needed.
   - Run all cells top to bottom. The IMDb dataset (~1 GB) will be downloaded automatically on first run via `kagglehub`.


## Run the scraper

```bash
python scripts/scrape_metacritic.py
```

## Analyze review-date coverage

```bash
python scripts/analyze_metacritic_dates.py
```

## Preprocess reviews before each 2012-2020 Oscar ceremony

```bash
python scripts/preprocess_reviews.py
```

## Model
Pipeline:
BERT  → [CLS] per review  → WeightedAggregator → review film vector <br>
BERTweet → [CLS] per tweet → WeightedAggregator → tweet film vector <br>
Concatenate → Classification Head → P(win)

Weighted Aggregator: aggregates review embeddings from BERT and BERTweet into a single film vector

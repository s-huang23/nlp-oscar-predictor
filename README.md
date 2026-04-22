# nlp-oscar-predictor

## Note
For the final cleaned-up model, see `final_oscar_predictor.ipynb`. For model development and experimentation, see `model_development.ipynb`.

## Overview
This repo documents the development and experimentation of a transformer-based model for predicting the Academy Award for Best Picture, and results are compared to 2 other NLP models for baseline analysis. All models operate on the same core premise: given a cohort of nominated films in a given year, can discourse captured in critic and audience reviews predict which film will win? For each nominee within its annual pool, the model learns linguistic signals of prestige, sentiment, and narrative momentum and outputs a win probability.

---

## Data Sources

Two review sources are used, covering ceremony years 2012–2020 (9 cohorts, 77–78 films total):

- **Metacritic** — professional critic reviews (~48 reviews per film). Short, technically precise quotes using film-critical vocabulary.
- **IMDb** — audience user reviews (~547 reviews per film). Longer, emotionally driven responses reflecting broader public discourse.

Nominees and winner labels come from a separate nominees table joined to both review sources by film title and ceremony year.

---

## Dataset
The dataset combines Metacritic critic reviews and iMDb reviews of Oscar Best Picture nominees and winners between 2012-2020. 

Metacritic collection is now set up as a standalone raw-data step:

- Nominee manifest: `data/nominees.csv`
- Scraper: `scripts/scrape_metacritic.py`
- Output: `data/raw/metacritic_reviews.csv`
- Failures log: `data/raw/metacritic_failures.csv`
- Oscar campaign windows: `data/oscar_windows.csv`

---

## Evaluation

All models are evaluated using **leave-one-year-out cross validation (LOOCV)** — train on 8 years, test on the held-out year, rotate through all 9 folds. This is the only statistically valid evaluation strategy given 9 cohort-level examples.

Three metrics are reported for each model:

- **Top-1 accuracy** — did the model predict the exact winner? (random baseline: ~12.5%)
- **Top-3 accuracy** — was the winner in the model's top 3 predictions? (random baseline: ~33.3%)
- **Mean Reciprocal Rank (MRR)** — average of 1/rank across folds, gives partial credit for near-misses (random baseline: ~0.31)

---

## Models

### Model 1 — Transformer with Cross-Attention
The primary model. Critic and IMDb reviews are encoded separately using a shared sentence transformer base (`all-mpnet-base-v2`), producing discriminative sentence-level embeddings. Each review stream passes through independent self-attention layers — allowing reviews within a stream to contextualize each other — followed by bidirectional cross-attention between the critic and IMDb streams. The cross-attention captures the relationship between critical consensus and audience discourse. Representations are fused using concatenation, difference, and element-wise product signals before a final MLP scorer.

**Key development decisions:**
- RoBERTa was the original encoder but produced near-identical embeddings across all films (cosine similarity ~0.997), making it impossible for downstream layers to distinguish nominees. Replaced with `all-mpnet-base-v2` which reduced pairwise similarity to ~0.70–0.80.
- Encoder frozen initially, then partially unfrozen — neither approach overcame the fundamental data constraint of 8 training examples per fold.
- Model capacity progressively reduced (heads: 8→4, layers: 2→1, feedforward: 4x→2x hidden dim) to mitigate overfitting.

---

### Model 2 — Simple MLP Baseline
An ablation baseline using the same sentence transformer embeddings as Model 1. Reviews are mean-pooled per stream with no attention layers. The two stream vectors are fused identically to Model 1 (concat + diff + product) and passed through a smaller MLP scorer.

**Purpose:** isolates the contribution of the attention architecture. Any performance gap between Model 1 and Model 2 is attributable specifically to self-attention and cross-attention, not to the embedding quality or fusion strategy.

---

### Model 3 — TF-IDF + Logistic Regression
A non-neural baseline. All critic reviews for a film are concatenated into one string; same for IMDb reviews. Separate TF-IDF vectorizers build vocabulary representations for each source, which are concatenated into one feature vector per film. A logistic regression classifier predicts P(winner).

**Purpose:** sanity check. If neural models cannot beat word-count statistics, something is wrong with either the neural pipeline or the data is too noisy for learned representations to add value. Also the most interpretable model — the top TF-IDF coefficients directly reveal which words the model associates with Best Picture winners.

**Key development decision:** C=0.01 (heavy regularization) collapsed all predictions to uniform probability. C=20 produced meaningful differentiation.

---

## Key Findings

The primary constraint throughout development is **dataset size** — 9 cohort-level training examples per fold. This creates a fundamental ceiling on what any supervised model can learn regardless of architecture complexity. The attention model overfits to training cohorts within 10–20 epochs and generalizes poorly to the test year. The TF-IDF model is the most robust to this constraint given its limited parameter count.

A secondary constraint is **data quality** — title mismatches between sources mean several actual winners have missing or partial review data in the current preprocessing pipeline, artificially suppressing model performance on those years. Results reported here should be interpreted with this caveat in mind. Cleaned data and an extended year range (incorporating BAFTA/Golden Globe cohorts) are identified as the highest-leverage improvements for future work.

---

## Quick Start

### Google Collab (Recommended)
1. Click [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/s-huang23/nlp-oscar-predictor/blob/main/final_oscar_predictor.ipynb) or open [final_oscar_predictor.ipynb](final_oscar_predictor.ipynb) in GitHub, then click **Open in Colab** (or go to `File → Open notebook → GitHub` in Colab and paste the repo URL).
2. In Colab, go to **Runtime → Run all**.
   - The notebook clones this repo automatically so `data/` is available — no Drive upload needed.
   - The IMDb dataset is fetched via `kagglehub` on first run (uses Colab's cache on subsequent runs).
3. If you want to use a Kaggle API key for faster downloads, add it as a Colab secret named `KAGGLE_KEY` before running.

**GPU recommended.** In Colab, go to **Runtime → Change runtime type → A100 GPU (or T4)** before running the code.

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

## Preprocess reviews for the 2012-2020 Oscar campaign windows

```bash
python scripts/preprocess_reviews.py
```

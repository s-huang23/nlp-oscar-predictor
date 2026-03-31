# nlp-oscar-predictor

## Overview
A transformer-based model to estimate the probability that an Oscar-nominated film wins Best Picture by analyzing textual discourse surrounding each nominee. For each nominee within its annual pool, the model learns linguistic signals of prestige, sentiment, and narrative momentum and outputs a win probability ∈ [0, 1] via a sigmoid activation.

## Dataset
The dataset combines Metacritic critic reviews and Twitter discourse during the two-month window between Oscar nominations (mid-January) and the awards ceremony (mid-March) across 10 years of Oscar races (2009–2019; 8-10 nominees/year). 

## Model
Pipeline:
BERT  → [CLS] per review  → WeightedAggregator → review film vector <br>
BERTweet → [CLS] per tweet → WeightedAggregator → tweet film vector <br>
Concatenate → Classification Head → P(win)

Weighted Aggregator: aggregates review embeddings from BERT and BERTweet into a single film vector

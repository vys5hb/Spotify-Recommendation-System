# Spotify Recommendation System

A two-tower retrieval model that continues Spotify playlists, built on the Spotify
Million Playlist Dataset (1M playlists, 66M playlist-track interactions, 33GB of raw
JSON). My goal was to beat two classic baselines (global popularity and track
co-occurrence) with a learned model, under an honest full-catalog evaluation. The
data pipeline is PySpark, the model is PyTorch, and training ran on a Kaggle T4.

## Results

All three models are scored the same way: rank every track in the 2.1M-track catalog
for each held-out playlist, then check the hidden tracks against the top K.

Validation split (50,110 playlists):

| model | Recall@10 | Recall@100 | NDCG@100 |
|---|---|---|---|
| two-tower (this repo) | 0.120 | 0.358 | 0.215 |
| co-occurrence baseline | 0.073 | 0.236 | 0.137 |
| popularity baseline | 0.008 | 0.044 | 0.020 |

On the untouched test split (49,747 playlists) the final model gets `Recall@100 = 0.359`
and `NDCG@100 = 0.215`, within 0.001 of validation, so the tuning didn't overfit.

How it got there:

- plain in-batch softmax actually **lost** to co-occurrence (Recall@100 = 0.182)
- adding a logQ sampling-bias correction brought it to 0.328
- adding a small MLP head on each tower brought it to 0.358

I also tried a deeper [256, 256] head and a bigger 8192 batch. Both were measured
against the final config and neither beat it, so neither is in it.

## How it works

- [`scripts/ingest_mpd.py`](scripts/ingest_mpd.py) — flattens the 33GB of nested MPD JSON into 3 Parquet tables (playlists, tracks, playlist_tracks) with PySpark
- [`scripts/build_splits.py`](scripts/build_splits.py) — deterministic 90/5/5 train/val/test split; for each val/test playlist it hides 20% of the tracks (capped at 10) as the targets to predict
- [`scripts/run_baselines.py`](scripts/run_baselines.py) — the popularity and co-occurrence baselines, plus the Recall/Precision/NDCG@K definitions everything else reuses
- [`scripts/build_vocab.py`](scripts/build_vocab.py) — maps 2.1M track / 283K artist / 705K album string IDs to integer indices, with index 0 reserved for padding and 1 for unknown IDs
- [`scripts/twotower/dataset.py`](scripts/twotower/dataset.py) — encodes the 60M-row train table once into a flat `.npz` cache; each epoch samples one held-out positive per playlist and uses the rest as context
- [`scripts/twotower/model.py`](scripts/twotower/model.py) — the model itself: both towers share track/artist/album embedding tables, the playlist tower mean-pools its context, and the loss is in-batch softmax with the logQ correction
- [`scripts/twotower/train.py`](scripts/twotower/train.py) / [`evaluate.py`](scripts/twotower/evaluate.py) — training loop and full-catalog evaluation

## Design choices

- **In-batch negatives with a logQ correction.** Each batch of 4096 positives doubles as 4095 free negatives per playlist, but that samples negatives by popularity, so the model over-penalizes popular tracks. Subtracting log(track frequency) from the logits (Yi et al. 2019) undoes the bias — this one change was worth +80% Recall@100.
- **Shared embedding tables.** A track has one vector whether it appears as playlist context or as the candidate being scored, so the two towers speak the same space.
- **PAD frozen at zero, UNK for cold-start.** Padding can never leak into the mean-pool, and tracks never seen in training still encode to a shared "unknown" vector instead of breaking retrieval.
- **A cache instead of Spark at train time.** Training reads one `.npz` of flat arrays with offsets, so epochs never touch Spark.
- **The eval matches the baselines exactly.** Same metric definitions, same masked targets, same catalog — the table above is apples to apples.
- **One hidden layer, not two.** I ablated the MLP head: [256] beat both the plain linear towers and [256, 256].

## Running it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Spark needs Java 17 (`export JAVA_HOME=$(/usr/libexec/java_home -v 17)` on macOS).
`data/bronze` expects the raw MPD JSON slices; everything downstream regenerates
from there (`data/` and `artifacts/` are gitignored).

```bash
python3 scripts/ingest_mpd.py --input data/bronze --output data/silver --overwrite
python3 scripts/build_splits.py --input data/silver --output data/gold --overwrite
python3 scripts/run_baselines.py --input data/gold --output artifacts/baselines/sample --overwrite
python3 scripts/build_vocab.py --input data/gold/train_playlist_tracks.parquet --output artifacts/vocab
python3 scripts/twotower/dataset.py --input data/gold/train_playlist_tracks.parquet
python3 scripts/twotower/train.py --dim 128 --hidden-dims 256 --batch-size 4096 --epochs 30 --temperature 0.10 --logq
python3 scripts/twotower/evaluate.py --checkpoint artifacts/twotower/checkpoints/best.pt --split validation
```

Tests: `python -m pytest tests/ -q` — 32 tests covering the vocab, dataset, model, training, and eval code.

## What I'd add next

- a FAISS/ANN index over the item vectors, so retrieval doesn't need a full 2.1M-track matmul
- playlist titles as a model feature (already ingested, not used yet)
- a second-stage ranker over the top few hundred retrieved candidates

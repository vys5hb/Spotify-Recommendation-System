#!/usr/bin/env python3
"""Evaluate a trained two-tower model with full-catalog retrieval.

For each eval playlist we encode its visible context with the playlist tower,
score that vector against EVERY track's item vector, take the top-K, and compare
to the hidden target tracks. Metrics are Recall/Precision/NDCG@K, computed with
the *same definitions* as run_baselines.py so the two-tower numbers sit directly
next to the popularity / co-occurrence baselines.

This is a pure torch + pandas script (no Spark): the item index is derived from
the train cache, and the gold val/test tables are read with pandas.

    python scripts/twotower/evaluate.py --checkpoint artifacts/twotower/checkpoints/best.pt --split validation
"""
import argparse
import glob
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_ROOT = SCRIPT_DIR.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))
from twotower.vocab import Vocabulary  # noqa: E402
from twotower.model import build_model_from_vocab_sizes, PAD_INDEX  # noqa: E402

ENTITIES = ("track", "artist", "album")
UNK_INDEX = Vocabulary.UNK_INDEX  # unseen / below-cutoff IDs encode here


def select_device(requested):
    if requested and requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(checkpoint_path, device):
    """Rebuild the model from a checkpoint's saved config and load its weights."""
    ck = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ck["config"]
    model = build_model_from_vocab_sizes(
        cfg["vocab_sizes"], embedding_dim=cfg["embedding_dim"], temperature=cfg["temperature"],
    )
    model.load_state_dict(ck["model_state_dict"])
    model = model.to(device).eval()   # eval() = inference mode (no dropout/bn updates)
    return model, cfg


def build_item_index(model, cache_path, device, encode_chunk=200_000):
    """Encode every track into a normalized item vector -> [num_tracks, D] on device.

    Each track needs its artist + album to embed. We read those straight from the
    train cache: for each track index, its artist/album is the one it appeared
    with (first occurrence). Rows 0 (PAD) and 1 (UNK) are included but will be
    masked out of retrieval — they're not recommendable.
    """
    data = np.load(cache_path)
    track_idx, artist_idx, album_idx = data["track_idx"], data["artist_idx"], data["album_idx"]
    num_tracks = int(track_idx.max()) + 1

    # First-occurrence artist/album per track index (np.unique gives first indices).
    track_to_artist = np.zeros(num_tracks, dtype=np.int64)
    track_to_album = np.zeros(num_tracks, dtype=np.int64)
    uniq, first = np.unique(track_idx, return_index=True)
    track_to_artist[uniq] = artist_idx[first]
    track_to_album[uniq] = album_idx[first]

    item_vecs = torch.empty(num_tracks, model.embedding_dim, device=device)
    with torch.no_grad():
        for start in range(0, num_tracks, encode_chunk):
            end = min(start + encode_chunk, num_tracks)
            tracks = torch.arange(start, end, device=device)
            artists = torch.from_numpy(track_to_artist[start:end]).to(device)
            albums = torch.from_numpy(track_to_album[start:end]).to(device)
            vecs = model.encode_item(tracks, artists, albums)
            item_vecs[start:end] = F.normalize(vecs, dim=-1)   # cosine -> pre-normalize
    return item_vecs


def _encode_column(series, id_to_index):
    """Map a column of string IDs to integer indices (unknown -> UNK)."""
    return series.map(id_to_index).fillna(UNK_INDEX).astype(np.int64).to_numpy()


def _read_parquet_path(path):
    """Read a parquet path that may be a single .parquet file OR a Spark-style
    directory of part files. Lets the gold tables be uploaded either way."""
    path = Path(path)
    if path.is_dir():
        files = sorted(glob.glob(str(path / "*.parquet")))
        if not files:
            raise FileNotFoundError(f"No .parquet part files found under {path}")
        return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    if not path.exists():
        raise FileNotFoundError(f"Parquet path does not exist: {path}")
    return pd.read_parquet(path)


def load_eval_playlists(gold_dir, split, vocabs, limit, seed):
    """Read the split's context + targets, encode, and group per playlist.

    Returns a list of dicts, one per playlist that has at least one target:
        context_track/artist/album : np.int64 arrays (the visible songs)
        target_indices             : set of target track indices (for hit checks)
        target_count               : number of DISTINCT target tracks (recall denom)
    Matches run_baselines: targets are deduplicated, context is excluded at scoring.
    """
    gold_dir = Path(gold_dir)
    ctx = _read_parquet_path(gold_dir / f"{split}_context.parquet")
    tgt = _read_parquet_path(gold_dir / f"{split}_targets.parquet")

    ctx = ctx[ctx["track_id"].notna()].sort_values(["pid", "pos"])
    ctx["t"] = _encode_column(ctx["track_id"], vocabs["track"].id_to_index)
    ctx["a"] = _encode_column(ctx["artist_id"], vocabs["artist"].id_to_index)
    ctx["al"] = _encode_column(ctx["album_id"], vocabs["album"].id_to_index)

    # Deduplicate targets by (pid, track_id) exactly like run_baselines.
    tgt = tgt[tgt["track_id"].notna()][["pid", "track_id"]].drop_duplicates()
    tgt["ti"] = _encode_column(tgt["track_id"], vocabs["track"].id_to_index)
    target_by_pid = {
        pid: {"indices": set(g["ti"].tolist()), "count": len(g)}
        for pid, g in tgt.groupby("pid")
    }

    # Optionally subsample playlists for a fast local run.
    pids = sorted(target_by_pid.keys())
    if limit and limit < len(pids):
        pids = list(np.random.default_rng(seed).choice(pids, size=limit, replace=False))
    keep = set(pids)

    playlists = []
    for pid, g in ctx.groupby("pid"):
        if pid not in keep:
            continue
        tgt_info = target_by_pid[pid]
        playlists.append({
            "pid": int(pid),
            "context_track": g["t"].to_numpy(),
            "context_artist": g["a"].to_numpy(),
            "context_album": g["al"].to_numpy(),
            "target_indices": tgt_info["indices"],
            "target_count": tgt_info["count"],
        })
    return playlists


def _pad_chunk(playlists, device):
    """Pad a chunk of playlists' contexts into [c, L] tensors + a mask."""
    c = len(playlists)
    max_len = max(len(p["context_track"]) for p in playlists)
    track = torch.full((c, max_len), PAD_INDEX, dtype=torch.long)
    artist = torch.full((c, max_len), PAD_INDEX, dtype=torch.long)
    album = torch.full((c, max_len), PAD_INDEX, dtype=torch.long)
    mask = torch.zeros((c, max_len), dtype=torch.bool)
    for i, p in enumerate(playlists):
        n = len(p["context_track"])
        track[i, :n] = torch.from_numpy(p["context_track"])
        artist[i, :n] = torch.from_numpy(p["context_artist"])
        album[i, :n] = torch.from_numpy(p["context_album"])
        mask[i, :n] = True
    return track.to(device), artist.to(device), album.to(device), mask.to(device)


def playlist_metrics(ranked_indices, target_indices, target_count, k_values):
    """Recall/Precision/NDCG at each K for one playlist.

    Definitions match run_baselines.compute_metrics exactly:
        recall    = hits / target_count
        precision = hits / K
        NDCG      = DCG / IDCG,  DCG = sum over hits of 1/log2(rank+1),
                    IDCG = sum_{rank=1..min(target_count,K)} 1/log2(rank+1)
    ranked_indices are the top track indices (rank 1 first).
    """
    out = {}
    for k in k_values:
        topk = ranked_indices[:k]
        hits, dcg = 0, 0.0
        for rank, idx in enumerate(topk, start=1):      # rank is 1-indexed
            if idx in target_indices:
                hits += 1
                dcg += 1.0 / math.log2(rank + 1)
        idcg = sum(1.0 / math.log2(r + 1) for r in range(1, min(target_count, k) + 1))
        out[k] = {
            "recall": hits / target_count,
            "precision": hits / k,
            "ndcg": (dcg / idcg) if idcg > 0 else 0.0,
        }
    return out


def evaluate(model, item_vecs, playlists, k_values, device, chunk_size):
    """Retrieve top-K for every playlist and average the metrics."""
    max_k = max(k_values)
    sums = {k: {"recall": 0.0, "precision": 0.0, "ndcg": 0.0} for k in k_values}
    n = 0

    with torch.no_grad():
        for start in range(0, len(playlists), chunk_size):
            chunk = playlists[start:start + chunk_size]
            track, artist, album, mask = _pad_chunk(chunk, device)

            playlist_vec = model.encode_playlist(track, artist, album, mask)   # [c, D]
            playlist_vec = F.normalize(playlist_vec, dim=-1)

            scores = playlist_vec @ item_vecs.t()          # [c, num_tracks] cosine
            scores[:, PAD_INDEX] = float("-inf")           # never recommend PAD
            scores[:, UNK_INDEX] = float("-inf")           # never recommend UNK
            # Exclude each playlist's own visible context tracks (like the baselines).
            for i, p in enumerate(chunk):
                seen = torch.from_numpy(p["context_track"]).to(device)
                scores[i, seen] = float("-inf")

            topk = scores.topk(max_k, dim=1).indices.cpu().numpy()   # [c, max_k] track indices

            for i, p in enumerate(chunk):
                m = playlist_metrics(topk[i], p["target_indices"], p["target_count"], k_values)
                for k in k_values:
                    for name in ("recall", "precision", "ndcg"):
                        sums[k][name] += m[k][name]
                n += 1

    metrics = {}
    for k in k_values:
        metrics[f"Recall@{k}"] = sums[k]["recall"] / n
        metrics[f"Precision@{k}"] = sums[k]["precision"] / n
        metrics[f"NDCG@{k}"] = sums[k]["ndcg"] / n
    return metrics, n


def load_baseline_metrics(path):
    """Read run_baselines' validation_metrics.json if present, for a side-by-side."""
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text()).get("baselines")


def print_comparison(twotower_metrics, baseline_metrics, k_values):
    """Print two-tower next to the baselines for each metric@K."""
    rows = [("twotower", twotower_metrics)]
    if baseline_metrics:
        for name in ("popularity", "cooccurrence"):
            if name in baseline_metrics:
                rows.append((name, baseline_metrics[name]))

    for metric in ("Recall", "Precision", "NDCG"):
        header = "  ".join(f"{metric}@{k:<4}" for k in k_values)
        print(f"\n{'model':<14} {header}")
        print("-" * (14 + len(header) + 2))
        for name, m in rows:
            cells = "  ".join(f"{m.get(f'{metric}@{k}', float('nan')):<9.4f}" for k in k_values)
            print(f"{name:<14} {cells}")


def parse_k_values(text):
    ks = sorted({int(v) for v in text.split(",") if v.strip()})
    if not ks:
        raise ValueError("At least one K value is required.")
    return ks


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the two-tower model with full-catalog retrieval.")
    parser.add_argument("--checkpoint", default="artifacts/twotower/checkpoints/best.pt", help="Trained checkpoint (best.pt).")
    parser.add_argument("--cache", default="artifacts/twotower/train_playlists.npz", help="Train cache (.npz) — used to build the item index.")
    parser.add_argument("--gold-dir", default="data/gold", help="Directory with the gold split parquet tables.")
    parser.add_argument("--vocab-dir", default="artifacts/vocab", help="Directory with the vocab JSONs.")
    parser.add_argument("--split", default="validation", choices=["validation", "test"], help="Which split to evaluate.")
    parser.add_argument("--k-values", default="10,50,100", help="Comma-separated K values.")
    parser.add_argument("--limit-playlists", type=int, default=0, help="Evaluate a random subset of N playlists (0 = all). For fast local runs.")
    parser.add_argument("--chunk-size", type=int, default=256, help="Playlists scored per retrieval chunk (memory knob).")
    parser.add_argument("--device", default="auto", help="cuda / mps / cpu, or auto.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the optional playlist subsample.")
    parser.add_argument("--baseline-metrics", default="artifacts/baselines/sample/validation_metrics.json", help="Baseline metrics JSON for side-by-side (optional).")
    parser.add_argument("--out", default="", help="Optional path to write the metrics JSON.")
    return parser.parse_args()


def main():
    args = parse_args()
    k_values = parse_k_values(args.k_values)
    device = select_device(args.device)
    print(f"device: {device}")

    vocabs = {e: Vocabulary.load(Path(args.vocab_dir) / f"{e}_vocab.json") for e in ENTITIES}

    model, cfg = load_model(args.checkpoint, device)
    print(f"loaded checkpoint (dim {cfg['embedding_dim']}, temperature {cfg['temperature']})")

    t0 = time.time()
    item_vecs = build_item_index(model, args.cache, device)
    print(f"item index: {item_vecs.shape[0]:,} tracks encoded ({time.time()-t0:.1f}s)")

    playlists = load_eval_playlists(args.gold_dir, args.split, vocabs, args.limit_playlists, args.seed)
    print(f"evaluating {len(playlists):,} {args.split} playlists")

    t0 = time.time()
    metrics, n = evaluate(model, item_vecs, playlists, k_values, device, args.chunk_size)
    print(f"retrieval done ({time.time()-t0:.1f}s)\n")

    baseline_metrics = load_baseline_metrics(args.baseline_metrics)
    print_comparison(metrics, baseline_metrics, k_values)

    print("\n" + json.dumps({"split": args.split, "playlists": n, "metrics": metrics}, indent=2, sort_keys=True))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"split": args.split, "playlists": n, "metrics": metrics}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

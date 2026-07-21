#!/usr/bin/env python3
"""PyTorch data pipeline for the two-tower retrieval model.

Two stages live here:

1. **Cache builder** (`build_cache` / the CLI). A one-time Spark job that reads
   the gold ``train_playlist_tracks`` table, encodes every track/artist/album ID
   through the saved ``Vocabulary``, groups the rows into per-playlist sequences,
   and writes a compact CSR-style ``.npz``. Encoding 59.7M rows on every epoch
   would be wasteful, so we pay that cost once here.

2. **Training-time dataset** (`PlaylistDataset` + `collate_playlists` +
   `make_dataloader`). Loads the ``.npz`` into memory and serves
   (playlist context, one positive track) pairs. In-batch negatives mean the
   other rows' positives act as negatives for free, so the dataset only ever
   emits positives.

Each example is a random leave-one-out split of a playlist: one track is sampled
as the positive, the rest (capped at ``max_context_len``) form the context. Over
many epochs a playlist is seen with many different positives.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from pyspark.sql import SparkSession, functions as F

# The Vocabulary class lives beside this file. When dataset.py is run as a
# script, scripts/ is on sys.path[0], so `from twotower.vocab import Vocabulary`
# resolves; we also add scripts/ explicitly for imports from elsewhere.
SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_ROOT = SCRIPT_DIR.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))
from twotower.vocab import Vocabulary  # noqa: E402  (import after sys.path setup)


# The three entities each playlist row carries. Kept in one place so the cache
# builder and the dataset agree on array names and ordering.
ENTITIES = ("track", "artist", "album")
ID_COLUMNS = {"track": "track_id", "artist": "artist_id", "album": "album_id"}

# PAD is index 0 in every vocab (see vocab.py). collate pads short contexts with
# it and the mask marks those slots as empty.
PAD_INDEX = Vocabulary.PAD_INDEX


# ----------------------------------------------------------------------------
# Stage 1: cache builder
# ----------------------------------------------------------------------------

# Mirrors the SparkSession config used across the other scripts.
def create_spark_session(app_name, master, driver_memory):
    return (
        SparkSession.builder.appName(app_name)
        .master(master)
        .config("spark.driver.memory", driver_memory)
        .config("spark.sql.shuffle.partitions", "64")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )


def load_vocabularies(vocab_dir):
    """Load the three saved vocabularies as {entity: Vocabulary}."""
    vocab_dir = Path(vocab_dir)
    vocabs = {}
    for entity in ENTITIES:
        path = vocab_dir / f"{entity}_vocab.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing vocab file: {path}. Run scripts/build_vocab.py first.")
        vocabs[entity] = Vocabulary.load(path)
    return vocabs


def build_cache(train_path, vocab_dir, output_path, min_playlist_length, spark):
    """Group train playlists into encoded per-playlist sequences and save a .npz.

    The output stores flat, CSR-style arrays so N playlists of varying length
    live in contiguous memory without any Python-level ragged structure:

        track_idx, artist_idx, album_idx : int32, length T (all occurrences)
        offsets                          : int64, length N + 1
        pids                             : int64, length N (for reference)

    Playlist i occupies the half-open slice [offsets[i], offsets[i+1]) in every
    ``*_idx`` array, ordered by the original ``pos`` within the playlist.

    Args:
        train_path: gold train_playlist_tracks Parquet path (train only).
        vocab_dir: directory holding {track,artist,album}_vocab.json.
        output_path: destination .npz path.
        min_playlist_length: drop playlists shorter than this. Must be >= 2 so a
            leave-one-out (context, positive) split is always possible.
        spark: an active SparkSession.
    """
    if min_playlist_length < 2:
        raise ValueError(
            f"min_playlist_length must be >= 2 to form a (context, positive) split, got {min_playlist_length}"
        )

    vocabs = load_vocabularies(vocab_dir)

    train_df = spark.read.parquet(str(train_path)).select(
        "pid", "pos", "track_id", "artist_id", "album_id"
    )

    # Keep only playlists long enough to split. Spark computes the per-pid length
    # and we inner-join it back to drop short playlists' rows.
    lengths = train_df.groupBy("pid").count().where(F.col("count") >= F.lit(min_playlist_length))
    kept = train_df.join(lengths.select("pid"), on="pid", how="inner")

    # Total occurrences (T) and number of playlists (N) drive array preallocation.
    num_occurrences = kept.count() # T
    num_playlists = lengths.count() # N

    # One row per playlist, its items sorted by pos. sort_array sorts the array
    # of structs by the struct's first field (pos), so the sequence keeps the
    # playlist's original track order.
    grouped = kept.groupBy("pid").agg(
        F.sort_array(
            F.collect_list(F.struct("pos", "track_id", "artist_id", "album_id"))
        ).alias("items")
    )

    # Preallocate the flat arrays, then stream playlists to the driver one at a
    # time (toLocalIterator) and encode them with the in-memory vocabularies.
    # Streaming keeps peak driver memory to one partition + the growing arrays,
    # instead of collecting all 59.7M rows at once.
    track_idx = np.empty(num_occurrences, dtype=np.int32)
    artist_idx = np.empty(num_occurrences, dtype=np.int32)
    album_idx = np.empty(num_occurrences, dtype=np.int32)
    offsets = np.empty(num_playlists + 1, dtype=np.int64)
    pids = np.empty(num_playlists, dtype=np.int64)
    offsets[0] = 0

    cursor = 0
    for i, row in enumerate(grouped.toLocalIterator()):
        items = row["items"]
        n = len(items)
        # encode_batch maps each string ID to its integer index (UNK for any not
        # in the vocab). Real train IDs at min_freq=1 all resolve to real indices.
        track_idx[cursor:cursor + n] = vocabs["track"].encode_batch([it["track_id"] for it in items])
        artist_idx[cursor:cursor + n] = vocabs["artist"].encode_batch([it["artist_id"] for it in items])
        album_idx[cursor:cursor + n] = vocabs["album"].encode_batch([it["album_id"] for it in items])
        cursor += n
        offsets[i + 1] = cursor
        pids[i] = row["pid"]

    # Guard against a mismatch between the counted total and what we actually
    # streamed (would indicate a Spark ordering / filtering surprise).
    if cursor != num_occurrences:
        raise RuntimeError(f"Filled {cursor} occurrences but expected {num_occurrences}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        track_idx=track_idx,
        artist_idx=artist_idx,
        album_idx=album_idx,
        offsets=offsets,
        pids=pids,
        min_playlist_length=np.int64(min_playlist_length),
    )
    return {
        "output_path": str(output_path),
        "num_playlists": int(num_playlists),
        "num_occurrences": int(num_occurrences),
        "min_playlist_length": int(min_playlist_length),
    }


# ----------------------------------------------------------------------------
# Stage 2: training-time dataset
# ----------------------------------------------------------------------------

class PlaylistDataset(Dataset):
    """Serves (context, positive) pairs from the cached train playlists.

    One item == one playlist. ``__getitem__`` samples a single positive track and
    returns the remaining tracks as context (a random leave-one-out split), so
    across epochs the same playlist yields different positives. Sampling uses
    NumPy's global RNG; seed it per worker with :func:`seed_worker` for
    reproducible, fork-safe randomness.
    """

    def __init__(self, cache_path, max_context_len=100):
        """Args:
            cache_path: .npz produced by :func:`build_cache`.
            max_context_len: cap on context length; longer contexts are randomly
                subsampled (without replacement) so compute/memory stay bounded.
        """
        if max_context_len < 1:
            raise ValueError(f"max_context_len must be >= 1, got {max_context_len}")

        data = np.load(cache_path)
        # Kept resident in memory; these are the flat CSR arrays.
        self.track_idx = data["track_idx"]
        self.artist_idx = data["artist_idx"]
        self.album_idx = data["album_idx"]
        self.offsets = data["offsets"]
        self.pids = data["pids"]
        self.max_context_len = max_context_len

    def __len__(self):
        return len(self.offsets) - 1

    def __getitem__(self, index):
        start, end = int(self.offsets[index]), int(self.offsets[index + 1])
        n = end - start  # playlist length; guaranteed >= 2 by the cache builder

        # Pick one occurrence in [start, end) as the positive.
        pos = start + int(np.random.randint(n))

        # Context = every other occurrence in the playlist (leave-one-out). We
        # build absolute indices into the flat arrays, excluding the positive.
        context = np.concatenate([np.arange(start, pos), np.arange(pos + 1, end)])

        # Cap long playlists by sampling without replacement.
        if context.shape[0] > self.max_context_len:
            context = np.random.choice(context, size=self.max_context_len, replace=False)

        return {
            "context_track": torch.from_numpy(self.track_idx[context].astype(np.int64)),
            "context_artist": torch.from_numpy(self.artist_idx[context].astype(np.int64)),
            "context_album": torch.from_numpy(self.album_idx[context].astype(np.int64)),
            "pos_track": torch.tensor(int(self.track_idx[pos]), dtype=torch.long),
            "pos_artist": torch.tensor(int(self.artist_idx[pos]), dtype=torch.long),
            "pos_album": torch.tensor(int(self.album_idx[pos]), dtype=torch.long),
            "pid": torch.tensor(int(self.pids[index]), dtype=torch.long),
        }


def collate_playlists(batch, pad_index=PAD_INDEX):
    """Collate variable-length contexts into padded tensors + a mask.

    Contexts are padded to the batch's longest context with ``pad_index`` (PAD,
    index 0). ``context_mask`` is True on real tokens and False on PAD, so the
    playlist tower can exclude padded slots when it pools.

    Returns a dict of batched tensors:
        context_track/artist/album : long [B, L]
        context_mask               : bool [B, L]  (True = real, False = PAD)
        pos_track/artist/album     : long [B]
        pid                        : long [B]
    """
    batch_size = len(batch)
    lengths = [sample["context_track"].shape[0] for sample in batch]
    max_len = max(lengths)

    context = {
        entity: torch.full((batch_size, max_len), pad_index, dtype=torch.long)
        for entity in ENTITIES
    }
    context_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)

    for i, sample in enumerate(batch):
        n = lengths[i]
        context["track"][i, :n] = sample["context_track"]
        context["artist"][i, :n] = sample["context_artist"]
        context["album"][i, :n] = sample["context_album"]
        context_mask[i, :n] = True

    return {
        "context_track": context["track"],
        "context_artist": context["artist"],
        "context_album": context["album"],
        "context_mask": context_mask,
        "pos_track": torch.stack([s["pos_track"] for s in batch]),
        "pos_artist": torch.stack([s["pos_artist"] for s in batch]),
        "pos_album": torch.stack([s["pos_album"] for s in batch]),
        "pid": torch.stack([s["pid"] for s in batch]),
    }


def seed_worker(worker_id):
    """DataLoader worker_init_fn: give each worker a distinct, reproducible seed.

    Without this, forked workers copy the parent RNG state and draw identical
    "random" positives. Derives each worker's seed from torch's per-worker base
    seed (set from the DataLoader's generator).
    """
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)


def make_dataloader(dataset, batch_size, shuffle=True, num_workers=0, seed=None, pad_index=PAD_INDEX):
    """Wrap a :class:`PlaylistDataset` in a DataLoader with the padding collate.

    Pass ``seed`` for a reproducible shuffle order and worker seeding.
    """
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    def _collate(batch):
        return collate_playlists(batch, pad_index=pad_index)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=_collate,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=generator,
        drop_last=True,  # keep in-batch-negative batches a uniform size
    )


# ----------------------------------------------------------------------------
# CLI: build the cache
# ----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Build the cached per-playlist tensor for the two-tower model from the gold train split."
    )
    parser.add_argument(
        "--input",
        default="data/gold/train_playlist_tracks.parquet",
        help="Gold train_playlist_tracks Parquet table (train only).",
    )
    parser.add_argument("--vocab-dir", default="artifacts/vocab", help="Directory with the saved vocab JSONs.")
    parser.add_argument("--output", default="artifacts/twotower/train_playlists.npz", help="Destination .npz path.")
    parser.add_argument("--master", default="local[*]", help="Spark master URL. Default: local[*].")
    parser.add_argument("--app-name", default="spotify-mpd-twotower-cache", help="Spark application name.")
    parser.add_argument("--driver-memory", default="8g", help="Spark driver memory. Example: 8g.")
    parser.add_argument(
        "--min-playlist-length",
        type=int,
        default=2,
        help="Drop playlists shorter than this (must be >= 2 for a leave-one-out split).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing cache.")
    return parser.parse_args()


def main():
    import json

    args = parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input train table does not exist: {input_path}")

    output_path = Path(args.output)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Cache already exists and --overwrite was not set: {output_path}")

    spark = create_spark_session(args.app_name, args.master, args.driver_memory)
    try:
        summary = build_cache(
            input_path, args.vocab_dir, output_path, args.min_playlist_length, spark
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

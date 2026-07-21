"""Unit tests for the two-tower data pipeline in scripts/twotower/dataset.py.

These target the non-Spark parts — PlaylistDataset, collate_playlists, and
make_dataloader — using a hand-built .npz cache, so they need only torch/numpy.
The Spark cache builder (build_cache) is exercised separately as a smoke test.
"""
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from twotower.dataset import (  # noqa: E402
    PlaylistDataset,
    collate_playlists,
    make_dataloader,
    PAD_INDEX,
)


def _write_cache(path):
    """Two playlists in CSR layout: pid 10 has 3 tracks, pid 20 has 4 tracks.

    Track/artist/album indices start at 2 (real IDs; 0=PAD, 1=UNK are reserved).
    """
    # playlist 10: 3 occurrences | playlist 20: 4 occurrences
    track_idx = np.array([2, 3, 4, 5, 6, 7, 8], dtype=np.int32)
    artist_idx = np.array([12, 13, 14, 15, 16, 17, 18], dtype=np.int32)
    album_idx = np.array([22, 23, 24, 25, 26, 27, 28], dtype=np.int32)
    offsets = np.array([0, 3, 7], dtype=np.int64)
    pids = np.array([10, 20], dtype=np.int64)
    np.savez(
        path,
        track_idx=track_idx,
        artist_idx=artist_idx,
        album_idx=album_idx,
        offsets=offsets,
        pids=pids,
        min_playlist_length=np.int64(2),
    )


def test_leave_one_out_shapes(tmp_path):
    """Each item drops exactly one track as the positive; context is the rest."""
    cache = tmp_path / "cache.npz"
    _write_cache(cache)
    ds = PlaylistDataset(cache, max_context_len=100)

    assert len(ds) == 2

    # playlist 0 has 3 tracks -> context length 2; playlist 1 has 4 -> length 3.
    item0 = ds[0]
    assert item0["context_track"].shape[0] == 2
    assert item0["pos_track"].ndim == 0  # scalar positive
    assert int(item0["pid"]) == 10

    item1 = ds[1]
    assert item1["context_track"].shape[0] == 3
    assert int(item1["pid"]) == 20


def test_positive_excluded_from_context(tmp_path):
    """The sampled positive occurrence is never also in the context.

    Track indices within each synthetic playlist are unique, so the positive's
    (track, artist, album) triple must not reappear in the context.
    """
    cache = tmp_path / "cache.npz"
    _write_cache(cache)
    ds = PlaylistDataset(cache, max_context_len=100)

    np.random.seed(0)
    for _ in range(50):  # many draws to hit different sampled positives
        item = ds[1]
        ctx = set(item["context_track"].tolist())
        assert int(item["pos_track"]) not in ctx
        assert len(ctx) == 3  # 4-track playlist, one held out


def test_max_context_len_caps_and_subsamples(tmp_path):
    """A context longer than max_context_len is subsampled without replacement."""
    cache = tmp_path / "cache.npz"
    _write_cache(cache)
    ds = PlaylistDataset(cache, max_context_len=2)  # playlist 1 has 3 context tracks

    np.random.seed(1)
    item = ds[1]
    ctx = item["context_track"].tolist()
    assert len(ctx) == 2                     # capped
    assert len(set(ctx)) == 2                # no duplicates (without replacement)


def test_collate_pads_and_masks(tmp_path):
    """collate pads to the batch max with PAD and marks real tokens in the mask."""
    cache = tmp_path / "cache.npz"
    _write_cache(cache)
    ds = PlaylistDataset(cache, max_context_len=100)

    np.random.seed(2)
    batch = [ds[0], ds[1]]  # context lengths 2 and 3
    out = collate_playlists(batch)

    B, L = out["context_track"].shape
    assert B == 2
    assert L == 3  # padded to the longest context in the batch

    # Row 0 (length 2) must have its last slot padded and masked out.
    assert out["context_track"][0, 2].item() == PAD_INDEX
    assert out["context_mask"][0].tolist() == [True, True, False]
    assert out["context_mask"][1].tolist() == [True, True, True]

    # Mask must be True exactly where the token is not PAD.
    assert torch.equal(out["context_mask"], out["context_track"] != PAD_INDEX)

    # Positives are stacked to [B].
    assert out["pos_track"].shape == (2,)
    assert out["pid"].tolist() == [10, 20]


def test_dataloader_determinism_with_seed(tmp_path):
    """Same seed -> identical batches; the pipeline is reproducible end to end."""
    cache = tmp_path / "cache.npz"
    _write_cache(cache)

    def first_batch(seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        ds = PlaylistDataset(cache, max_context_len=100)
        loader = make_dataloader(ds, batch_size=2, shuffle=True, num_workers=0, seed=seed)
        return next(iter(loader))

    a = first_batch(123)
    b = first_batch(123)
    assert torch.equal(a["context_track"], b["context_track"])
    assert torch.equal(a["pos_track"], b["pos_track"])
    assert torch.equal(a["pid"], b["pid"])

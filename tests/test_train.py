"""Unit tests for the training loop in scripts/twotower/train.py.

Uses a tiny hand-built cache + small model so it runs fast on CPU. Exercises one
epoch of the real loop and the checkpoint save/reload (both full and model-only).
"""
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from twotower.dataset import PlaylistDataset, make_dataloader  # noqa: E402
from twotower.model import TwoTowerModel  # noqa: E402
from twotower.train import train_one_epoch, save_checkpoint, in_batch_accuracy  # noqa: E402


def _write_cache(path):
    """Two playlists in CSR layout (indices stay < the model vocab sizes below)."""
    np.savez(
        path,
        track_idx=np.array([2, 3, 4, 5, 6, 7, 8], dtype=np.int32),
        artist_idx=np.array([12, 13, 14, 15, 16, 17, 18], dtype=np.int32),
        album_idx=np.array([22, 23, 24, 25, 26, 27, 28], dtype=np.int32),
        offsets=np.array([0, 3, 7], dtype=np.int64),
        pids=np.array([10, 20], dtype=np.int64),
        min_playlist_length=np.int64(2),
    )


def _tiny_setup(tmp_path):
    cache = tmp_path / "cache.npz"
    _write_cache(cache)
    ds = PlaylistDataset(cache, max_context_len=10)
    loader = make_dataloader(ds, batch_size=2, shuffle=True, num_workers=0, seed=0)
    model = TwoTowerModel(track_vocab_size=50, artist_vocab_size=20, album_vocab_size=30,
                          embedding_dim=8, temperature=0.05)
    return loader, model


def test_train_one_epoch_runs(tmp_path):
    """One epoch of the real loop produces a finite loss, a valid accuracy, and steps."""
    torch.manual_seed(0)
    np.random.seed(0)
    loader, model = _tiny_setup(tmp_path)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)

    avg_loss, avg_acc, global_step = train_one_epoch(
        model, loader, opt, torch.device("cpu"), epoch=1,
        log_every=100, max_steps=0, global_step=0,
    )
    assert np.isfinite(avg_loss)
    assert 0.0 <= avg_acc <= 1.0
    assert global_step >= 1


def test_max_steps_stops_early(tmp_path):
    """global_step honors max_steps."""
    torch.manual_seed(0)
    np.random.seed(0)
    loader, model = _tiny_setup(tmp_path)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)

    _, _, global_step = train_one_epoch(
        model, loader, opt, torch.device("cpu"), epoch=1,
        log_every=100, max_steps=1, global_step=0,
    )
    assert global_step == 1


def test_checkpoint_full_and_model_only(tmp_path):
    """Full checkpoint carries optimizer state; model-only omits it. Both reload."""
    _, model = _tiny_setup(tmp_path)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    config = {"vocab_sizes": {"track": 50, "artist": 20, "album": 30}, "embedding_dim": 8, "temperature": 0.05}

    full = tmp_path / "latest.pt"
    save_checkpoint(full, model, opt, epoch=3, config=config, include_optimizer=True)
    ck = torch.load(full, map_location="cpu", weights_only=False)
    assert ck["epoch"] == 3
    assert "optimizer_state_dict" in ck
    assert set(ck["model_state_dict"].keys()) == {"track_emb.weight", "artist_emb.weight", "album_emb.weight"}

    model_only = tmp_path / "best.pt"
    save_checkpoint(model_only, model, opt, epoch=3, config=config, include_optimizer=False)
    ck2 = torch.load(model_only, map_location="cpu", weights_only=False)
    assert "optimizer_state_dict" not in ck2
    assert ck2["config"] == config
    # model-only file should be smaller than the full one (no Adam state).
    assert model_only.stat().st_size < full.stat().st_size


def test_learns_on_a_fixed_batch(tmp_path):
    """Sanity: training many steps on the tiny data drives the loss down."""
    torch.manual_seed(0)
    np.random.seed(0)
    loader, model = _tiny_setup(tmp_path)
    opt = torch.optim.Adam(model.parameters(), lr=0.05)

    first, last, gstep = None, None, 0
    for _ in range(40):
        avg_loss, _, gstep = train_one_epoch(
            model, loader, opt, torch.device("cpu"), epoch=1,
            log_every=100, max_steps=0, global_step=gstep,
        )
        if first is None:
            first = avg_loss
        last = avg_loss
    assert last < first, f"loss did not drop: {first:.3f} -> {last:.3f}"

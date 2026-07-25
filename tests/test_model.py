"""Unit tests for the two-tower model in scripts/twotower/model.py.

Small synthetic vocab sizes and batches, so these run fast on CPU and exercise
the real forward pass, the in-batch loss, and one training step.
"""
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from twotower.model import TwoTowerModel, PAD_INDEX  # noqa: E402


def _tiny_model(dim=16):
    # Vocab sizes are small; indices in the batches below stay within them.
    torch.manual_seed(0)
    return TwoTowerModel(
        track_vocab_size=50, artist_vocab_size=20, album_vocab_size=30,
        embedding_dim=dim, temperature=0.05,
    )


def _fake_batch(B=4, L=5, dim_ok_vocab=(50, 20, 30)):
    """A collate-shaped batch with one PAD tail in row 0 to exercise the mask."""
    tv, av, alv = dim_ok_vocab
    torch.manual_seed(1)
    ctx_track = torch.randint(2, tv, (B, L))
    ctx_artist = torch.randint(2, av, (B, L))
    ctx_album = torch.randint(2, alv, (B, L))
    mask = torch.ones(B, L, dtype=torch.bool)
    # Row 0: only the first 2 tokens are real; rest are PAD.
    ctx_track[0, 2:] = PAD_INDEX
    ctx_artist[0, 2:] = PAD_INDEX
    ctx_album[0, 2:] = PAD_INDEX
    mask[0, 2:] = False
    return {
        "context_track": ctx_track, "context_artist": ctx_artist,
        "context_album": ctx_album, "context_mask": mask,
        "pos_track": torch.randint(2, tv, (B,)),
        "pos_artist": torch.randint(2, av, (B,)),
        "pos_album": torch.randint(2, alv, (B,)),
        "pid": torch.arange(B),
    }


def test_forward_shapes():
    """Both towers produce [B, D] vectors."""
    model = _tiny_model(dim=16)
    batch = _fake_batch(B=4, L=5)
    playlist_vec, item_vec = model(batch)
    assert playlist_vec.shape == (4, 16)
    assert item_vec.shape == (4, 16)


def test_loss_is_scalar_and_differentiable():
    """The in-batch loss is a finite scalar that carries gradient."""
    model = _tiny_model()
    batch = _fake_batch()
    playlist_vec, item_vec = model(batch)
    loss = model.in_batch_softmax_loss(playlist_vec, item_vec)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.requires_grad


def test_masked_pool_ignores_pad():
    """Padding must not affect the pooled playlist vector.

    Rebuild row 0's context with the PAD slots filled with arbitrary junk indices
    but still masked out; the pooled vector must be identical either way.
    """
    model = _tiny_model()
    batch = _fake_batch(B=2, L=5)

    v1 = model.encode_playlist(
        batch["context_track"], batch["context_artist"],
        batch["context_album"], batch["context_mask"],
    )

    # Corrupt the masked-out (PAD) positions of row 0 with different indices.
    tampered = {k: v.clone() for k, v in batch.items()}
    tampered["context_track"][0, 2:] = 7
    tampered["context_artist"][0, 2:] = 5
    tampered["context_album"][0, 2:] = 9
    v2 = model.encode_playlist(
        tampered["context_track"], tampered["context_artist"],
        tampered["context_album"], batch["context_mask"],   # same mask
    )

    assert torch.allclose(v1, v2, atol=1e-6), "masked PAD positions changed the pooled vector"


def test_perfect_alignment_beats_shuffled():
    """Loss should be lower when each playlist's positive is its true match.

    Feed identical playlist/item vectors (perfect alignment) vs. a shuffled
    pairing; the aligned version must have the smaller in-batch loss.
    """
    model = _tiny_model()
    torch.manual_seed(3)
    vecs = torch.randn(8, 16)

    aligned = model.in_batch_softmax_loss(vecs, vecs.clone())
    shuffled = model.in_batch_softmax_loss(vecs, vecs[torch.randperm(8)])
    assert aligned < shuffled


def test_pad_row_stays_zero_after_step():
    """After a real optimizer step, the PAD embedding row is still all zeros."""
    model = _tiny_model()
    batch = _fake_batch()
    opt = torch.optim.Adam(model.parameters(), lr=0.1)

    playlist_vec, item_vec = model(batch)
    loss = model.in_batch_softmax_loss(playlist_vec, item_vec)
    opt.zero_grad()
    loss.backward()
    opt.step()

    for emb in (model.track_emb, model.artist_emb, model.album_emb):
        assert torch.all(emb.weight[PAD_INDEX] == 0), "PAD row moved off zero"


def test_logq_setter_and_shapes():
    """set_item_log_q builds log(Q); a more frequent track has a larger (less
    negative) log_q than a rarer one."""
    model = _tiny_model()
    counts = torch.zeros(50)
    counts[5] = 100   # popular track
    counts[6] = 1     # rare track
    model.set_item_log_q(counts)
    assert model.item_log_q.shape == (50,)
    assert model.item_log_q[5] > model.item_log_q[6]   # popular -> higher log Q


def test_logq_noop_when_uniform():
    """Uniform sampling probabilities subtract the same constant from every column,
    which softmax is invariant to — so the loss matches the uncorrected one."""
    model = _tiny_model()
    batch = _fake_batch(B=4, L=5)
    playlist_vec, item_vec = model(batch)

    base = model.in_batch_softmax_loss(playlist_vec, item_vec)          # no correction
    model.set_item_log_q(torch.ones(50))                               # uniform counts
    uniform = model.in_batch_softmax_loss(playlist_vec, item_vec, item_indices=batch["pos_track"])
    assert torch.allclose(base, uniform, atol=1e-5)


def test_logq_changes_loss_when_skewed():
    """A skewed popularity distribution actually changes the loss."""
    model = _tiny_model()
    batch = _fake_batch(B=4, L=5)
    playlist_vec, item_vec = model(batch)

    base = model.in_batch_softmax_loss(playlist_vec, item_vec)
    skewed = torch.arange(1, 51, dtype=torch.float32)                  # very non-uniform counts
    model.set_item_log_q(skewed)
    corrected = model.in_batch_softmax_loss(playlist_vec, item_vec, item_indices=batch["pos_track"])
    assert not torch.allclose(base, corrected)
    assert torch.isfinite(corrected)


def test_training_reduces_loss():
    """A few steps on a fixed batch should drive the loss down (it can learn)."""
    model = _tiny_model()
    batch = _fake_batch(B=8, L=6)
    opt = torch.optim.Adam(model.parameters(), lr=0.05)

    losses = []
    for _ in range(30):
        playlist_vec, item_vec = model(batch)
        loss = model.in_batch_softmax_loss(playlist_vec, item_vec)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0], f"loss did not decrease: {losses[0]:.3f} -> {losses[-1]:.3f}"

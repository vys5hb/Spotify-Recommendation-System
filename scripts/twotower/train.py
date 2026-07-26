#!/usr/bin/env python3
"""Training loop for the two-tower retrieval model.

Reads the cached train playlists (the .npz from dataset.py), builds the model
(sized from the saved vocab), and trains it with the in-batch-negative softmax
loss. The same script runs on CPU locally for a quick smoke test and on a CUDA
GPU (the P100) for real training — it auto-detects the device.

Typical local smoke test (a few steps, small model, on CPU):

    python scripts/twotower/train.py --dim 32 --batch-size 64 --max-steps 20 --device cpu

Real run on the GPU:

    python scripts/twotower/train.py --dim 128 --batch-size 1024 --epochs 10
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Make the sibling modules importable whether run as a script or imported.
SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_ROOT = SCRIPT_DIR.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))
from twotower.dataset import PlaylistDataset, make_dataloader  # noqa: E402
from twotower.model import build_model_from_vocab_sizes  # noqa: E402

ENTITIES = ("track", "artist", "album")


def select_device(requested):
    """Pick the device: honor an explicit request, else auto-detect cuda > mps > cpu."""
    if requested and requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():   # Apple Silicon GPU
        return torch.device("mps")
    return torch.device("cpu")


def parse_hidden_dims(spec):
    """Turn a --hidden-dims string into a list of ints (or None for "no MLP").

    "256" -> [256]        one hidden layer
    "256,256" -> [256, 256]
    "" / "none" / "0" -> None   (purely linear towers, the original architecture)
    """
    if not spec or spec.strip().lower() in {"none", "0", ""}:
        return None
    return [int(part) for part in spec.split(",") if part.strip()]


def load_vocab_sizes(vocab_dir):
    """Read the three vocab sizes from vocab_metadata.json (fast, no full load)."""
    meta = json.loads((Path(vocab_dir) / "vocab_metadata.json").read_text())
    return {e: meta["entities"][e]["vocab_size"] for e in ENTITIES}


def move_batch(batch, device):
    """Move every tensor in the collated batch dict onto the target device."""
    return {key: value.to(device) for key, value in batch.items()}


def compute_item_counts(cache_path, num_tracks):
    """Per-track occurrence counts in train, for the logQ correction.

    bincount over the flat track_idx gives how many times each track index appears
    across all playlists — i.e. its sampling frequency as an in-batch negative.
    """
    data = np.load(cache_path)
    counts = np.bincount(data["track_idx"], minlength=num_tracks)
    return torch.from_numpy(counts)


def in_batch_accuracy(playlist_vec, item_vec):
    """Fraction of playlists whose top-scoring item (in the batch) is its own positive.

    A cheap training-time signal: the diagonal of the [B, B] similarity matrix
    should win each row. Not a real retrieval metric (that's evaluate.py's job),
    just a quick "is it learning?" readout.
    """
    pl = F.normalize(playlist_vec, dim=-1)
    it = F.normalize(item_vec, dim=-1)
    logits = pl @ it.t()                        # [B, B]
    preds = logits.argmax(dim=1)                # each row's best item
    targets = torch.arange(logits.size(0), device=logits.device)
    return (preds == targets).float().mean().item()


def train_one_epoch(model, loader, optimizer, device, epoch, log_every, max_steps, global_step):
    """Run one pass over the loader. Returns (avg_loss, avg_acc, updated_global_step)."""
    model.train()   # PyTorch: switch to training mode (affects dropout/batchnorm; a no-op here but good practice)

    total_loss, total_acc, n_batches = 0.0, 0.0, 0
    for step, batch in enumerate(loader):
        batch = move_batch(batch, device)

        # Forward: encode both towers, then the in-batch-negative softmax loss.
        # pos_track is passed so the loss can apply the logQ correction (a no-op
        # unless model.set_item_log_q was called).
        playlist_vec, item_vec = model(batch)
        loss = model.in_batch_softmax_loss(playlist_vec, item_vec, item_indices=batch["pos_track"])

        # Backward + update: the standard three-step dance.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrics (detached from the graph; just for logging).
        with torch.no_grad():
            acc = in_batch_accuracy(playlist_vec, item_vec)
        total_loss += loss.item()
        total_acc += acc
        n_batches += 1
        global_step += 1

        if step % log_every == 0:
            print(f"  epoch {epoch}  step {step:>6d}  loss {loss.item():.4f}  in-batch acc {acc:.3f}")

        if max_steps and global_step >= max_steps:
            break

    avg_loss = total_loss / max(n_batches, 1)
    avg_acc = total_acc / max(n_batches, 1)
    return avg_loss, avg_acc, global_step


def save_checkpoint(path, model, optimizer, epoch, config, include_optimizer=True):
    """Save a checkpoint.

    With include_optimizer=True the file also holds Adam's state (2x the model
    size) so training can resume. With include_optimizer=False it's model-only
    (~1/3 the size) — enough for evaluation / deployment, where the optimizer is
    not needed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "config": config,
    }
    if include_optimizer:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(payload, path)


def parse_args():
    parser = argparse.ArgumentParser(description="Train the two-tower retrieval model.")
    parser.add_argument("--cache", default="artifacts/twotower/train_playlists.npz", help="Cached train playlists (.npz from dataset.py).")
    parser.add_argument("--vocab-dir", default="artifacts/vocab", help="Directory with the vocab JSONs + metadata.")
    parser.add_argument("--out", default="artifacts/twotower/checkpoints", help="Directory to write checkpoints.")
    parser.add_argument("--dim", type=int, default=128, help="Embedding dimension.")
    parser.add_argument("--hidden-dims", default="", help="Comma-separated MLP head widths per tower, e.g. '256' or '256,256'. Empty (default) = no MLP, purely linear towers.")
    parser.add_argument("--temperature", type=float, default=0.05, help="Softmax temperature for the loss.")
    parser.add_argument("--batch-size", type=int, default=1024, help="Playlists per step (also the in-batch negative count).")
    parser.add_argument("--epochs", type=int, default=5, help="Number of full passes over the training playlists.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--max-context-len", type=int, default=100, help="Cap on context length per playlist.")
    parser.add_argument("--limit-playlists", type=int, default=0, help="Train on a random subset of N playlists (0 = all). For fast local sanity runs.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker processes.")
    parser.add_argument("--device", default="auto", help="cuda / mps / cpu, or auto (default).")
    parser.add_argument("--seed", type=int, default=42, help="Seed for shuffle, sampling, and init.")
    parser.add_argument("--log-every", type=int, default=50, help="Print a line every N steps.")
    parser.add_argument("--max-steps", type=int, default=0, help="Stop after N total steps (0 = no limit). Use for smoke tests.")
    parser.add_argument("--logq", action="store_true", help="Apply the logQ sampling-bias correction (debias in-batch negatives against popular tracks).")
    return parser.parse_args()


def main():
    args = parse_args()

    # Reproducibility: seed the RNGs the model init and dataloader draw from.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = select_device(args.device)
    print(f"device: {device}")

    # Build the model, sized from the saved vocab, and move it to the device.
    vocab_sizes = load_vocab_sizes(args.vocab_dir)
    hidden_dims = parse_hidden_dims(args.hidden_dims)
    model = build_model_from_vocab_sizes(
        vocab_sizes, embedding_dim=args.dim, temperature=args.temperature, hidden_dims=hidden_dims,
    )
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"vocab sizes: {vocab_sizes}")
    print(f"model: {n_params:,} params at dim {args.dim}")
    print(f"tower head: {'MLP ' + str(hidden_dims) if hidden_dims else 'linear (no MLP)'}")

    # logQ correction: fill the model's per-track log sampling probability from the
    # train frequencies. Without this the correction stays a no-op.
    if args.logq:
        model.set_item_log_q(compute_item_counts(args.cache, vocab_sizes["track"]).to(device))
        print("logQ correction: ENABLED (in-batch negatives debiased by track popularity)")

    # Data.
    dataset = PlaylistDataset(args.cache, max_context_len=args.max_context_len)
    # Optional subset for fast local runs: pick N random playlists from the cache
    # (no Spark re-run needed — we just index into the in-memory dataset).
    if args.limit_playlists and args.limit_playlists < len(dataset):
        rng = np.random.default_rng(args.seed)
        subset_idx = rng.choice(len(dataset), size=args.limit_playlists, replace=False)
        dataset = torch.utils.data.Subset(dataset, subset_idx.tolist())
        print(f"limiting to {args.limit_playlists:,} random playlists (local sanity run)")
    loader = make_dataloader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, seed=args.seed,
    )
    print(f"dataset: {len(dataset):,} playlists  |  {len(loader):,} batches/epoch at batch size {args.batch_size}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Config saved into every checkpoint so evaluate.py can rebuild the model.
    config = {
        "vocab_sizes": vocab_sizes,
        "embedding_dim": args.dim,
        "temperature": args.temperature,
        "hidden_dims": hidden_dims,
        "logq": bool(args.logq),
    }

    out_dir = Path(args.out)
    global_step = 0
    best_acc = -1.0
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        avg_loss, avg_acc, global_step = train_one_epoch(
            model, loader, optimizer, device, epoch,
            args.log_every, args.max_steps, global_step,
        )
        elapsed = time.time() - start
        print(f"epoch {epoch} done  avg loss {avg_loss:.4f}  avg in-batch acc {avg_acc:.3f}  ({elapsed:.1f}s)")

        # latest.pt: full state (weights + optimizer), overwritten each epoch, for resuming.
        save_checkpoint(out_dir / "latest.pt", model, optimizer, epoch, config, include_optimizer=True)

        # best.pt: model-only (smaller), saved only when in-batch accuracy improves.
        # This is the checkpoint evaluate.py loads.
        if avg_acc > best_acc:
            best_acc = avg_acc
            save_checkpoint(out_dir / "best.pt", model, optimizer, epoch, config, include_optimizer=False)
            print(f"  new best in-batch acc {best_acc:.3f} -> saved best.pt")

        if args.max_steps and global_step >= args.max_steps:
            print(f"reached max_steps={args.max_steps}, stopping early.")
            break

    print(json.dumps({
        "checkpoints_dir": str(out_dir),
        "epochs_run": epoch,
        "total_steps": global_step,
        "best_in_batch_acc": round(best_acc, 4),
    }, indent=2))


if __name__ == "__main__":
    main()

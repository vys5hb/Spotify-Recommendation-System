#!/usr/bin/env python3
"""The two-tower retrieval model.

A "two-tower" model learns two encoders that map into the SAME vector space:

  - the **playlist tower** turns a playlist's context tracks into one vector
  - the **item tower** turns a single candidate track into one vector

Training pulls a playlist's vector close to its held-out positive track's vector,
and pushes it away from other tracks. At serving time you precompute every track's
item vector once, then retrieval = "find the item vectors nearest this playlist
vector" (a fast nearest-neighbor search) — which is why this is a *retrieval* model.

Both towers read from the SAME three embedding tables (track / artist / album).
That sharing is deliberate: a track has one learned vector whether it appears as
context in a playlist or as a candidate item, so the two towers speak the same
language and their vectors are directly comparable.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# Index 0 is PAD in every vocab (see vocab.py). We hand it to nn.Embedding as
# padding_idx so its row is frozen at the zero vector and never gets gradient.
PAD_INDEX = 0

# Defaults. embedding_dim is the DIM_SIZE knob (capacity vs. memory); temperature
# sharpens the softmax over in-batch scores (smaller = sharper / more confident).
DEFAULT_EMBEDDING_DIM = 128
DEFAULT_TEMPERATURE = 0.05

# hidden_dims=None means "no MLP head" — each tower stays purely linear (sum the
# entity embeddings, pool, done), which is the original architecture.
DEFAULT_HIDDEN_DIMS = None


def build_mlp(input_dim, hidden_dims, output_dim):
    """Build the tower head: Linear -> ReLU -> ... -> Linear, or a pass-through.

    With hidden_dims falsy (None or []) this returns nn.Identity, so the tower
    behaves EXACTLY as it did before the MLP existed — that keeps old checkpoints
    loadable and makes the with/without comparison a clean ablation.

    With e.g. hidden_dims=[256] you get Linear(D, 256) -> ReLU -> Linear(256, D).
    Hidden layers get ReLU; the final layer stays linear because its output is the
    embedding we L2-normalize (an activation there would distort the vector).
    """
    if not hidden_dims:
        return nn.Identity()   # nn.Identity: a built-in no-op layer, returns its input unchanged

    layers = []
    prev_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(nn.ReLU())
        prev_dim = hidden_dim
    layers.append(nn.Linear(prev_dim, output_dim))   # project back to D, no activation
    return nn.Sequential(*layers)   # nn.Sequential: runs the layers in order


class TwoTowerModel(nn.Module):
    """Shared-embedding two-tower model with in-batch-negative softmax training."""

    def __init__(
        self,
        track_vocab_size,
        artist_vocab_size,
        album_vocab_size,
        embedding_dim=DEFAULT_EMBEDDING_DIM,
        temperature=DEFAULT_TEMPERATURE,
        hidden_dims=DEFAULT_HIDDEN_DIMS,
        pad_index=PAD_INDEX,
    ):
        """Args:
            track/artist/album_vocab_size: number of rows in each embedding table
                (each vocab's ``.size``, which already includes PAD + UNK).
            embedding_dim: length D of every entity vector (and of the final
                playlist / item vectors, since we sum same-sized embeddings).
            temperature: divides the similarity scores before softmax. Lower =
                sharper distribution, larger gradients on the hardest negatives.
            hidden_dims: list of hidden layer widths for each tower's MLP head,
                e.g. [256]. None/[] = no MLP (purely linear towers, the original
                architecture). Each tower gets its OWN MLP — the towers are meant
                to be independent encoders, even though they share the embedding
                tables underneath.
            pad_index: the reserved PAD row to freeze at zero (index 0).
        """
        super().__init__()  # required: sets up the nn.Module machinery before we add layers

        self.embedding_dim = embedding_dim
        self.temperature = temperature
        self.hidden_dims = hidden_dims

        # The three shared embedding tables. padding_idx=pad_index keeps row 0
        # (PAD) pinned to zeros with no gradient; every other row (including UNK
        # at index 1) is a normal learnable vector.
        # nn.Embedding needs int64 tensors
        self.track_emb = nn.Embedding(track_vocab_size, embedding_dim, padding_idx=pad_index)
        self.artist_emb = nn.Embedding(artist_vocab_size, embedding_dim, padding_idx=pad_index)
        self.album_emb = nn.Embedding(album_vocab_size, embedding_dim, padding_idx=pad_index)

        # logQ / sampling-bias correction (Yi et al. 2019): per-track log sampling
        # probability, subtracted from the in-batch logits to undo the popularity
        # bias of in-batch negatives (popular tracks appear as negatives far more
        # often, so the model over-penalizes them without this). Non-persistent
        # buffer (not saved in the checkpoint): all zeros = no correction, which is
        # the default. train.py fills it from the train track frequencies via
        # set_item_log_q when --logq is enabled.
        self.register_buffer("item_log_q", torch.zeros(track_vocab_size), persistent=False)

        # MLP heads, one per tower. These sit AFTER pooling (playlist side) and
        # after the embedding sum (item side), giving each tower the capacity to
        # learn a nonlinear transform instead of just averaging vectors.
        self.playlist_mlp = build_mlp(embedding_dim, hidden_dims, embedding_dim)
        self.item_mlp = build_mlp(embedding_dim, hidden_dims, embedding_dim)

        self._init_weights()

    def set_item_log_q(self, item_counts):
        """Populate the logQ table from per-track train occurrence counts.

        Q(track) = count / total; we store log(Q). A track's Q is how likely it is
        to appear as an in-batch negative, so subtracting log(Q) later removes that
        sampling bias. Counts of 0 (PAD/UNK/unused rows) are floored so log is finite.
        """
        counts = item_counts.to(self.item_log_q.device, dtype=torch.float64)
        q = counts / counts.sum()
        q = q.clamp(min=1e-12)   # floor so log(0) never happens
        self.item_log_q.copy_(torch.log(q).to(self.item_log_q.dtype))

    def _init_weights(self):
        """Small-magnitude init so early scores stay in a sane range for softmax."""
        for emb in (self.track_emb, self.artist_emb, self.album_emb):
            nn.init.normal_(emb.weight, mean=0.0, std=0.05)
            # nn.init overwrites the whole matrix, so re-zero the PAD row it clobbered.
            with torch.no_grad():
                emb.weight[PAD_INDEX].zero_()

    # ------------------------------------------------------------------
    # Encoders (the two towers). Both go through embed_tokens, so both read
    # the same shared tables.
    # ------------------------------------------------------------------

    def embed_tokens(self, track, artist, album):
        """Turn (track, artist, album) index tensors into one vector per token.

        We SUM the three entity embeddings, so a token's vector carries its track,
        artist, and album signal at once and stays dimension D. Works for any
        shape: [B, L] context -> [B, L, D], or [B] items -> [B, D].
        """
        return self.track_emb(track) + self.artist_emb(artist) + self.album_emb(album) # Adds the three embeddings together

    def encode_playlist(self, context_track, context_artist, context_album, context_mask):
        """Playlist tower: embed the context tokens and masked-mean-pool them.

        Args:
            context_track/artist/album: [B, L] index tensors (0 where PADded).
            context_mask: [B, L] bool, True on real tokens, False on PAD.

        Returns:
            [B, D] playlist vectors.
        """
        tokens = self.embed_tokens(context_track, context_artist, context_album)  # Element-wise sum of the three embeddings over all tracks in playlist

        # Masked mean over the L (context) dimension: sum only the real tokens,
        # divide by how many there were. PAD tokens are zeroed out by the mask,
        # so they contribute nothing — this is what context_mask is for.
        mask = context_mask.unsqueeze(-1).to(tokens.dtype)  # [B, L, 1]
        summed = (tokens * mask).sum(dim=1)                  # [B, D]
        counts = mask.sum(dim=1).clamp(min=1.0)              # [B, 1]  (avoid /0)
        pooled = summed / counts                             # [B, D]
        # Returns average of the embeddings of all tracks in the playlist (gives general representation of the playlist through the average of the 3 embedding tables)

        # MLP head runs AFTER pooling, so PAD tokens are already masked out and
        # can't leak in. With hidden_dims=None this is nn.Identity (a no-op).
        return self.playlist_mlp(pooled)                     # [B, D]

    def encode_item(self, track, artist, album):
        """Item tower: embed a single candidate track into a [B, D] vector.

        No pooling — one token in, one vector out. Uses the same tables as the
        playlist tower, so item vectors live in the same space as playlist vectors.
        """
        tokens = self.embed_tokens(track, artist, album)  # [B, D]
        return self.item_mlp(tokens)                      # [B, D]  (Identity if no MLP)

    def forward(self, batch):
        """Encode a training batch with both towers.

        ``forward`` is the fixed method name PyTorch calls when you do
        ``model(batch)``; don't call it directly, call the module.

        Args:
            batch: the dict from collate_playlists (context_* [B, L], context_mask
                [B, L], pos_* [B]).

        Returns:
            (playlist_vec [B, D], item_vec [B, D]).
        """
        playlist_vec = self.encode_playlist(
            batch["context_track"], batch["context_artist"],
            batch["context_album"], batch["context_mask"],
        )
        item_vec = self.encode_item(
            batch["pos_track"], batch["pos_artist"], batch["pos_album"],
        )
        return playlist_vec, item_vec

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def in_batch_softmax_loss(self, playlist_vec, item_vec, item_indices=None):
        """In-batch-negative softmax cross-entropy (optionally logQ-corrected).

        For a batch of B playlists and their B positive items, score every
        playlist against every item -> a [B, B] matrix. Row i's correct answer is
        item i (its own positive on the diagonal); the other B-1 items in the row
        are the "free" negatives. Cross-entropy then trains each row to prefer its
        diagonal item.

        We L2-normalize both sides first, so the score is cosine similarity in
        [-1, 1]; dividing by the temperature scales it into useful softmax logits.

        If ``item_indices`` (the batch's B track indices, i.e. pos_track) is given,
        we apply the **logQ correction**: subtract log(Q(item_j)) from column j,
        where Q(item_j) is that track's sampling probability. Popular tracks (large
        Q) get a bigger subtraction, undoing the in-batch-negative popularity bias.
        With the default zero ``item_log_q`` this is a no-op, so passing indices is
        harmless until set_item_log_q has been called.
        """
        playlist_vec = F.normalize(playlist_vec, dim=-1)
        item_vec = F.normalize(item_vec, dim=-1)

        # [B, D] @ [B, D]^T = [B, B]
        logits = playlist_vec @ item_vec.t() / self.temperature

        # logQ correction: subtract log(Q(item_j)) from column j (broadcast over
        # rows). Column j is item j's column for EVERY playlist, so the correction
        # is per-candidate, not per-query.
        if item_indices is not None:
            log_q = self.item_log_q[item_indices]        # [B]
            logits = logits - log_q.unsqueeze(0)         # [B, B] - [1, B]

        # The positive for row i sits at column i (both came from the same batch position), so the targets are just 0, 1, ..., B-1.
        targets = torch.arange(logits.size(0), device=logits.device)
        return F.cross_entropy(logits, targets)


def build_model_from_vocab_sizes(
    vocab_sizes,
    embedding_dim=DEFAULT_EMBEDDING_DIM,
    temperature=DEFAULT_TEMPERATURE,
    hidden_dims=DEFAULT_HIDDEN_DIMS,
):
    """Convenience constructor from a {'track','artist','album': size} dict.

    Mirrors the entity naming used by dataset.py / build_vocab.py so training code
    can wire vocab sizes straight in.
    """
    return TwoTowerModel(
        track_vocab_size=vocab_sizes["track"],
        artist_vocab_size=vocab_sizes["artist"],
        album_vocab_size=vocab_sizes["album"],
        embedding_dim=embedding_dim,
        temperature=temperature,
        hidden_dims=hidden_dims,
    )

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


class TwoTowerModel(nn.Module):
    """Shared-embedding two-tower model with in-batch-negative softmax training."""

    def __init__(
        self,
        track_vocab_size,
        artist_vocab_size,
        album_vocab_size,
        embedding_dim=DEFAULT_EMBEDDING_DIM,
        temperature=DEFAULT_TEMPERATURE,
        pad_index=PAD_INDEX,
    ):
        """Args:
            track/artist/album_vocab_size: number of rows in each embedding table
                (each vocab's ``.size``, which already includes PAD + UNK).
            embedding_dim: length D of every entity vector (and of the final
                playlist / item vectors, since we sum same-sized embeddings).
            temperature: divides the similarity scores before softmax. Lower =
                sharper distribution, larger gradients on the hardest negatives.
            pad_index: the reserved PAD row to freeze at zero (index 0).
        """
        super().__init__()  # required: sets up the nn.Module machinery before we add layers

        self.embedding_dim = embedding_dim
        self.temperature = temperature

        # The three shared embedding tables. padding_idx=pad_index keeps row 0
        # (PAD) pinned to zeros with no gradient; every other row (including UNK
        # at index 1) is a normal learnable vector.
        # nn.Embedding needs int64 tensors
        self.track_emb = nn.Embedding(track_vocab_size, embedding_dim, padding_idx=pad_index)
        self.artist_emb = nn.Embedding(artist_vocab_size, embedding_dim, padding_idx=pad_index)
        self.album_emb = nn.Embedding(album_vocab_size, embedding_dim, padding_idx=pad_index)

        self._init_weights()

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
        return summed / counts                               # [B, D]
        # Returns average of the embeddings of all tracks in the playlist (gives general representation of the playlist through the average of the 3 embedding tables)

    def encode_item(self, track, artist, album):
        """Item tower: embed a single candidate track into a [B, D] vector.

        No pooling — one token in, one vector out. Uses the same tables as the
        playlist tower, so item vectors live in the same space as playlist vectors.
        """
        return self.embed_tokens(track, artist, album)  # [B, D]

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

    def in_batch_softmax_loss(self, playlist_vec, item_vec):
        """In-batch-negative softmax cross-entropy.

        For a batch of B playlists and their B positive items, score every
        playlist against every item -> a [B, B] matrix. Row i's correct answer is
        item i (its own positive on the diagonal); the other B-1 items in the row
        are the "free" negatives. Cross-entropy then trains each row to prefer its
        diagonal item.

        We L2-normalize both sides first, so the score is cosine similarity in
        [-1, 1]; dividing by the temperature scales it into useful softmax logits.
        (Swap the normalization out for a raw dot product if you prefer.)
        """
        playlist_vec = F.normalize(playlist_vec, dim=-1)
        item_vec = F.normalize(item_vec, dim=-1)
        
        # [B, D] @ [B, D]^T = [B, B]
        logits = playlist_vec @ item_vec.t() / self.temperature 

        # The positive for row i sits at column i (both came from the same batch position), so the targets are just 0, 1, ..., B-1.
        targets = torch.arange(logits.size(0), device=logits.device)
        return F.cross_entropy(logits, targets)


def build_model_from_vocab_sizes(vocab_sizes, embedding_dim=DEFAULT_EMBEDDING_DIM, temperature=DEFAULT_TEMPERATURE):
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
    )

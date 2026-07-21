"""Unit tests for the Vocabulary class in scripts/twotower/vocab.py.

These tests target vocab.py with the PAD/UNK split: index 0 = PAD, index 1 = UNK
(the index every unknown / below-cutoff ID encodes to), and real IDs start at
index 2.
"""
import random
import sys
from pathlib import Path

# The Vocabulary class lives in scripts/twotower/vocab.py. twotower is a package
# (has __init__.py), so we put the scripts/ directory on the path and import it
# the same way build_vocab.py does.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from twotower.vocab import Vocabulary  # noqa: E402  (import after sys.path setup)


# The index every unknown / below-cutoff ID encodes to. With the PAD/UNK split,
# UNK is index 1 (index 0 is PAD, which encode() never returns).
UNKNOWN_INDEX = Vocabulary.UNK_INDEX


def test_round_trip_encode_decode():
    """Every known ID encodes to an index that decodes back to the same ID."""
    ids = ["track_a", "track_b", "track_c"]
    vocab = Vocabulary.build(ids)

    # Real IDs occupy contiguous indices starting right after the reserved slots.
    assert min(vocab.id_to_index.values()) == Vocabulary.NUM_RESERVED
    assert Vocabulary.PAD_INDEX not in vocab.index_to_id
    assert Vocabulary.UNK_INDEX not in vocab.index_to_id

    for string_id in ids:
        index = vocab.encode(string_id)
        assert index != UNKNOWN_INDEX, "a known ID must not encode to the unknown index"
        assert vocab.decode(index) == string_id

    # encode_batch should match element-wise encode.
    assert vocab.encode_batch(ids) == [vocab.encode(i) for i in ids]


def test_unknown_ids_map_to_reserved_index():
    """IDs not in the vocab encode to the reserved unknown index, and that index
    does not decode back to any real ID."""
    vocab = Vocabulary.build(["track_a", "track_b"])

    assert vocab.encode("never_seen") == UNKNOWN_INDEX
    assert vocab.encode_batch(["track_a", "never_seen", "track_b"]) == [
        vocab.encode("track_a"),
        UNKNOWN_INDEX,
        vocab.encode("track_b"),
    ]
    # The reserved index has no real string ID behind it.
    assert vocab.decode(UNKNOWN_INDEX) is None


def test_save_load_identical_mapping(tmp_path):
    """A saved vocab reloads to an identical id->index mapping and encodes the
    same way, including for unknown IDs."""
    ids = ["c", "a", "b", "d"]
    vocab = Vocabulary.build(ids)

    path = tmp_path / "track_vocab.json"
    vocab.save(path)
    reloaded = Vocabulary.load(path)

    assert reloaded.id_to_index == vocab.id_to_index
    assert reloaded.size == vocab.size
    # Encoding behaviour survives the round trip for both known and unknown IDs.
    probe = ids + ["missing"]
    assert reloaded.encode_batch(probe) == vocab.encode_batch(probe)
    # Decoding survives too.
    for string_id in ids:
        assert reloaded.decode(vocab.encode(string_id)) == string_id


def test_determinism_across_shuffled_input():
    """Building from the same IDs in a different order yields identical indices.

    This is the guarantee build_vocab.py relies on: because Spark collect()
    ordering is not stable, the vocab must depend only on the SET of IDs.
    """
    ids = [f"id_{n}" for n in range(200)]

    shuffled = ids[:]
    random.Random(1234).shuffle(shuffled)
    assert shuffled != ids, "sanity: inputs should actually be in a different order"

    vocab_a = Vocabulary.build(ids)
    vocab_b = Vocabulary.build(shuffled)

    assert vocab_a.id_to_index == vocab_b.id_to_index
    assert vocab_a.index_to_id == vocab_b.index_to_id
    # Duplicates in the input must not change the mapping either.
    vocab_c = Vocabulary.build(ids + ids)
    assert vocab_c.id_to_index == vocab_a.id_to_index

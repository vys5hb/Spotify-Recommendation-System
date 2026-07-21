import json
from pathlib import Path

# Builds a "vocabulary" for each of track, artist, and album.
# Translates each _id into a continuous integer, these become indices in an embedding table
# Two indices are reserved and never map to a real ID:
#   index 0 = PAD. Playlists have variable length; when we batch them together in
#             PyTorch, short playlists are padded with 0 to mark an empty slot.
#             PAD is masked out during pooling so it contributes nothing.
#   index 1 = UNK. At encode() time, any ID not in the vocab (never seen in
#             training, or dropped by the frequency cutoff) maps here. UNK is a
#             single learnable "generic unseen item" vector, kept separate from
#             PAD so an empty slot and an unknown item don't share an embedding.
class Vocabulary:
    """Maps string IDs (track_id, artist_id, album_id) to contiguous integers.

    Index 0 = PAD (masked empty slot), index 1 = UNK (unseen / below-cutoff ID).
    Real IDs start at index 2.
    """
    PAD_INDEX = 0
    UNK_INDEX = 1
    NUM_RESERVED = 2  # reserved slots (PAD, UNK) preceding the first real ID

    def __init__(self, id_to_index, index_to_id):
        self.id_to_index = id_to_index
        self.index_to_id = index_to_id

    @classmethod # Allows the method to be called on the class itself. EX:
    # vocab = Vocabulary.build(ids), instead of 
    # vocab = Vocabulary(id_to_index, index_to_id), vocab.encode("abc")
    def build(cls, ids):
        """Build a vocabulary from a list of unique string IDs.

        Args:
            ids: list of unique string IDs (e.g., all track_ids from training data).

        Returns:
            Vocabulary with index 0 reserved for PAD and index 1 for UNK.
        """
        sorted_ids = sorted(set(ids)) # List of alphanumerically ordered unique string IDs
        id_to_index = {}
        index_to_id = {}
        for idx, string_id in enumerate(sorted_ids): # enumerate gives tuples (0, string1), (1, string2), etc.
            index = idx + cls.NUM_RESERVED # real IDs start after the PAD/UNK slots
            id_to_index[string_id] = index
            index_to_id[index] = string_id
        return cls(id_to_index, index_to_id) # same data with key/value flipped

    def encode(self, string_id):
        """Convert a string ID to its integer index. Returns UNK_INDEX (1) if unknown."""
        return self.id_to_index.get(string_id, self.UNK_INDEX)

    def encode_batch(self, string_ids):
        """Convert a list of string IDs to a list of integer indices."""
        return [self.id_to_index.get(sid, self.UNK_INDEX) for sid in string_ids]

    def decode(self, index):
        """Convert an integer index back to its string ID. Returns None if unknown."""
        return self.index_to_id.get(index, None)


    @property # Allows size to be called like .size, instead of .size()
    def size(self):
        """Total vocabulary size including the reserved PAD (0) and UNK (1) tokens."""
        return len(self.id_to_index) + self.NUM_RESERVED

    def save(self, path):
        """Save vocabulary to a JSON file."""
        path = Path(path)
        payload = {
            "id_to_index": self.id_to_index,
        }
        path.write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path):
        """Load vocabulary from a JSON file."""
        path = Path(path)
        payload = json.loads(path.read_text())
        id_to_index = payload["id_to_index"]
        # JSON keys are strings, but our indices should be integers
        index_to_id = {int(idx): string_id for string_id, idx in id_to_index.items()}
        return cls(id_to_index, index_to_id)

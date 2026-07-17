import json
from pathlib import Path

# Builds a "vocabulary" for each of track, artist, and album.
# Translates each _id into a continuous integer, these become indices in an embedding table
# 0 is masked, since playlists have variable lengths, when we bath them together in PyTorch, we need to ensure there is an index that we know is empty
# 0 is also masked so that during inference time, if a track appears that wasn't in our training data, we can encode it to 0 instead of causing an error
class Vocabulary:
    """Maps string IDs (track_id, artist_id, album_id) to contiguous integers.
    
    Index 0 is reserved for padding / unknown tokens. Real IDs start at index 1.
    """
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
            Vocabulary with index 0 reserved for padding/unknown.
        """
        sorted_ids = sorted(set(ids))
        id_to_index = {string_id: idx + 1 for idx, string_id in enumerate(sorted_ids)}
        index_to_id = {idx + 1: string_id for idx, string_id in enumerate(sorted_ids)}
        return cls(id_to_index, index_to_id)

    def encode(self, string_id):
        """Convert a string ID to its integer index. Returns 0 if unknown."""
        return self.id_to_index.get(string_id, 0)

    def encode_batch(self, string_ids):
        """Convert a list of string IDs to a list of integer indices."""
        return [self.id_to_index.get(sid, 0) for sid in string_ids]

    def decode(self, index):
        """Convert an integer index back to its string ID. Returns None if unknown."""
        return self.index_to_id.get(index, None)


    @property # Allows size to be called like .size, instead of .size()
    def size(self):
        """Total vocabulary size including the padding token at index 0."""
        return len(self.id_to_index) + 1

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

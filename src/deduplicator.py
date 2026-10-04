"""Near-duplicate removal через MinHash LSH."""
from datasketch import MinHash, MinHashLSH
from typing import List
from src.schemas import IntentSample


def _minhash(text: str, num_perm: int = 128) -> MinHash:
    m = MinHash(num_perm=num_perm)
    for token in text.lower().split():
        m.update(token.encode("utf8"))
    return m


def deduplicate(samples: List[IntentSample], threshold: float = 0.8) -> List[IntentSample]:
    """Убирает near-duplicates по Jaccard similarity."""
    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    keep = []
    for i, s in enumerate(samples):
        mh = _minhash(s.text)
        if lsh.query(mh):
            continue  # уже есть похожий
        lsh.insert(f"s_{i}", mh)
        keep.append(s)
    return keep

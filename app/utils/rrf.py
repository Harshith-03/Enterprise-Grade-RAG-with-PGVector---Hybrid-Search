"""Reciprocal Rank Fusion helper."""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple


def reciprocal_rank_fusion(
    rankings: Dict[str, Sequence[Tuple[str, float]]],
    k: int = 60,
) -> List[Tuple[str, float, str]]:
    """Fuse multiple ranked lists into a single ranking.

    Args:
        rankings: Mapping source -> list of (chunk_id, score).
        k: Hyper-parameter controlling the level of dampening.
    Returns:
        List of tuples (chunk_id, fused_score, provenance_source) sorted by score.
    """

    accumulator: Dict[str, float] = defaultdict(float)
    provenance: Dict[str, str] = {}

    for source, items in rankings.items():
        for rank, (chunk_id, _score) in enumerate(items, start=1):
            accumulator[chunk_id] += 1.0 / (k + rank)
            provenance.setdefault(chunk_id, source)

    return sorted(
        ((chunk_id, score, provenance[chunk_id]) for chunk_id, score in accumulator.items()),
        key=lambda item: item[1],
        reverse=True,
    )

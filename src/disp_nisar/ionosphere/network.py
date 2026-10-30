"""Pair selection for the two-band network in forward mode (requirement R-F3).

In historical mode the A and B interferogram networks are identical. In forward mode A's
stack carries the previous ministacks' compressed SLCs while B is run fresh, so A has
pairs with compressed references that B does not have, and compressed epochs have no raw
GSLC mask. The double-difference D is needed only on pairs whose two epochs are real
acquisitions, and the output product (second-to-last → last epoch) is such a pair. So
the common network is the intersection of the two networks restricted to real-date
pairs, and it must still connect every output epoch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from .inversion import build_design_matrix

__all__ = ["PairSelection", "select_pairs"]


@dataclass(frozen=True)
class PairSelection:
    """The common real-date pairs of A and B, and what was dropped to get them."""

    pairs: list  # sorted common real-date pairs
    dropped_a_only: list = field(default_factory=list)
    dropped_b_only: list = field(default_factory=list)
    dropped_compressed: list = field(default_factory=list)

    @property
    def summary(self) -> str:
        """One log line with the kept and dropped pair counts."""
        return (
            f"{len(self.pairs)} common real-date pairs; dropped"
            f" {len(self.dropped_a_only)} A-only, {len(self.dropped_b_only)} B-only,"
            f" {len(self.dropped_compressed)} with a compressed epoch"
        )


def _connected(pairs: Sequence[tuple], epochs: Iterable) -> bool:
    """Check that every epoch in `epochs` lies in one component of the pair graph."""
    epochs = list(epochs)
    if not epochs:
        return True
    adj: dict = {}
    for a, b in pairs:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    if any(e not in adj for e in epochs):
        return False
    seen, stack = set(), [epochs[0]]
    while stack:
        n = stack.pop()
        if n in seen:
            continue
        seen.add(n)
        stack.extend(adj[n] - seen)
    return all(e in seen for e in epochs)


def select_pairs(
    pairs_a: Iterable[tuple],
    pairs_b: Iterable[tuple],
    real_dates: Iterable,
    output_pairs: Iterable[tuple] = (),
    strict: bool = False,
) -> PairSelection:
    """Select the common network of A and B on real-date pairs.

    Raises ValueError when that network cannot serve the output epochs.

    strict=True reproduces the historical behaviour (the two networks must match
    exactly).
    """
    sa, sb, real = set(pairs_a), set(pairs_b), set(real_dates)
    if strict and sa != sb:
        raise ValueError("A/B IFG and correlation networks must match exactly")
    common = sa & sb
    keep = sorted(p for p in common if p[0] in real and p[1] in real)
    sel = PairSelection(
        pairs=keep,
        dropped_a_only=sorted(sa - sb),
        dropped_b_only=sorted(sb - sa),
        dropped_compressed=sorted(
            p for p in common if not (p[0] in real and p[1] in real)
        ),
    )
    if not keep:
        raise ValueError("No common real-date A/B pairs: the two-band network is empty")
    out = list(output_pairs)
    need = {d for p in out for d in p}
    missing = need - {d for p in keep for d in p}
    if missing:
        raise ValueError(
            f"Output epochs absent from the common real-date network: {sorted(missing)}"
        )
    if not _connected(keep, need):
        raise ValueError(
            "Output epochs are not connected through the common real-date network"
        )
    dates = sorted({d for p in keep for d in p})
    if np.linalg.matrix_rank(build_design_matrix(keep, dates)) != len(dates) - 1:
        raise ValueError("Common real-date A/D network is disconnected")
    return sel

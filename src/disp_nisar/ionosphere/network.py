"""Pair selection for the two-band network.

In historical mode the A and B interferogram networks are identical. In forward mode
A's stack carries compressed SLCs from earlier ministacks while B is phase-linked fresh
on the batch's real dates, so A has pairs B lacks, and compressed epochs have no GSLC
mask. D is only needed on pairs of two real acquisitions, and the output products are
such pairs, so the network is the two bands' common real-date pairs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable

import numpy as np

from .inversion import build_design_matrix

DatePair = tuple[datetime, datetime]


@dataclass(frozen=True)
class PairSelection:
    """The common real-date pairs of A and B, and what was dropped to get them."""

    pairs: list[DatePair]
    dropped_a_only: list[DatePair] = field(default_factory=list)
    dropped_b_only: list[DatePair] = field(default_factory=list)
    dropped_compressed: list[DatePair] = field(default_factory=list)

    @property
    def summary(self) -> str:
        """One log line with the kept and dropped pair counts."""
        return (
            f"{len(self.pairs)} common real-date pairs; dropped"
            f" {len(self.dropped_a_only)} A-only, {len(self.dropped_b_only)} B-only,"
            f" {len(self.dropped_compressed)} with a compressed epoch"
        )


def select_pairs(
    pairs_a: Iterable[DatePair],
    pairs_b: Iterable[DatePair],
    real_dates: Iterable[datetime],
    output_pairs: Iterable[DatePair] = (),
    strict: bool = True,
) -> PairSelection:
    """Select the common network of A and B on real-date pairs.

    Parameters
    ----------
    pairs_a, pairs_b : Iterable[DatePair]
        The interferogram pairs of each band.
    real_dates : Iterable[datetime]
        Dates of real acquisitions, as opposed to compressed SLCs.
    output_pairs : Iterable[DatePair]
        Pairs the network must be able to serve.
    strict : bool
        Require the two networks to match exactly, as in historical runs.

    Returns
    -------
    PairSelection
        The sorted common pairs and the dropped ones.

    Raises
    ------
    ValueError
        If the networks differ under `strict`, or the common network is empty,
        misses an output epoch, or is disconnected.

    """
    set_a, set_b, real = set(pairs_a), set(pairs_b), set(real_dates)
    if strict and set_a != set_b:
        raise ValueError("A/B IFG and correlation networks must match exactly")
    common = set_a & set_b
    keep = sorted(p for p in common if p[0] in real and p[1] in real)
    if not keep:
        raise ValueError("No common real-date A/B pairs: the two-band network is empty")
    dates = sorted({d for pair in keep for d in pair})
    missing = {d for pair in output_pairs for d in pair} - set(dates)
    if missing:
        raise ValueError(f"Output requests epochs absent from the network: {missing}")
    if np.linalg.matrix_rank(build_design_matrix(keep, dates)) != len(dates) - 1:
        raise ValueError("Common real-date A/D network is disconnected")
    return PairSelection(
        pairs=keep,
        dropped_a_only=sorted(set_a - set_b),
        dropped_b_only=sorted(set_b - set_a),
        dropped_compressed=sorted(set(common) - set(keep)),
    )

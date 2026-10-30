"""The two-band pair network in historical and forward mode."""

from datetime import datetime

import pytest

from disp_nisar.ionosphere.network import select_pairs

DATES = [
    datetime(2026, 1, 1),
    datetime(2026, 1, 13),
    datetime(2026, 1, 25),
    datetime(2026, 2, 6),
    datetime(2026, 2, 18),
]
COMPRESSED = datetime(2025, 12, 1)


def _network(dates, bandwidth=2):
    return [
        (dates[i], dates[j])
        for i in range(len(dates))
        for j in range(i + 1, min(i + 1 + bandwidth, len(dates)))
    ]


def test_identical_networks_pass_through():
    pairs = _network(DATES)
    selection = select_pairs(pairs, pairs, DATES, [(DATES[0], d) for d in DATES[1:]])
    assert selection.pairs == sorted(pairs)
    assert not selection.dropped_a_only
    assert not selection.dropped_compressed


def test_strict_mode_rejects_a_mismatch():
    pairs = _network(DATES)
    with pytest.raises(ValueError, match="must match exactly"):
        select_pairs(pairs, pairs[:-1], DATES)


def test_forward_mode_uses_the_common_real_date_network():
    compressed = [(COMPRESSED, d) for d in DATES]
    selection = select_pairs(
        compressed + _network(DATES),
        _network(DATES),
        DATES,
        [(DATES[-2], DATES[-1])],
        strict=False,
    )
    assert selection.pairs == sorted(_network(DATES))
    assert selection.dropped_a_only == sorted(compressed)
    assert not selection.dropped_b_only
    assert "5 A-only" in selection.summary


def test_compressed_pairs_in_both_bands_are_dropped():
    compressed = [(COMPRESSED, d) for d in DATES]
    pairs = compressed + _network(DATES)
    selection = select_pairs(pairs, pairs, DATES, [(DATES[-2], DATES[-1])])
    assert selection.dropped_compressed == sorted(compressed)
    assert selection.pairs == sorted(_network(DATES))


def test_network_must_connect():
    pairs = _network(DATES[:3]) + _network(DATES[3:])
    with pytest.raises(ValueError, match="disconnected"):
        select_pairs(pairs, pairs, DATES, [(DATES[0], DATES[-1])])


def test_output_epoch_must_be_in_the_network():
    pairs = _network(DATES[:3])
    with pytest.raises(ValueError, match="absent"):
        select_pairs(pairs, pairs, DATES, [(DATES[0], DATES[-1])])


def test_empty_common_network_raises():
    with pytest.raises(ValueError, match="empty"):
        select_pairs(_network(DATES[:3]), _network(DATES[3:]), DATES, strict=False)

from datetime import datetime

import pytest

from disp_nisar.ionosphere.network import select_pairs

D = [datetime(2026, 1, d) for d in (1, 13, 25)] + [
    datetime(2026, 2, 6),
    datetime(2026, 2, 18),
]
C = datetime(2025, 12, 1)  # a compressed-SLC "epoch"


def _net(dates, k=2):
    return [
        (dates[i], dates[j])
        for i in range(len(dates))
        for j in range(i + 1, min(i + 1 + k, len(dates)))
    ]


def test_historical_identical_networks_pass_through():
    a = _net(D)
    sel = select_pairs(a, a, D, output_pairs=[(D[0], d) for d in D[1:]])
    assert (
        sel.pairs == sorted(a) and not sel.dropped_a_only and not sel.dropped_compressed
    )


def test_strict_mode_rejects_a_mismatch():
    a = _net(D)
    with pytest.raises(ValueError, match="match exactly"):
        select_pairs(a, a[:-1], D, strict=True)


def test_forward_mode_uses_the_common_real_date_network():
    # A: compressed reference C paired with every real date, plus real-date pairs; B:
    # real-date pairs only
    a = [(C, d) for d in D] + _net(D)
    b = _net(D)
    sel = select_pairs(a, b, D, output_pairs=[(D[-2], D[-1])])
    assert sel.pairs == sorted(_net(D))
    assert sel.dropped_a_only == sorted((C, d) for d in D)
    assert not sel.dropped_b_only
    assert "5 A-only" in sel.summary


def test_compressed_pairs_present_in_both_bands_are_dropped():
    a = [(C, d) for d in D] + _net(D)
    sel = select_pairs(a, a, D, output_pairs=[(D[-2], D[-1])])
    assert sel.dropped_compressed == sorted((C, d) for d in D)
    assert sel.pairs == sorted(_net(D))


def test_output_epoch_must_be_connected():
    a = _net(D[:3]) + _net(D[3:])  # two islands in time
    with pytest.raises(ValueError, match="not connected|disconnected"):
        select_pairs(a, a, D, output_pairs=[(D[0], D[-1])])
    with pytest.raises(ValueError, match="absent"):
        select_pairs(_net(D[:3]), _net(D[:3]), D, output_pairs=[(D[0], D[-1])])


def test_empty_common_network_raises():
    with pytest.raises(ValueError, match="empty"):
        select_pairs(_net(D[:3]), _net(D[3:]), D)

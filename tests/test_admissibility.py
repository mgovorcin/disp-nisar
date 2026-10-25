import pytest

from disp_nisar.ionosphere.admissibility import check_admissibility, parse_granule

G = (
    "NISAR_L2_PR_GSLC_{n:03d}_042_D_069_4005_{pol}_A_{d}T025606_{d}T025641_{crid}"
    "_N_F_J_001.h5"
)


def gr(d, crid="P05023", pol="DHDH", n=4):
    return G.format(n=n, d=d, crid=crid, pol=pol)


def test_parse_granule():
    e, crid, mode = parse_granule(gr("20251101"))
    assert (e, crid, mode) == ("20251101", "P05023", "4005_DHDH")


def test_clean_stack_has_no_flags():
    a = [gr(d) for d in ("20251101", "20251113", "20251125", "20251207")]
    recs, flags = check_admissibility(
        a, a, center_frequencies=lambda _p: (1239.0e6, 1293.5e6)
    )
    assert len(recs) == 4 and flags == []


def test_missing_b_and_crid_change_and_mode_change():
    dates = ("20251101", "20251113", "20251125", "20251207", "20251219")
    a = [gr(d) for d in dates[:-1]] + [gr(dates[-1], crid="P05030")]
    b = [gr(d) for d in dates if d != "20251113"]

    def freqs(p):
        return (1257.5e6, 1293.5e6) if "20251125" in str(p) else (1239.0e6, 1293.5e6)

    recs, flags = check_admissibility(a, b, center_frequencies=freqs)
    by = {(f.epoch, f.reason): f for f in flags}
    assert by[("20251113", "missing_B")].reject
    assert (
        by[("20251125", "mode_change")].reject
        and "f_A 1257.500" in by[("20251125", "mode_change")].detail
    )
    assert not by[("20251219", "crid_change")].reject
    assert len(flags) == 3


def test_single_band_mode_reads_as_missing_b():
    dates = ("20251101", "20251113", "20251125", "20251207")
    a = [gr(d) for d in dates]

    def freqs(p):
        if "20251207" in str(p):
            raise KeyError("frequencyB")
        return (1239.0e6, 1293.5e6)

    # stub: the first-or-none fallback also calls freqs(), which raises -> cannot read
    # -> error is clear
    with pytest.raises(ValueError, match="Cannot read centre frequencies"):
        check_admissibility(a, a, center_frequencies=freqs)


def test_duplicate_and_too_few():
    a = [gr("20251101"), gr("20251101", n=5)]
    with pytest.raises(ValueError, match="Duplicate frequency-A"):
        check_admissibility(a, a)
    a = [gr(d) for d in ("20251101", "20251113", "20251125")]
    with pytest.raises(ValueError, match="admissible two-band epochs"):
        check_admissibility(
            a, [gr("20251101")], center_frequencies=lambda _p: (1239.0e6, 1293.5e6)
        )

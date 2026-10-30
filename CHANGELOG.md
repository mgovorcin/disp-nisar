# Unreleased

**Added**

- Initial workflows ported over time dolphin

**Changed**

- Opt-in `ionosphere_options.method: main_diff` estimates the ionosphere from
  frequency A and A*conj(B). The default `main_side`
  keeps the split-spectrum workflow.
- `main_diff` unwraps A*conj(B) on a grid `ionosphere_options.d_looks` times coarser
  than the B grid (default 10), which stops most gross cycle errors. Set it to 1 for
  native unwrapping.
- `main_diff` runs in forward mode on the common real-date pairs of A and B, with B
  phase-linked only on the real acquisitions.
- `filtering.method: robust` smooths the `main_diff` screen with background clipping,
  empirical noise weights, a support floor and an 8 km near-range exclusion. The
  default filter is unchanged.

**Fixed**

- Split-spectrum corrections were scaled by lambda/4pi after already being written
  in meters, leaving the ionosphere layer about 52 times too small.
- The validity mask is now built for runs without an ancillary water mask.

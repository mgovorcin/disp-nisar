"""DISP-specific settings for optional main-differential estimation."""

from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class IonosphereOptions(BaseModel):
    """Retain the existing estimator by default; main_diff is opt-in."""

    model_config = ConfigDict(extra="forbid")
    method: Literal["main_side", "main_diff"] = "main_side"
    mask_reduction: Literal["all_valid", "nearest"] = "all_valid"
    correlation_threshold: float = Field(0.2, gt=0, le=1)
    similarity_threshold: float = Field(0.4, gt=0, le=1)
    min_valid_fraction: float = Field(1.0, gt=0, le=1)
    min_phasor_magnitude: float = Field(0.1, gt=0, le=1)
    block_size: int = Field(256, ge=16, le=2048)
    mask_work_mb: int = Field(128, ge=16)
    nlooks: float = Field(
        5.0,
        gt=0,
        description=(
            "Effective looks for the correlation proxy; validate for the chosen PL"
            " settings."
        ),
    )
    smoothing: Literal["legacy", "robust"] = "legacy"
    smooth_sigma_px: float = Field(
        50.0,
        gt=0,
        description="Screen-scale kernel sigma on the B grid (50 px = 2 km at 40 m).",
    )
    smooth_min_support: float = Field(
        0.2,
        gt=0,
        le=1,
        description="Kernel support below which the coarse kernel is used.",
    )
    smooth_fallback_sigma_px: float = Field(
        250.0, gt=0, description="Coarse kernel sigma (250 px = 10 km at 40 m)."
    )
    noise_weights: bool = Field(
        True, description="Weight the smoothing by 1/(empirical per-pixel noise)^2."
    )
    incidence_path: Optional[Path] = Field(
        None,
        description=(
            "Incidence-angle raster (static layers) used to locate the near-range band;"
            " set by the workflow."
        ),
    )
    near_range_exclude_km: float = Field(
        8.0,
        ge=0,
        description=(
            "Width of the near-range band excluded from the screen (filled from the"
            " interior); needs an incidence raster; 0 disables."
        ),
    )
    exclude_noise_factor: float = Field(
        3.0,
        gt=0,
        description=(
            "Pixels noisier than this × the median empirical noise are excluded, not"
            " just down-weighted; 0 disables."
        ),
    )
    background_clip_cycles: float = Field(
        0.2,
        ge=0,
        description=(
            "Drop pixels further than this (cycles of the screen at A) from the 10 km"
            " background; 0 disables."
        ),
    )
    background_block_px: int = Field(
        250,
        ge=10,
        description="Block size of the background median (250 px = 10 km at 40 m).",
    )
    resume: bool = True
    keep_intermediates: bool = False

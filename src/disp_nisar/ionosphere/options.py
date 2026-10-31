"""DISP-specific settings for optional main-differential estimation."""

from typing import Literal

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
    main_side_recipe: Literal["stock", "improved"] = Field(
        "stock",
        description=(
            "With method main_side: 'stock' keeps the original pixel-referenced 150 m"
            " estimate; 'improved' is the area-referenced, masked, consistency-checked"
            " estimate smoothed at the ionosphere's scale."
        ),
    )
    main_side_sigma_km: float = Field(
        5.0, gt=0, description="Improved main_side: smoothing kernel sigma."
    )
    main_side_consistency_km: float = Field(
        10.0,
        gt=0,
        description=(
            "Improved main_side: block size of the band-consistency background."
        ),
    )
    resume: bool = True
    keep_intermediates: bool = False

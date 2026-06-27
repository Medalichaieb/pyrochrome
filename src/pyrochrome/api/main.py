"""FastAPI prediction service (skeleton).

Contract:
    POST /predict  {chemistry (UMF oxides), cone, atmosphere}
        → {surface, transparency, colour ± confidence, nearest real recipes}

Honesty rule (brief §7): never return a single, falsely-precise colour. Always
return a confidence index, a colour range, and real example tiles. Cone and
atmosphere are required inputs.

Run locally:
    uv run uvicorn pyrochrome.api.main:app --reload
"""

from __future__ import annotations

from enum import StrEnum

from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(
    title="Pyrochrome API",
    summary="Predict the post-firing render of a ceramic glaze.",
    version="0.1.0",
)


class Atmosphere(StrEnum):
    """Firing atmosphere — a required input (drives redox-sensitive colours)."""

    OXIDATION = "oxidation"
    REDUCTION = "reduction"
    NEUTRAL = "neutral"


class PredictRequest(BaseModel):
    """Prediction request: glaze chemistry plus firing conditions."""

    chemistry_umf: dict[str, float] = Field(
        ..., description="UMF oxide amounts, e.g. {'SiO2_umf': 3.2, 'Al2O3_umf': 0.4, ...}"
    )
    cone: str = Field(..., description="Orton cone label, e.g. '6' or '05.5'.")
    atmosphere: Atmosphere = Field(..., description="Required firing atmosphere.")


class ColourPrediction(BaseModel):
    """Predicted colour with an honest confidence and range."""

    family: str = Field(..., description="Predicted colour family (top-1).")
    family_top2: list[str] = Field(default_factory=list, description="Top-2 families.")
    lab: list[float] | None = Field(None, description="Predicted CIELAB colour [L, a, b].")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence index in [0, 1].")


class PredictResponse(BaseModel):
    """Prediction response with confidence and real nearest recipes."""

    surface: str
    transparency: str
    colour: ColourPrediction
    neighbours: list[dict[str, object]] = Field(
        default_factory=list, description="Nearest real Glazy recipes (id, name, photo)."
    )


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """Predict the render for a glaze (skeleton).

    TODO: load the trained models + k-NN index and return real predictions.
    """
    raise NotImplementedError(
        "Prediction is not wired to the trained models yet. See models.baseline / models.export."
    )

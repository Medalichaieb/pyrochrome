"""FastAPI prediction service.

Contract:
    POST /predict  {chemistry (UMF oxides), cone, atmosphere}
        → {surface, transparency, colour (± confidence, top-2, Lab), neighbours}

Honesty rule (brief §7): never return a single, falsely-precise colour. We return
a colour family with confidence and top-2, a Lab value, and the nearest real
recipes. Cone and atmosphere are required inputs.

The trained artifacts (``models_out/``) are loaded once at startup via
:class:`pyrochrome.models.inference.Predictor`. Run them first with
``make train && make color && make neighbors``.

Run locally:
    uv run uvicorn pyrochrome.api.main:app --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from pyrochrome.models.inference import Predictor

# Local dev origins for the Vite frontend. Override/extend for deployment.
CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:4173",
    "http://127.0.0.1:4173",
]

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

# Loaded once at startup and reused across requests.
_state: dict[str, Predictor] = {}


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Load the persisted models on startup (tolerant of missing artifacts)."""
    _state["predictor"] = Predictor.load()
    yield
    _state.clear()


app = FastAPI(
    title="Pyrochrome API",
    summary="Predict the post-firing render of a ceramic glaze.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class Atmosphere(StrEnum):
    """Firing atmosphere — a required input (drives redox-sensitive colours)."""

    OXIDATION = "oxidation"
    REDUCTION = "reduction"
    NEUTRAL = "neutral"


class PredictRequest(BaseModel):
    """Prediction request: glaze chemistry plus firing conditions."""

    chemistry_umf: dict[str, float] = Field(
        ...,
        description="UMF oxide amounts, e.g. {'SiO2': 3.2, 'Al2O3': 0.4, 'CuO': 0.05}. "
        "Keys may be bare oxides or the '<oxide>_umf' column names.",
    )
    cone: str = Field(..., description="Orton cone label, e.g. '6' or '05.5'.")
    atmosphere: Atmosphere = Field(..., description="Required firing atmosphere.")


class ClassPrediction(BaseModel):
    """A classification prediction with confidence and top-2."""

    label: str
    top2: list[str] = Field(default_factory=list)
    confidence: float = Field(..., ge=0.0, le=1.0)


class ColourPrediction(ClassPrediction):
    """Colour family prediction, plus a predicted CIELAB value."""

    lab: list[float] | None = Field(None, description="Predicted CIELAB [L*, a*, b*].")


class PredictResponse(BaseModel):
    """Prediction response with confidence and nearest real recipes."""

    surface: ClassPrediction | None = None
    transparency: ClassPrediction | None = None
    colour: ColourPrediction | None = None
    neighbours: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Nearest real Glazy recipes (id, name, rgb, lab, distance).",
    )


@app.get("/health")
def health() -> dict[str, Any]:
    """Liveness probe reporting which models are loaded."""
    predictor = _state.get("predictor")
    loaded = sorted(predictor.classifiers) if predictor else []
    return {
        "status": "ok",
        "classifiers": loaded,
        "colour_lab": bool(predictor and predictor.colour_lab),
        "neighbours": bool(predictor and predictor.neighbors),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """Predict the render for a glaze from its chemistry and firing conditions."""
    predictor = _state.get("predictor")
    if predictor is None or not predictor.classifiers:
        raise HTTPException(
            status_code=503,
            detail="Models not available. Run `make train && make color && make neighbors`.",
        )
    result = predictor.predict(request.chemistry_umf, request.cone, request.atmosphere.value)
    return PredictResponse(**result)

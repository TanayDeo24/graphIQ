"""GraphIQ FastAPI service layer.

Every route queries precomputed result tables in Postgres (populated by
src/models/attrition/evaluate.py and src/models/spend/evaluate.py) — no
route computes a metric at request time.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import attrition, cross_component, headline, spend

app = FastAPI(
    title="GraphIQ API",
    description=(
        "Unified-schema workforce analytics demo: attrition risk (survival analysis) and "
        "spend anomaly detection, both reading from the same employees table. Methodology "
        "demonstration only — see /docs and the project README for data provenance and scope."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

app.include_router(attrition.router, prefix="/api/attrition", tags=["attrition"])
app.include_router(spend.router, prefix="/api/spend", tags=["spend"])
app.include_router(cross_component.router, prefix="/api/cross-component", tags=["cross-component"])
app.include_router(headline.router, prefix="/api/headline", tags=["headline"])


@app.get("/")
def root():
    return {
        "name": "GraphIQ API",
        "docs": "/docs",
        "scope_disclaimer": (
            "Demonstrates methodology and evaluation rigor on public/synthetic data. Not a "
            "real finding about any company. Counterfactual sensitivity results are "
            "correlational, never causal."
        ),
    }

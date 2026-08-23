"""GraphIQ FastAPI service layer.

Every route queries precomputed result tables in Postgres (populated by
src/models/attrition/evaluate.py and src/models/spend/evaluate.py) — no
route computes a metric at request time.
"""

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import agent, attrition, cross_component, headline, spend

app = FastAPI(
    title="GraphIQ API",
    description=(
        "Unified-schema workforce analytics demo: attrition risk (survival analysis) and "
        "spend anomaly detection, both reading from the same employees table. Methodology "
        "demonstration only — see /docs and the project README for data provenance and scope."
    ),
    version="1.0.0",
)

# CORS_ALLOWED_ORIGIN: the deployed Cloudflare Workers dashboard domain in
# production (set in Render's env vars, never hardcoded here since it's
# only known after the Workers deployment exists). Local dev origins are
# always allowed in addition, so `npm run dev` keeps working unconfigured.
_LOCAL_DEV_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]
_prod_origin = os.environ.get("CORS_ALLOWED_ORIGIN")
_allowed_origins = _LOCAL_DEV_ORIGINS + ([_prod_origin] if _prod_origin else [])

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(attrition.router, prefix="/api/attrition", tags=["attrition"])
app.include_router(spend.router, prefix="/api/spend", tags=["spend"])
app.include_router(cross_component.router, prefix="/api/cross-component", tags=["cross-component"])
app.include_router(headline.router, prefix="/api/headline", tags=["headline"])
app.include_router(agent.router, prefix="/api/agent", tags=["agent"])


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

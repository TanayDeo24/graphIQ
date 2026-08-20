import pandas as pd
from fastapi import APIRouter

from src.db.connection import get_engine

router = APIRouter()


@router.get("/quadrant")
def quadrant():
    """Per-employee attrition-risk x spend-anomaly quadrant assignments,
    the correlation/significance summary, and per-quadrant department/
    tenure characteristics. See src/analysis/cross_component.py for
    methodology. Correlational only — the summary's `disclaimer` field
    states this explicitly in the payload itself, not just in docs."""
    engine = get_engine()
    quadrant_df = pd.read_sql("SELECT * FROM cross_component_quadrant", engine)
    summary_df = pd.read_sql("SELECT * FROM cross_component_summary ORDER BY id DESC LIMIT 1", engine)
    characteristics_df = pd.read_sql("SELECT * FROM cross_component_quadrant_characteristics", engine)

    summary = summary_df.to_dict(orient="records")[0] if not summary_df.empty else None
    quadrant_counts = quadrant_df["quadrant"].value_counts().to_dict() if not quadrant_df.empty else {}

    return {
        "summary": summary,
        "quadrant_counts": quadrant_counts,
        "employees": quadrant_df.to_dict(orient="records"),
        "characteristics": characteristics_df.to_dict(orient="records"),
    }

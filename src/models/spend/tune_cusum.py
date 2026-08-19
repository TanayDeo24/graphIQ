"""One-time CUSUM h-threshold tuning, per the mandatory anti-overfitting
procedure documented in the README's "Spend" results section:

1. Generate a separate tuning dataset (same methodology, different seed)
   that no reported result has ever been evaluated against.
2. Search h using ONLY the tuning dataset.
3. Apply the selected h to the real, already-existing test dataset
   (data/generated/expense_transactions_5pct.csv) exactly once, and
   report that as final -- see evaluate.py's run(), which imports
   H_SIGMA_TUNED from this module rather than re-deriving it, so the
   search never re-runs against the reported-on test data.

Honest note on how the search range was chosen: the initial hypothesis,
grounded in the effect-size diagnosis (median ~1.8 clean-baseline-sigma
per drift-month; after the robust-estimator fix in cusum.py, effective
internal signal is roughly that divided by the measured ~1.37x residual
contamination ratio), was that h=5 might be too CONSERVATIVE for a
typical 4-8 month drift span, and a LOWER h (tested: 2.0-5.0) would help.
That hypothesis was wrong: F1 on the tuning dataset increased
monotonically across that entire range, with h=5 (the unchanged original)
outperforming every lower candidate tried. Extending the search upward
(tested: 5.0-20.0) found F1 peaks somewhere on a broad, gentle plateau
roughly spanning h=12-17 (not a sharp/unstable spike) -- i.e. the data
says the *opposite* of the original hypothesis: ordinary month-to-month
spend volatility accumulates enough spurious C+ signal under the null
that a much STRICTER threshold, not a looser one, is what actually
improves precision on the (rare, ~1.67% of transactions) slow_drift
class enough to help F1, despite the recall cost. Reported as measured,
not adjusted to match the original hypothesis.

Also reported honestly: re-running this exact search twice (same
TUNING_SEED_OFFSET) produced slightly different generated datasets and
slightly different exact peak locations within that plateau (16.5 the
first run, 12.0 the second) -- traced to spend_generator.py's hot-
employee pool being materialized via list(some_set), whose iteration
order depends on Python's per-process hash randomization and isn't
pinned by the RNG seed alone. A real, minor, pre-existing reproducibility
gap, out of scope to fix as part of this request. The plateau's location
(roughly h=12-17) was consistent across both runs even though its exact
peak wasn't, which is why H_SIGMA_TUNED below is chosen from the middle
of that range rather than either run's specific maximum.

This is a standalone script, not invoked by the main pipeline -- run it
once, read off the result, and the chosen constant is hardcoded into
H_SIGMA_TUNED below with the reasoning that produced it.
"""

import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from src.generation.spend_generator import generate_all as generate_spend_all
from src.models.spend import cusum
from src.models.spend.features import build_feature_frame

TUNING_OUT_DIR = "data/generated/_cusum_tuning"
TUNING_SEED_OFFSET = 9001  # arbitrary, far from any seed used elsewhere in this project

# k is left at 0.5 (unchanged) -- only h is searched; see module docstring
# for why this range ended up spanning far wider than originally hypothesized.
H_CANDIDATES = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 13.5, 15.0, 16.0, 16.5, 17.0, 20.0]

# Final choice: h=14.0, the middle of the ~12-17 plateau observed across
# two independent tuning-dataset generations (see module docstring) --
# deliberately not either run's specific maximum, since the exact peak
# moved between runs but the plateau's location didn't.
H_SIGMA_TUNED = 14.0


def run_tuning() -> dict:
    import src.generation.spend_generator as sg

    original_seed = sg.SEED
    try:
        sg.SEED = original_seed + TUNING_SEED_OFFSET
        stats = generate_spend_all(employees_path="data/generated/employees.csv", out_dir=TUNING_OUT_DIR)
    finally:
        sg.SEED = original_seed

    tuning_df = pd.read_csv(f"{TUNING_OUT_DIR}/expense_transactions_5pct.csv", parse_dates=["transaction_date"])
    featured = build_feature_frame(tuning_df)
    monthly = cusum.compute_cusum_flags(featured)

    # PR-AUC is threshold-free (it integrates over every possible cutoff of the
    # underlying continuous statistic, which h does not change) -- it cannot
    # distinguish between h candidates at all. F1 at the h-implied operating
    # point is the right metric for tuning an actual decision threshold.
    y_true = (featured["anomaly_type"] == "slow_drift").astype(int).values
    results = {}
    for h in H_CANDIDATES:
        flagged = (monthly["cusum_statistic"] > h).astype(int)
        mapped = cusum.map_flags_to_transactions(featured, monthly.assign(flagged=flagged.astype(bool)))
        y_pred = mapped["flagged"].astype(int).values
        results[h] = {
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "alerts_per_1000_txns": float(y_pred.mean() * 1000),
        }

    best_h = max(results, key=lambda h: results[h]["f1"])
    return {"generation_stats": stats, "results_by_h": results, "best_h": best_h}


if __name__ == "__main__":
    outcome = run_tuning()
    print("Tuning dataset generation stats:", outcome["generation_stats"])
    print()
    print("h | precision | recall | F1 | alerts/1000 txns")
    for h, r in outcome["results_by_h"].items():
        print(f"{h:.1f} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} | {r['alerts_per_1000_txns']:.2f}")
    print()
    print(f"Best h on tuning dataset (by F1): {outcome['best_h']}")

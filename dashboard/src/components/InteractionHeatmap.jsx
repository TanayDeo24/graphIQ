import { Fragment } from "react";

const TENURE_BANDS = ["0-2", "2-5", "5+"];
const TREND_BUCKETS = ["declining", "stable", "improving"];

function riskColor(value, min, max) {
  const t = max > min ? (value - min) / (max - min) : 0.5;
  const light = [252, 224, 214]; // low risk: light warm
  const dark = [178, 34, 34]; // high risk: deep red
  const rgb = light.map((l, i) => Math.round(l + (dark[i] - l) * Math.min(1, Math.max(0, t))));
  return `rgb(${rgb.join(",")})`;
}

export default function InteractionHeatmap({ data }) {
  if (!data || data.length === 0) {
    return <div className="loading">Loading...</div>;
  }

  const populated = data.filter((d) => !d.low_confidence && d.mean_gbm_risk_score != null);
  const values = populated.map((d) => d.mean_gbm_risk_score);
  const min = Math.min(...values);
  const max = Math.max(...values);

  const cellFor = (band, bucket) => data.find((d) => d.tenure_band === band && d.review_trend_bucket === bucket);

  return (
    <div>
      <div className="calibration-grid" style={{ gridTemplateColumns: `110px repeat(${TREND_BUCKETS.length}, 1fr)` }}>
        <div />
        {TREND_BUCKETS.map((b) => (
          <div key={b} style={{ fontSize: 11, color: "var(--text-muted)", textAlign: "center" }}>
            {b}
          </div>
        ))}
        {TENURE_BANDS.map((band) => (
          <Fragment key={band}>
            <div style={{ fontSize: 12, alignSelf: "center" }}>{band} yrs</div>
            {TREND_BUCKETS.map((bucket) => {
              const cell = cellFor(band, bucket);
              const lowConf = !cell || cell.low_confidence;
              return (
                <div
                  key={`${band}-${bucket}`}
                  className="calibration-cell"
                  style={{
                    background: lowConf ? "repeating-linear-gradient(45deg, #e5e4df, #e5e4df 4px, #f2f1ec 4px, #f2f1ec 8px)" : riskColor(cell.mean_gbm_risk_score, min, max),
                    color: lowConf ? "var(--text-muted)" : "#fff",
                    border: lowConf ? "1px dashed var(--border)" : "none",
                  }}
                  title={
                    lowConf
                      ? `Low confidence: n=${cell ? cell.n : 0} (below the n=10 threshold) — not colored as if reliable.`
                      : `n=${cell.n}, mean GBM risk score=${cell.mean_gbm_risk_score.toFixed(3)}`
                  }
                >
                  {lowConf ? `n=${cell ? cell.n : 0}` : cell.mean_gbm_risk_score.toFixed(2)}
                </div>
              );
            })}
          </Fragment>
        ))}
      </div>
      <p style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 10 }}>
        Cell = mean baseline GBM risk score (darker = higher risk). Hatched/grey cells have n &lt; 10 and
        are not colored as if reliable — hover for the count. Note: the 5+ tenure band shows nearly
        identical average risk across all three review-trend buckets — the model's predictions for that
        population are dominated by tenure band, a real (if unglamorous) finding, not a rendering bug.
      </p>
    </div>
  );
}

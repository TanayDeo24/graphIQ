const ANOMALY_TYPE_COLS = ["point_spike", "slow_drift", "coordinated_pattern", "overall"];

// Sequential single-hue (blue) scale, matching the calibration heatmap's palette.
function prAucColor(value) {
  const t = Math.min(1, Math.max(0, value));
  const light = [214, 231, 248];
  const dark = [15, 58, 110];
  const rgb = light.map((l, i) => Math.round(l + (dark[i] - l) * t));
  return `rgb(${rgb.join(",")})`;
}

function textColor(value) {
  return value > 0.5 ? "#fff" : "var(--text-primary)";
}

export default function DetectorComparisonTable({ comparison }) {
  if (!comparison || comparison.length === 0) {
    return <div className="loading">Loading...</div>;
  }

  return (
    <div>
      <table>
        <thead>
          <tr>
            <th>Detector</th>
            {ANOMALY_TYPE_COLS.map((c) => (
              <th key={c}>{c}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {comparison.map((row) => (
            <tr key={row.detector}>
              <td>{row.detector}</td>
              {ANOMALY_TYPE_COLS.map((c) => {
                const prAuc = row[`${c}_pr_auc`];
                const lift = row[`${c}_lift`];
                return (
                  <td
                    key={c}
                    style={{
                      background: prAuc != null ? prAucColor(prAuc) : undefined,
                      color: prAuc != null ? textColor(prAuc) : undefined,
                      fontWeight: 600,
                    }}
                  >
                    {prAuc != null ? (
                      <>
                        {prAuc.toFixed(3)}
                        {lift != null && (
                          <span style={{ fontWeight: 400, opacity: 0.85, fontSize: 11 }}> ({lift.toFixed(1)}x)</span>
                        )}
                      </>
                    ) : (
                      "—"
                    )}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <p style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 10 }}>
        Cell = PR-AUC (0-1, higher is better), with lift-over-random in parentheses (PR-AUC divided by
        that anomaly type's actual injected prevalence rate — a detector no better than random guessing
        scores ~1.0x regardless of type, so lift stays comparable across types with very different base
        rates). CUSUM's own precision/recall/PR-AUC per anomaly type, computed standalone rather than
        only as part of the ensemble — shown here honestly whether or not it outperforms the
        point-anomaly detectors on slow_drift.
      </p>
    </div>
  );
}

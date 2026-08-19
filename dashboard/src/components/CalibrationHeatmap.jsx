import { Fragment } from "react";

const MAX_ERROR_FOR_SCALE = 0.12;

// Sequential single-hue (blue) scale, light -> dark, per the project's chart palette.
function errorColor(error) {
  const t = Math.min(1, Math.abs(error) / MAX_ERROR_FOR_SCALE);
  const light = [214, 231, 248]; // near-white blue
  const dark = [15, 58, 110]; // deep blue
  const rgb = light.map((l, i) => Math.round(l + (dark[i] - l) * t));
  return `rgb(${rgb.join(",")})`;
}

export default function CalibrationHeatmap({ calibration }) {
  const crossTab = calibration.filter((c) => c.segment_dimension === "department_x_tenure_band");
  if (crossTab.length === 0) {
    return <div className="empty-state">No calibration data.</div>;
  }

  const departments = [...new Set(crossTab.map((c) => c.segment_value.split(" / ")[0]))].sort();
  const tenureBands = ["0-2", "2-5", "5+"];

  const cellFor = (dept, band) =>
    crossTab.find((c) => c.segment_value === `${dept} / ${band}`);

  const logrankByDimension = Object.fromEntries(
    ["department", "tenure_band", "comp_band"].map((dim) => {
      const row = calibration.find((c) => c.segment_dimension === dim);
      return [dim, row ? row.logrank_p_value : null];
    })
  );

  return (
    <div>
      <div
        className="calibration-grid"
        style={{ gridTemplateColumns: `120px repeat(${tenureBands.length}, 1fr)` }}
      >
        <div />
        {tenureBands.map((band) => (
          <div key={band} style={{ fontSize: 11, color: "var(--text-muted)", textAlign: "center" }}>
            {band} yrs
          </div>
        ))}
        {departments.map((dept) => (
          <Fragment key={dept}>
            <div style={{ fontSize: 12, alignSelf: "center" }}>{dept}</div>
            {tenureBands.map((band) => {
              const cell = cellFor(dept, band);
              return (
                <div
                  key={`${dept}-${band}`}
                  className="calibration-cell"
                  style={{ background: cell ? errorColor(cell.calibration_error) : "#eee" }}
                  title={
                    cell
                      ? `predicted ${(cell.predicted_survival * 100).toFixed(1)}% vs observed ${(cell.observed_survival * 100).toFixed(1)}%`
                      : "no data"
                  }
                >
                  {cell ? `${(cell.calibration_error * 100).toFixed(1)}pp` : "—"}
                </div>
              );
            })}
          </Fragment>
        ))}
      </div>
      <p style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 10 }}>
        Cell = |predicted - observed| 12-month survival probability (percentage points). Log-rank test
        p-values across segments: department {formatP(logrankByDimension.department)}, tenure band{" "}
        {formatP(logrankByDimension.tenure_band)}, comp band {formatP(logrankByDimension.comp_band)}.
      </p>
    </div>
  );
}

function formatP(p) {
  if (p === null || p === undefined || Number.isNaN(p)) return "n/a";
  return p < 0.001 ? "< 0.001" : p.toFixed(3);
}

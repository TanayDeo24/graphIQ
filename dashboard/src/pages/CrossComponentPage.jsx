import { useEffect, useState } from "react";
import { api } from "../api";
import QuadrantScatter from "../components/QuadrantScatter";

const QUADRANT_LABEL = {
  high_risk_high_anomaly: "High risk / high anomaly",
  high_risk_low_anomaly: "High risk / low anomaly",
  low_risk_high_anomaly: "Low risk / high anomaly",
  low_risk_low_anomaly: "Low risk / low anomaly",
};

export default function CrossComponentPage() {
  const [data, setData] = useState(null);

  useEffect(() => {
    api.crossComponentQuadrant().then(setData);
  }, []);

  return (
    <div>
      <h2 style={{ marginTop: 0 }}>Cross-component analysis</h2>
      <p style={{ color: "var(--text-secondary)", maxWidth: 700, fontSize: 13 }}>
        Does an employee's attrition risk relate to their spend-anomaly signal? Both components read
        from the same employees table, but have never been directly compared until now. Correlational
        only — see the disclaimer below the chart.
      </p>

      {!data ? (
        <div className="loading">Loading...</div>
      ) : (
        <>
          <div className="card">
            <h2>Attrition risk vs. spend anomaly, per employee</h2>
            <div className="card-sub">
              Every employee plotted: baseline GBM attrition risk score (x) vs. count of their transactions
              flagged at the operating threshold (y), colored by quadrant. Dashed lines mark each score's
              own top-quartile threshold.
            </div>
            <QuadrantScatter employees={data.employees} summary={data.summary} />
          </div>

          <div className="card">
            <h2>Quadrant counts &amp; characteristics</h2>
            <table>
              <thead>
                <tr>
                  <th>Quadrant</th>
                  <th>Count</th>
                  <th>Top department</th>
                  <th>Top tenure band</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(data.quadrant_counts).map(([q, count]) => {
                  const rows = data.characteristics.filter((c) => c.quadrant === q);
                  const topDept = rows
                    .filter((c) => c.dimension === "department")
                    .sort((a, b) => b.count - a.count)[0];
                  const topTenure = rows
                    .filter((c) => c.dimension === "tenure_band")
                    .sort((a, b) => b.count - a.count)[0];
                  return (
                    <tr key={q}>
                      <td>{QUADRANT_LABEL[q] || q}</td>
                      <td>{count}</td>
                      <td>{topDept ? `${topDept.dimension_value} (${(topDept.pct_of_quadrant * 100).toFixed(0)}%)` : "—"}</td>
                      <td>{topTenure ? `${topTenure.dimension_value} (${(topTenure.pct_of_quadrant * 100).toFixed(0)}%)` : "—"}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}

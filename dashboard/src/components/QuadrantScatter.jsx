import { CartesianGrid, Legend, ReferenceLine, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis } from "recharts";

const QUADRANT_COLOR = {
  high_risk_high_anomaly: "var(--status-critical)",
  high_risk_low_anomaly: "var(--series-2)",
  low_risk_high_anomaly: "var(--series-1)",
  low_risk_low_anomaly: "var(--series-3)",
};

const QUADRANT_LABEL = {
  high_risk_high_anomaly: "high risk / high anomaly",
  high_risk_low_anomaly: "high risk / low anomaly",
  low_risk_high_anomaly: "low risk / high anomaly",
  low_risk_low_anomaly: "low risk / low anomaly",
};

export default function QuadrantScatter({ employees, summary }) {
  if (!employees || employees.length === 0) {
    return <div className="loading">Loading...</div>;
  }

  const byQuadrant = {};
  employees.forEach((e) => {
    byQuadrant[e.quadrant] = byQuadrant[e.quadrant] || [];
    byQuadrant[e.quadrant].push(e);
  });

  const riskThreshold = employees.find((e) => e.is_top_risk_quartile)
    ? Math.min(...employees.filter((e) => e.is_top_risk_quartile).map((e) => e.gbm_risk_score))
    : null;
  const anomalyThreshold = employees.find((e) => e.is_top_spend_quartile)
    ? Math.min(...employees.filter((e) => e.is_top_spend_quartile).map((e) => e.spend_anomaly_score))
    : null;

  return (
    <div>
      {summary && (
        <p style={{ fontSize: 12, color: "var(--text-secondary)" }}>
          Spearman correlation: <strong>{summary.spearman_correlation.toFixed(3)}</strong>{" "}
          (p {summary.p_value < 0.001 ? "< 0.001" : summary.p_value.toFixed(3)}, {summary.n_permutations}{" "}
          permutations, n={summary.n_employees}). {summary.disclaimer}
        </p>
      )}
      <ResponsiveContainer width="100%" height={460}>
        <ScatterChart margin={{ top: 10, right: 20, bottom: 40, left: 10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
          <XAxis type="number" dataKey="gbm_risk_score" name="Attrition risk score" tick={{ fontSize: 11 }} label={{ value: "Attrition risk score (GBM)", position: "bottom", offset: 10, fontSize: 11 }} />
          <YAxis type="number" dataKey="spend_anomaly_score" name="Flagged transaction count" tick={{ fontSize: 11 }} label={{ value: "Flagged transaction count", angle: -90, position: "insideLeft", fontSize: 11 }} />
          <Tooltip
            cursor={{ strokeDasharray: "3 3" }}
            formatter={(value, name) => [typeof value === "number" ? value.toFixed(3) : value, name]}
            labelFormatter={() => ""}
          />
          <Legend formatter={(value) => QUADRANT_LABEL[value] || value} wrapperStyle={{ fontSize: 11 }} />
          {riskThreshold != null && <ReferenceLine x={riskThreshold} stroke="var(--text-muted)" strokeDasharray="4 4" />}
          {anomalyThreshold != null && <ReferenceLine y={anomalyThreshold} stroke="var(--text-muted)" strokeDasharray="4 4" />}
          {Object.keys(QUADRANT_COLOR).map((q) => (
            <Scatter key={q} name={q} data={byQuadrant[q] || []} fill={QUADRANT_COLOR[q]} opacity={0.7} r={3} />
          ))}
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}

import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

function buildHistogram(distribution) {
  if (distribution.length === 0) return [];
  const values = distribution.map((d) => d.lead_time_months);
  const max = Math.max(...values, 1);
  const binSize = Math.max(1, Math.ceil(max / 10));
  const bins = {};
  values.forEach((v) => {
    const bin = Math.floor(v / binSize) * binSize;
    bins[bin] = (bins[bin] || 0) + 1;
  });
  return Object.entries(bins)
    .map(([bin, count]) => ({ bin: `${bin}-${Number(bin) + binSize}mo`, count, sortKey: Number(bin) }))
    .sort((a, b) => a.sortKey - b.sortKey);
}

function CIBand({ statistic }) {
  if (!statistic) return null;
  const { point_estimate, ci_low, ci_high } = statistic;
  const max = Math.max(ci_high * 1.15, point_estimate * 1.15, 1);
  const pct = (v) => `${(v / max) * 100}%`;

  return (
    <div style={{ margin: "10px 0" }}>
      <div style={{ fontSize: 12, marginBottom: 4 }}>
        {statistic.statistic}: <strong>{point_estimate.toFixed(1)} months</strong>{" "}
        <span style={{ color: "var(--text-muted)" }}>
          (95% CI: {ci_low.toFixed(1)}–{ci_high.toFixed(1)})
        </span>
      </div>
      <div style={{ position: "relative", height: 14, background: "var(--surface-0)", borderRadius: 7 }}>
        <div
          style={{
            position: "absolute",
            left: pct(ci_low),
            width: `calc(${pct(ci_high)} - ${pct(ci_low)})`,
            top: 0,
            bottom: 0,
            background: "var(--series-1)",
            opacity: 0.25,
            borderRadius: 7,
          }}
        />
        <div
          style={{
            position: "absolute",
            left: pct(point_estimate),
            top: -3,
            width: 3,
            height: 20,
            background: "var(--series-1)",
            borderRadius: 2,
          }}
        />
      </div>
    </div>
  );
}

export default function LeadTimeChart({ leadTime }) {
  const { summary = [], distribution = [] } = leadTime || {};
  const histogram = buildHistogram(distribution);

  if (distribution.length === 0) {
    return <div className="empty-state">No true-positive employees crossed the risk threshold before departure.</div>;
  }

  return (
    <div>
      {summary.map((s) => (
        <CIBand key={s.statistic} statistic={s} />
      ))}
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={histogram}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
          <XAxis dataKey="bin" tick={{ fontSize: 11 }} />
          <YAxis tick={{ fontSize: 11 }} allowDecimals={false} />
          <Tooltip />
          <Bar dataKey="count" fill="var(--series-1)" radius={[4, 4, 0, 0]} name="Employees" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

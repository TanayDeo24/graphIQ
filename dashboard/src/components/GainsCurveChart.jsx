import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

export default function GainsCurveChart({ gainsCurve }) {
  if (!gainsCurve || gainsCurve.length === 0) {
    return <div className="loading">Loading...</div>;
  }

  const data = gainsCurve.map((d) => ({
    alerts: Math.round(d.pct_alerts_raised * 100),
    captured: Math.round(d.pct_dollar_volume_captured * 1000) / 10,
    random: Math.round(d.pct_alerts_raised * 100 * 10) / 10,
  }));

  const at10 = data.reduce((best, d) => (Math.abs(d.alerts - 10) < Math.abs(best.alerts - 10) ? d : best), data[0]);

  return (
    <div>
      <p style={{ fontSize: 12, color: "var(--text-secondary)" }}>
        Top 10% of alerts capture <strong>{at10.captured}%</strong> of flagged dollar volume.
      </p>
      <ResponsiveContainer width="100%" height={260}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
          <XAxis dataKey="alerts" tick={{ fontSize: 11 }} label={{ value: "% of alerts raised", position: "insideBottom", offset: -4, fontSize: 11 }} />
          <YAxis tick={{ fontSize: 11 }} label={{ value: "% $ captured", angle: -90, position: "insideLeft", fontSize: 11 }} />
          <Tooltip formatter={(v) => `${v}%`} />
          <Line type="monotone" dataKey="captured" stroke="var(--series-1)" strokeWidth={2} dot={false} name="Dollar volume captured" />
          <Line type="monotone" dataKey="random" stroke="var(--text-muted)" strokeWidth={1.5} strokeDasharray="4 4" dot={false} name="Random baseline" />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

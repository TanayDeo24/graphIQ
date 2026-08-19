import { CartesianGrid, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

const SERIES_COLORS = ["var(--series-1)", "var(--series-2)", "var(--series-3)", "var(--series-4)", "var(--series-5)"];

export default function CusumChart({ drift }) {
  if (!drift || drift.series.length === 0) {
    return <div className="empty-state">No CUSUM series for this employee.</div>;
  }

  const { control_limits, series } = drift;
  const categories = [...new Set(series.map((s) => s.merchant_category))];
  const months = [...new Set(series.map((s) => s.month))].sort();

  const data = months.map((month) => {
    const row = { month: month.slice(0, 7) };
    categories.forEach((cat) => {
      const point = series.find((s) => s.month === month && s.merchant_category === cat);
      row[cat] = point ? point.cusum_statistic : null;
    });
    return row;
  });

  return (
    <ResponsiveContainer width="100%" height={280}>
      <LineChart data={data}>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
        <XAxis dataKey="month" tick={{ fontSize: 10 }} />
        <YAxis tick={{ fontSize: 11 }} label={{ value: "CUSUM statistic (sigma units)", angle: -90, position: "insideLeft", fontSize: 10 }} />
        <Tooltip />
        <Legend wrapperStyle={{ fontSize: 11 }} />
        <ReferenceLine
          y={control_limits.h_sigma}
          stroke="var(--status-critical)"
          strokeDasharray="4 4"
          label={{ value: `control limit h=${control_limits.h_sigma}sigma`, fontSize: 10, position: "right" }}
        />
        {categories.map((cat, i) => (
          <Line
            key={cat}
            type="monotone"
            dataKey={cat}
            stroke={SERIES_COLORS[i % SERIES_COLORS.length]}
            strokeWidth={2}
            dot={{ r: 2 }}
            connectNulls
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}

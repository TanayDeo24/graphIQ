import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

const SINGLE = new Set(["only_IF", "only_AE", "only_CUSUM"]);
const PAIR = new Set(["IF+AE", "IF+CUSUM", "AE+CUSUM"]);

function colorFor(combination) {
  if (combination === "all_three") return "var(--status-critical)";
  if (PAIR.has(combination)) return "var(--series-4)";
  return "var(--series-1)";
}

export default function DetectorOverlapChart({ overlap }) {
  if (!overlap || overlap.length === 0) {
    return <div className="loading">Loading...</div>;
  }

  const total = overlap.reduce((sum, d) => sum + d.transaction_count, 0);
  const singleTotal = overlap.filter((d) => SINGLE.has(d.combination)).reduce((s, d) => s + d.transaction_count, 0);

  return (
    <div>
      <p style={{ fontSize: 12, color: "var(--text-secondary)" }}>
        {((singleTotal / total) * 100).toFixed(0)}% of flagged transactions are flagged by only one
        detector — the detectors mostly disagree, which is why the ensemble matters.
      </p>
      <ResponsiveContainer width="100%" height={260}>
        <BarChart data={overlap}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
          <XAxis dataKey="combination" tick={{ fontSize: 10 }} angle={-20} textAnchor="end" height={60} />
          <YAxis tick={{ fontSize: 11 }} label={{ value: "transactions", angle: -90, position: "insideLeft", fontSize: 10 }} />
          <Tooltip formatter={(v) => v.toLocaleString()} />
          <Bar dataKey="transaction_count" radius={[4, 4, 0, 0]} name="Transactions">
            {overlap.map((d) => (
              <Cell key={d.combination} fill={colorFor(d.combination)} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

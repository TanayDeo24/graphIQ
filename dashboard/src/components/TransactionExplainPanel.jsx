import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

const COLORS = { amount_deviation: "var(--series-1)", frequency: "var(--series-2)", merchant_novelty: "var(--series-3)" };

export default function TransactionExplainPanel({ explanation, transactionId }) {
  if (!explanation) {
    return <div className="loading">Loading...</div>;
  }
  if (explanation.length === 0) {
    return <div className="empty-state">No explanation available for transaction {transactionId}.</div>;
  }

  const data = explanation.map((e) => ({ ...e, pct: Math.round(e.contribution * 1000) / 10 }));

  return (
    <ResponsiveContainer width="100%" height={160}>
      <BarChart data={data} layout="vertical" margin={{ left: 16 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
        <XAxis type="number" tick={{ fontSize: 11 }} unit="%" />
        <YAxis dataKey="sub_signal" type="category" width={130} tick={{ fontSize: 11 }} />
        <Tooltip formatter={(v) => `${v}%`} />
        <Bar dataKey="pct" radius={[0, 4, 4, 0]} name="Contribution">
          {data.map((d) => (
            <Cell key={d.sub_signal} fill={COLORS[d.sub_signal] || "var(--series-4)"} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

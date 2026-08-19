import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

export default function ShapBarChart({ shap, employeeId }) {
  if (!shap || shap.length === 0) {
    return (
      <div className="empty-state">
        No SHAP breakdown for employee {employeeId} (only computed for the top-risk decile).
      </div>
    );
  }

  const data = [...shap].sort((a, b) => Math.abs(a.shap_value) - Math.abs(b.shap_value));

  return (
    <ResponsiveContainer width="100%" height={Math.max(200, data.length * 32)}>
      <BarChart data={data} layout="vertical" margin={{ left: 24 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
        <XAxis type="number" tick={{ fontSize: 11 }} />
        <YAxis dataKey="feature" type="category" width={140} tick={{ fontSize: 11 }} />
        <Tooltip formatter={(v) => v.toFixed(4)} />
        <Bar dataKey="shap_value" name="SHAP contribution" radius={[0, 4, 4, 0]}>
          {data.map((d, i) => (
            <Cell key={i} fill={d.shap_value >= 0 ? "var(--status-critical)" : "var(--series-1)"} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

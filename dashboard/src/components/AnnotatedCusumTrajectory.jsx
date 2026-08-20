import { CartesianGrid, ComposedChart, Line, ReferenceArea, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

function monthLabel(dateStr) {
  return dateStr ? dateStr.slice(0, 7) : null;
}

export default function AnnotatedCusumTrajectory({ data, controlLimits }) {
  if (!data || data.length === 0) {
    return <div className="loading">Loading...</div>;
  }

  const byCase = {};
  data.forEach((row) => {
    byCase[row.case_label] = byCase[row.case_label] || [];
    byCase[row.case_label].push(row);
  });

  return (
    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: 16 }}>
      {Object.entries(byCase).map(([caseLabel, rows]) => {
        const sorted = [...rows].sort((a, b) => a.month.localeCompare(b.month));
        const chartData = sorted.map((r) => ({ month: monthLabel(r.month), cusum_statistic: r.cusum_statistic }));
        const onset = monthLabel(sorted[0].onset_month);
        const end = monthLabel(sorted[0].end_month);
        const flagged = monthLabel(sorted[0].flagged_month);
        const duringActive = sorted[0].caught_during_active_window;

        return (
          <div key={caseLabel} style={{ border: "1px solid var(--border)", borderRadius: 8, padding: 10 }}>
            <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 4 }}>
              {caseLabel} — employee {sorted[0].employee_id} / {sorted[0].merchant_category}{" "}
              <span style={{ fontWeight: 400, color: duringActive ? "var(--series-3)" : "var(--status-critical)" }}>
                ({duringActive ? "caught while active" : "caught after drift ended"})
              </span>
            </div>
            <ResponsiveContainer width="100%" height={180}>
              <ComposedChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                <XAxis dataKey="month" tick={{ fontSize: 9 }} />
                <YAxis tick={{ fontSize: 10 }} />
                <Tooltip />
                {onset && end && (
                  <ReferenceArea x1={onset} x2={end} fill="var(--series-2)" fillOpacity={0.12} label={{ value: "drift window", fontSize: 9, position: "insideTop" }} />
                )}
                <ReferenceLine y={controlLimits?.h_sigma} stroke="var(--status-critical)" strokeDasharray="4 4" label={{ value: `h=${controlLimits?.h_sigma}`, fontSize: 9, position: "right" }} />
                {flagged && (
                  <ReferenceLine x={flagged} stroke="var(--series-1)" strokeWidth={2} label={{ value: "detected", fontSize: 9, position: "top" }} />
                )}
                <Line type="monotone" dataKey="cusum_statistic" stroke="var(--series-1)" strokeWidth={2} dot={{ r: 2 }} />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        );
      })}
    </div>
  );
}

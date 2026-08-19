export default function RiskScoreTable({ rows, onSelect, selectedId, page, pageSize, total, onPageChange }) {
  const totalPages = Math.max(1, Math.ceil(total / pageSize));

  return (
    <div>
      <table>
        <thead>
          <tr>
            <th>Employee</th>
            <th>Department</th>
            <th>Tenure band</th>
            <th>GBM risk score</th>
            <th>Predicted 12mo survival</th>
            <th>Top-risk quartile</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr
              key={row.employee_id}
              className="row-clickable"
              style={row.employee_id === selectedId ? { background: "var(--surface-0)" } : undefined}
              onClick={() => onSelect(row.employee_id)}
            >
              <td>{row.employee_id}</td>
              <td>{row.department}</td>
              <td>{row.tenure_band}</td>
              <td>{row.gbm_risk_score.toFixed(3)}</td>
              <td>{(row.gbm_predicted_survival_12m * 100).toFixed(1)}%</td>
              <td>
                <span className={`pill ${row.is_top_risk_quartile ? "high" : "low"}`}>
                  {row.is_top_risk_quartile ? "high" : "normal"}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="pagination">
        <button disabled={page <= 1} onClick={() => onPageChange(page - 1)}>
          Prev
        </button>
        <span>
          Page {page} of {totalPages} ({total} employees)
        </span>
        <button disabled={page >= totalPages} onClick={() => onPageChange(page + 1)}>
          Next
        </button>
      </div>
    </div>
  );
}

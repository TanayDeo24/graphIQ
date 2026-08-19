export default function AnomalyTable({ rows, onSelect, selectedId, page, pageSize, total, onPageChange }) {
  const totalPages = Math.max(1, Math.ceil(total / pageSize));

  return (
    <div>
      <table>
        <thead>
          <tr>
            <th>Transaction</th>
            <th>Employee</th>
            <th>Ensemble score</th>
            <th>Predicted flag</th>
            <th>Anomaly type (ground truth)</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr
              key={row.transaction_id}
              className="row-clickable"
              style={row.transaction_id === selectedId ? { background: "var(--surface-0)" } : undefined}
              onClick={() => onSelect(row.transaction_id)}
            >
              <td>{row.transaction_id}</td>
              <td>{row.employee_id}</td>
              <td>{row.ensemble_score.toFixed(3)}</td>
              <td>
                <span className={`pill ${row.predicted_flag ? "high" : "low"}`}>
                  {row.predicted_flag ? "flagged" : "clear"}
                </span>
              </td>
              <td>{row.anomaly_type || "—"}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="pagination">
        <button disabled={page <= 1} onClick={() => onPageChange(page - 1)}>
          Prev
        </button>
        <span>
          Page {page} of {totalPages} ({total.toLocaleString()} transactions)
        </span>
        <button disabled={page >= totalPages} onClick={() => onPageChange(page + 1)}>
          Next
        </button>
      </div>
    </div>
  );
}

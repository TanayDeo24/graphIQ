import { ResponsiveContainer, Tooltip, Treemap } from "recharts";

const ANOMALY_COLOR = {
  point_spike: "var(--series-1)",
  slow_drift: "var(--series-2)",
  coordinated_pattern: "var(--series-5)",
};

function CustomCell(props) {
  const { x, y, width, height, anomaly_type, department_name, merchant_category } = props;
  if (width < 2 || height < 2) return null;
  const fill = ANOMALY_COLOR[anomaly_type] || "var(--series-4)";
  return (
    <g>
      <rect x={x} y={y} width={width} height={height} fill={fill} stroke="var(--surface-1)" strokeWidth={1.5} fillOpacity={0.85} />
      {width > 60 && height > 24 && (
        <text x={x + 6} y={y + 16} fontSize={10} fill="#fff">
          {department_name} / {merchant_category}
        </text>
      )}
    </g>
  );
}

function CustomTooltip({ active, payload }) {
  if (!active || !payload || !payload.length) return null;
  const d = payload[0].payload;
  return (
    <div style={{ background: "var(--surface-2)", border: "1px solid var(--border)", borderRadius: 6, padding: 8, fontSize: 11 }}>
      <div>
        <strong>
          {d.department_name} / {d.merchant_category}
        </strong>
      </div>
      <div>anomaly type: {d.anomaly_type}</div>
      <div>dollar volume: ${d.value.toLocaleString(undefined, { maximumFractionDigits: 0 })}</div>
      <div>transactions: {d.transaction_count}</div>
    </div>
  );
}

export default function DollarTreemap({ data }) {
  if (!data || data.length === 0) {
    return <div className="loading">Loading...</div>;
  }

  const treeData = data.map((d) => ({
    name: `${d.department_name} / ${d.merchant_category} / ${d.anomaly_type}`,
    value: d.dollar_volume,
    department_name: d.department_name,
    merchant_category: d.merchant_category,
    anomaly_type: d.anomaly_type,
    transaction_count: d.transaction_count,
  }));

  return (
    <div>
      <ResponsiveContainer width="100%" height={360}>
        <Treemap data={treeData} dataKey="value" stroke="var(--surface-1)" content={<CustomCell />}>
          <Tooltip content={<CustomTooltip />} />
        </Treemap>
      </ResponsiveContainer>
      <div style={{ display: "flex", gap: 16, fontSize: 11, marginTop: 8 }}>
        {Object.entries(ANOMALY_COLOR).map(([type, color]) => (
          <span key={type} style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <span style={{ width: 10, height: 10, background: color, display: "inline-block", borderRadius: 2 }} />
            {type}
          </span>
        ))}
      </div>
    </div>
  );
}

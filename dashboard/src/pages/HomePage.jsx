import { useEffect, useState } from "react";
import { api } from "../api";

function formatValue(value, format) {
  if (format === "percent") return `${(value * 100).toFixed(1)}%`;
  if (format === "multiplier") return `${value.toFixed(1)}x`;
  return value.toFixed(3);
}

export default function HomePage() {
  const [stats, setStats] = useState(null);

  useEffect(() => {
    api.headlineStats().then((d) => setStats(d.stats));
  }, []);

  return (
    <div>
      <h2 style={{ marginTop: 0 }}>GraphIQ</h2>
      <p style={{ color: "var(--text-secondary)", maxWidth: 640 }}>
        Unified-schema workforce analytics: attrition risk (survival analysis) and spend anomaly
        detection, both reading from the same employees table. The numbers below are the strongest,
        already-validated results from each component — see the Attrition and Spend pages for full
        detail, methodology, and honest caveats.
      </p>

      {!stats ? (
        <div className="loading">Loading...</div>
      ) : (
        <div className="stat-tiles" style={{ marginTop: 24 }}>
          {stats.map((s) => (
            <div className="stat-tile" key={s.label} style={{ padding: "20px 18px" }}>
              <div className="label">{s.label}</div>
              <div className="value" style={{ fontSize: 30 }}>
                {formatValue(s.value, s.format)}
              </div>
              <div style={{ fontSize: 12, color: "var(--text-muted)", marginTop: 8, lineHeight: 1.4 }}>
                {s.note}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

import { useMemo } from "react";
import { sankey, sankeyLinkHorizontal } from "d3-sankey";

const CHECKPOINTS = [6, 12, 18, 24, 30, 36];
const TIERS = ["low", "medium", "high"];
const TIER_COLOR = { low: "var(--series-3)", medium: "var(--series-4)", high: "var(--status-critical)" };
const WIDTH = 900;
const HEIGHT = 360;

function nodeId(checkpoint, tier) {
  return `${checkpoint}-${tier}`;
}

export default function RiskMigrationSankey({ sankeyLinks }) {
  const graph = useMemo(() => {
    if (!sankeyLinks || sankeyLinks.length === 0) return null;

    const nodes = [];
    const nodeIndex = {};
    CHECKPOINTS.forEach((cp) => {
      TIERS.forEach((tier) => {
        const id = nodeId(cp, tier);
        nodeIndex[id] = nodes.length;
        nodes.push({ id, checkpoint: cp, tier });
      });
    });

    const links = sankeyLinks
      .filter((l) => l.employee_count > 0)
      .map((l) => ({
        source: nodeIndex[nodeId(l.checkpoint_from, l.tier_from)],
        target: nodeIndex[nodeId(l.checkpoint_to, l.tier_to)],
        value: l.employee_count,
        tierFrom: l.tier_from,
      }));

    const layout = sankey()
      .nodeId((d) => nodeIndex[d.id])
      .nodeWidth(14)
      .nodePadding(10)
      .extent([
        [10, 10],
        [WIDTH - 10, HEIGHT - 10],
      ]);

    return layout({ nodes: nodes.map((n) => ({ ...n })), links: links.map((l) => ({ ...l })) });
  }, [sankeyLinks]);

  if (!graph) {
    return <div className="loading">Loading...</div>;
  }

  const linkPath = sankeyLinkHorizontal();

  return (
    <div style={{ overflowX: "auto" }}>
      <svg width={WIDTH} height={HEIGHT + 24}>
        <g>
          {graph.links.map((link, i) => (
            <path
              key={i}
              d={linkPath(link)}
              fill="none"
              stroke={TIER_COLOR[link.tierFrom]}
              strokeOpacity={0.35}
              strokeWidth={Math.max(1, link.width)}
            />
          ))}
          {graph.nodes.map((node) => (
            <rect
              key={node.id}
              x={node.x0}
              y={node.y0}
              width={node.x1 - node.x0}
              height={Math.max(1, node.y1 - node.y0)}
              fill={TIER_COLOR[node.tier]}
            />
          ))}
        </g>
        {CHECKPOINTS.map((cp, i) => (
          <text
            key={cp}
            x={10 + (i * (WIDTH - 20)) / (CHECKPOINTS.length - 1)}
            y={HEIGHT + 18}
            fontSize={11}
            fill="var(--text-muted)"
            textAnchor="middle"
          >
            month {cp}
          </text>
        ))}
      </svg>
      <div style={{ display: "flex", gap: 16, fontSize: 12, marginTop: 4 }}>
        {TIERS.map((t) => (
          <span key={t} style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <span style={{ width: 10, height: 10, background: TIER_COLOR[t], display: "inline-block", borderRadius: 2 }} />
            {t}
          </span>
        ))}
      </div>
    </div>
  );
}

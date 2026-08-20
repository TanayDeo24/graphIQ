import { useEffect, useState } from "react";
import { api } from "../api";
import RiskScoreTable from "../components/RiskScoreTable";
import CalibrationHeatmap from "../components/CalibrationHeatmap";
import LeadTimeChart from "../components/LeadTimeChart";
import ShapBarChart from "../components/ShapBarChart";
import InteractionHeatmap from "../components/InteractionHeatmap";
import RiskMigrationSankey from "../components/RiskMigrationSankey";

const PAGE_SIZE = 15;

export default function AttritionPage() {
  const [department, setDepartment] = useState("");
  const [tenureBand, setTenureBand] = useState("");
  const [page, setPage] = useState(1);
  const [riskData, setRiskData] = useState({ results: [], total: 0 });
  const [calibration, setCalibration] = useState([]);
  const [leadTime, setLeadTime] = useState(null);
  const [selectedEmployee, setSelectedEmployee] = useState(null);
  const [shap, setShap] = useState(null);
  const [loading, setLoading] = useState(true);
  const [interactionHeatmap, setInteractionHeatmap] = useState(null);
  const [riskMigration, setRiskMigration] = useState(null);

  useEffect(() => {
    setLoading(true);
    api
      .riskScores({
        department: department || undefined,
        tenure_band: tenureBand || undefined,
        page,
        page_size: PAGE_SIZE,
      })
      .then(setRiskData)
      .finally(() => setLoading(false));
  }, [department, tenureBand, page]);

  useEffect(() => {
    api.calibration().then(setCalibration);
    api.leadTime().then(setLeadTime);
    api.interactionHeatmap().then(setInteractionHeatmap);
    api.riskMigration().then(setRiskMigration);
  }, []);

  useEffect(() => {
    if (selectedEmployee == null) return;
    api
      .shap(selectedEmployee)
      .then(setShap)
      .catch(() => setShap([]));
  }, [selectedEmployee]);

  return (
    <div>
      <h2 style={{ marginTop: 0 }}>Attrition risk</h2>

      <div className="card">
        <h2>Risk scores</h2>
        <div className="card-sub">GBM survival model risk score, sorted highest-risk first. Click a row for its SHAP breakdown.</div>
        <div className="filters-row">
          <select
            value={department}
            onChange={(e) => {
              setPage(1);
              setDepartment(e.target.value);
            }}
          >
            <option value="">All departments</option>
            <option value="Sales">Sales</option>
            <option value="Research & Development">Research & Development</option>
            <option value="Human Resources">Human Resources</option>
          </select>
          <select
            value={tenureBand}
            onChange={(e) => {
              setPage(1);
              setTenureBand(e.target.value);
            }}
          >
            <option value="">All tenure bands</option>
            <option value="0-2">0-2 years</option>
            <option value="2-5">2-5 years</option>
            <option value="5+">5+ years</option>
          </select>
        </div>
        {loading ? (
          <div className="loading">Loading...</div>
        ) : (
          <RiskScoreTable
            rows={riskData.results}
            total={riskData.total}
            page={page}
            pageSize={PAGE_SIZE}
            onPageChange={setPage}
            onSelect={setSelectedEmployee}
            selectedId={selectedEmployee}
          />
        )}
      </div>

      {selectedEmployee != null && (
        <div className="card">
          <h2>SHAP breakdown — employee {selectedEmployee}</h2>
          <div className="card-sub">Feature contributions to this employee's predicted risk score.</div>
          <ShapBarChart shap={shap} employeeId={selectedEmployee} />
        </div>
      )}

      <div className="grid-2">
        <div className="card">
          <h2>Segment calibration</h2>
          <div className="card-sub">Department x tenure-band, 12-month predicted vs. observed survival.</div>
          {calibration.length > 0 ? <CalibrationHeatmap calibration={calibration} /> : <div className="loading">Loading...</div>}
        </div>
        <div className="card">
          <h2>Lead-time distribution</h2>
          <div className="card-sub">Months between risk-threshold crossing and actual departure (true positives), 95% bootstrap CI.</div>
          {leadTime ? <LeadTimeChart leadTime={leadTime} /> : <div className="loading">Loading...</div>}
        </div>
      </div>

      <div className="card">
        <h2>Interaction risk heatmap</h2>
        <div className="card-sub">Baseline tenure band x review-score trend, cell = mean baseline GBM risk score.</div>
        <InteractionHeatmap data={interactionHeatmap} />
      </div>

      <div className="card">
        <h2>Risk-migration Sankey (illustrative)</h2>
        <div className="card-sub" style={{ color: "var(--status-critical)" }}>
          {riskMigration?.disclaimer || "Illustrative re-scoring for visualization only — not the validated model used for every other metric on this page."}
        </div>
        {riskMigration ? (
          <RiskMigrationSankey sankeyLinks={riskMigration.sankey_links} />
        ) : (
          <div className="loading">Loading...</div>
        )}
      </div>
    </div>
  );
}

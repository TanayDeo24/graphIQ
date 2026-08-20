import { useEffect, useState } from "react";
import { api } from "../api";
import AnomalyTable from "../components/AnomalyTable";
import GainsCurveChart from "../components/GainsCurveChart";
import CusumChart from "../components/CusumChart";
import DetectorComparisonTable from "../components/DetectorComparisonTable";
import TransactionExplainPanel from "../components/TransactionExplainPanel";
import DetectorOverlapChart from "../components/DetectorOverlapChart";
import AnnotatedCusumTrajectory from "../components/AnnotatedCusumTrajectory";
import DollarTreemap from "../components/DollarTreemap";

const PAGE_SIZE = 15;

export default function SpendPage() {
  const [anomalyType, setAnomalyType] = useState("");
  const [flaggedOnly, setFlaggedOnly] = useState(false);
  const [page, setPage] = useState(1);
  const [anomalyData, setAnomalyData] = useState({ results: [], total: 0 });
  const [gainsCurve, setGainsCurve] = useState(null);
  const [alertFatigue, setAlertFatigue] = useState(null);
  const [selectedTransaction, setSelectedTransaction] = useState(null);
  const [explanation, setExplanation] = useState(null);
  const [driftEmployeeId, setDriftEmployeeId] = useState("1173");
  const [drift, setDrift] = useState(null);
  const [comparison, setComparison] = useState(null);
  const [loading, setLoading] = useState(true);
  const [overlap, setOverlap] = useState(null);
  const [treemap, setTreemap] = useState(null);
  const [annotatedTrajectories, setAnnotatedTrajectories] = useState(null);

  useEffect(() => {
    setLoading(true);
    api
      .anomalies({
        anomaly_type: anomalyType || undefined,
        flagged_only: flaggedOnly || undefined,
        page,
        page_size: PAGE_SIZE,
      })
      .then(setAnomalyData)
      .finally(() => setLoading(false));
  }, [anomalyType, flaggedOnly, page]);

  useEffect(() => {
    api.gainsCurve().then(setGainsCurve);
    api.alertFatigue().then((rows) => setAlertFatigue(rows[0]));
    api.detectorComparison().then(setComparison);
    api.detectorOverlap().then(setOverlap);
    api.dollarTreemap().then(setTreemap);
    api.cusumAnnotatedTrajectories().then(setAnnotatedTrajectories);
  }, []);

  useEffect(() => {
    if (selectedTransaction == null) return;
    api
      .explainTransaction(selectedTransaction)
      .then(setExplanation)
      .catch(() => setExplanation([]));
  }, [selectedTransaction]);

  useEffect(() => {
    if (!driftEmployeeId) {
      setDrift(null);
      return;
    }
    api.drift({ employee_id: Number(driftEmployeeId) }).then(setDrift);
  }, [driftEmployeeId]);

  return (
    <div>
      <h2 style={{ marginTop: 0 }}>Spend anomaly detection</h2>

      {alertFatigue && (
        <div className="stat-tiles">
          <div className="stat-tile">
            <div className="label">Alerts / 1,000 txns</div>
            <div className="value">{alertFatigue.alerts_per_1000_txns.toFixed(1)}</div>
          </div>
          <div className="stat-tile">
            <div className="label">Precision at threshold</div>
            <div className="value">{(alertFatigue.precision_at_threshold * 100).toFixed(1)}%</div>
          </div>
          <div className="stat-tile">
            <div className="label">Alerts raised</div>
            <div className="value">{alertFatigue.alerts_raised.toLocaleString()}</div>
          </div>
          <div className="stat-tile">
            <div className="label">Total transactions</div>
            <div className="value">{alertFatigue.total_transactions.toLocaleString()}</div>
          </div>
        </div>
      )}

      <div className="card">
        <h2>Flagged transactions</h2>
        <div className="card-sub">Ensemble anomaly score (Isolation Forest + autoencoder + CUSUM, rank-averaged). Click a row for its sub-signal breakdown.</div>
        <div className="filters-row">
          <select
            value={anomalyType}
            onChange={(e) => {
              setPage(1);
              setAnomalyType(e.target.value);
            }}
          >
            <option value="">All anomaly types</option>
            <option value="point_spike">point_spike</option>
            <option value="slow_drift">slow_drift</option>
            <option value="coordinated_pattern">coordinated_pattern</option>
          </select>
          <label style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 13 }}>
            <input
              type="checkbox"
              checked={flaggedOnly}
              onChange={(e) => {
                setPage(1);
                setFlaggedOnly(e.target.checked);
              }}
            />
            Flagged only
          </label>
        </div>
        {loading ? (
          <div className="loading">Loading...</div>
        ) : (
          <AnomalyTable
            rows={anomalyData.results}
            total={anomalyData.total}
            page={page}
            pageSize={PAGE_SIZE}
            onPageChange={setPage}
            onSelect={setSelectedTransaction}
            selectedId={selectedTransaction}
          />
        )}
      </div>

      {selectedTransaction != null && (
        <div className="card">
          <h2>Sub-signal breakdown — transaction {selectedTransaction}</h2>
          <div className="card-sub">Amount deviation vs. frequency vs. merchant novelty, from autoencoder reconstruction error + standardized feature deviation.</div>
          <TransactionExplainPanel explanation={explanation} transactionId={selectedTransaction} />
        </div>
      )}

      <div className="card">
        <h2>Detector comparison</h2>
        <div className="card-sub">
          Standalone precision/recall/PR-AUC per detector, broken out by anomaly type — not just the
          ensemble. Multiple rounds of diagnosed fixes (contaminated-baseline corrections in both CUSUM
          and the Isolation Forest/autoencoder features, a tuning-dataset-derived CUSUM threshold retune)
          substantially improved every feature-consuming detector's slow_drift performance — see the
          README for the full history. Cohort-level aggregation (a tested hypothesis) did not help —
          shown for comparison, not silently dropped.
        </div>
        <DetectorComparisonTable comparison={comparison} />
      </div>

      <div className="grid-2">
        <div className="card">
          <h2>Dollar-weighted gains curve</h2>
          <div className="card-sub">Cumulative % of anomalous dollar volume captured vs. % of alerts raised.</div>
          <GainsCurveChart gainsCurve={gainsCurve} />
        </div>
        <div className="card">
          <h2>CUSUM drift chart</h2>
          <div className="card-sub">Monthly CUSUM statistic per category, with control limit.</div>
          <div className="filters-row">
            <input
              type="number"
              placeholder="employee_id (optional)"
              value={driftEmployeeId}
              onChange={(e) => setDriftEmployeeId(e.target.value)}
              style={{ width: 180 }}
            />
          </div>
          <CusumChart drift={drift} />
          {drift?.detection_timing && (
            <p style={{ fontSize: 12, color: "var(--text-secondary)", marginTop: 10 }}>
              Across all injected slow_drift cases: {drift.detection_timing.n_detected} of{" "}
              {drift.detection_timing.n_total_cases} were ever detected by CUSUM (
              {((drift.detection_timing.n_detected / drift.detection_timing.n_total_cases) * 100).toFixed(1)}%).
              Of those detected, {(drift.detection_timing.pct_caught_during_active * 100).toFixed(0)}% were
              caught while the drift was still active and{" "}
              {(drift.detection_timing.pct_caught_after_ended * 100).toFixed(0)}% only after it had already
              ended.
            </p>
          )}
        </div>
      </div>

      <div className="card">
        <h2>Detector overlap</h2>
        <div className="card-sub">At the operating threshold, how many transactions are flagged by exactly 1, 2, or 3 of {"{"}Isolation Forest, Autoencoder, CUSUM{"}"} (Cohort CUSUM excluded — dominated).</div>
        <DetectorOverlapChart overlap={overlap} />
      </div>

      <div className="card">
        <h2>Annotated CUSUM trajectories</h2>
        <div className="card-sub">5 real detected slow_drift cases, full monthly trajectory, drift window shaded, control limit and detection month marked.</div>
        <AnnotatedCusumTrajectory data={annotatedTrajectories?.cases} controlLimits={annotatedTrajectories?.control_limits} />
      </div>

      <div className="card">
        <h2>Dollar treemap</h2>
        <div className="card-sub">Department x category, sized by anomalous dollar volume, colored by anomaly type.</div>
        <DollarTreemap data={treemap} />
      </div>
    </div>
  );
}

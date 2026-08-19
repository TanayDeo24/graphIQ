import { useEffect, useState } from "react";
import { api } from "../api";
import AnomalyTable from "../components/AnomalyTable";
import GainsCurveChart from "../components/GainsCurveChart";
import CusumChart from "../components/CusumChart";
import DetectorComparisonTable from "../components/DetectorComparisonTable";
import TransactionExplainPanel from "../components/TransactionExplainPanel";

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
          ensemble. After fixing a diagnosed contaminated-baseline bug and a tuning-dataset-derived
          threshold retune (see README), CUSUM is now the best standalone slow_drift detector here.
          Cohort-level aggregation (a second hypothesis tested) did not help — shown for comparison,
          not silently dropped.
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
    </div>
  );
}

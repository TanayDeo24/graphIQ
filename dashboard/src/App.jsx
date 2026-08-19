import { NavLink, Route, HashRouter, Routes, Navigate } from "react-router-dom";
import "./App.css";
import AttritionPage from "./pages/AttritionPage";
import SpendPage from "./pages/SpendPage";

export default function App() {
  return (
    <HashRouter>
      <div className="app-shell">
        <aside className="app-nav">
          <h1>GraphIQ</h1>
          <div className="subtitle">Unified-schema workforce analytics demo</div>
          <nav>
            <NavLink to="/attrition" className={({ isActive }) => (isActive ? "active" : "")}>
              Attrition risk
            </NavLink>
            <NavLink to="/spend" className={({ isActive }) => (isActive ? "active" : "")}>
              Spend anomalies
            </NavLink>
          </nav>
        </aside>
        <main className="app-main">
          <div className="disclaimer-banner">
            Methodology demonstration on public/synthetic data — not a real finding about any
            company. Counterfactual sensitivity results are correlational, never causal.
          </div>
          <Routes>
            <Route path="/" element={<Navigate to="/attrition" replace />} />
            <Route path="/attrition" element={<AttritionPage />} />
            <Route path="/spend" element={<SpendPage />} />
          </Routes>
        </main>
      </div>
    </HashRouter>
  );
}

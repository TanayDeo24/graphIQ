import { NavLink, Route, HashRouter, Routes } from "react-router-dom";
import "./App.css";
import HomePage from "./pages/HomePage";
import AttritionPage from "./pages/AttritionPage";
import SpendPage from "./pages/SpendPage";
import CrossComponentPage from "./pages/CrossComponentPage";
import ChatPanel from "./components/ChatPanel";

export default function App() {
  return (
    <HashRouter>
      <div className="app-shell">
        <aside className="app-nav">
          <h1>GraphIQ</h1>
          <div className="subtitle">Unified-schema workforce analytics demo</div>
          <nav>
            <NavLink to="/" end className={({ isActive }) => (isActive ? "active" : "")}>
              Home
            </NavLink>
            <NavLink to="/attrition" className={({ isActive }) => (isActive ? "active" : "")}>
              Attrition risk
            </NavLink>
            <NavLink to="/spend" className={({ isActive }) => (isActive ? "active" : "")}>
              Spend anomalies
            </NavLink>
            <NavLink to="/cross-component" className={({ isActive }) => (isActive ? "active" : "")}>
              Cross-component
            </NavLink>
          </nav>
        </aside>
        <main className="app-main">
          <div className="disclaimer-banner">
            Methodology demonstration on public/synthetic data — not a real finding about any
            company. Counterfactual sensitivity and cross-component results are correlational, never
            causal.
          </div>
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/attrition" element={<AttritionPage />} />
            <Route path="/spend" element={<SpendPage />} />
            <Route path="/cross-component" element={<CrossComponentPage />} />
          </Routes>
        </main>
        <ChatPanel />
      </div>
    </HashRouter>
  );
}

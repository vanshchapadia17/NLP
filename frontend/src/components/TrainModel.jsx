import { useState } from "react";
import "./Classify.css";

const METRICS = ["accuracy", "precision", "recall", "f1"];

function MetricBar({ value }) {
  const pct = (value * 100).toFixed(1);
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
      <div style={{
        flex: 1, height: 10, background: "#e2e8f0", borderRadius: 5, overflow: "hidden"
      }}>
        <div style={{
          width: `${pct}%`, height: "100%",
          background: value >= 0.95 ? "#22c55e" : value >= 0.85 ? "#3b82f6" : "#f59e0b",
          borderRadius: 5, transition: "width 0.6s ease"
        }} />
      </div>
      <span style={{ minWidth: 42, fontSize: 13, fontWeight: 600 }}>{pct}%</span>
    </div>
  );
}

function VectorizerCard({ title, color, data }) {
  if (!data) return null;
  const classes = Object.keys(data.report);
  return (
    <div style={{
      flex: 1, border: `2px solid ${color}`, borderRadius: 12, padding: "20px 22px",
      background: "#fff", minWidth: 260,
    }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 14 }}>
        <h3 style={{ margin: 0, color, fontSize: 16, fontWeight: 700 }}>{title}</h3>
        <span style={{
          background: color, color: "#fff", borderRadius: 20,
          padding: "3px 12px", fontSize: 13, fontWeight: 600,
        }}>
          F1 {(data.f1_score * 100).toFixed(2)}%
        </span>
      </div>
      <p style={{ margin: "0 0 14px", fontSize: 13, color: "#64748b" }}>
        Best model: <strong>{data.best_model}</strong>
      </p>

      {classes.map((cls) => (
        <div key={cls} style={{ marginBottom: 12 }}>
          <p style={{ margin: "0 0 6px", fontSize: 12, fontWeight: 600, textTransform: "uppercase", color: "#94a3b8" }}>
            Class: {cls === "0" ? "Ham" : cls === "1" ? "Spam" : cls}
          </p>
          {METRICS.map((m) => (
            <div key={m} style={{ display: "grid", gridTemplateColumns: "80px 1fr", alignItems: "center", gap: 6, marginBottom: 4 }}>
              <span style={{ fontSize: 12, color: "#64748b", textTransform: "capitalize" }}>{m}</span>
              <MetricBar value={data.report[cls][m]} />
            </div>
          ))}
        </div>
      ))}
    </div>
  );
}

function ComparisonRow({ label, tfidfVal, w2vVal }) {
  const better = tfidfVal >= w2vVal ? "tfidf" : "w2v";
  return (
    <tr>
      <td style={{ padding: "8px 12px", fontWeight: 600, color: "#374151" }}>{label}</td>
      <td style={{
        padding: "8px 12px", textAlign: "center",
        background: better === "tfidf" ? "#dcfce7" : "transparent",
        fontWeight: better === "tfidf" ? 700 : 400,
      }}>
        {(tfidfVal * 100).toFixed(2)}%
        {better === "tfidf" && <span style={{ marginLeft: 4, color: "#16a34a" }}>▲</span>}
      </td>
      <td style={{
        padding: "8px 12px", textAlign: "center",
        background: better === "w2v" ? "#dcfce7" : "transparent",
        fontWeight: better === "w2v" ? 700 : 400,
      }}>
        {(w2vVal * 100).toFixed(2)}%
        {better === "w2v" && <span style={{ marginLeft: 4, color: "#16a34a" }}>▲</span>}
      </td>
    </tr>
  );
}

export default function TrainModel({ onTrained }) {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  const train = async () => {
    setLoading(true);
    setResult(null);
    setError("");
    try {
      const res = await fetch("/api/train", { method: "POST" });
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      setResult(data);
      onTrained();
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const tfidf = result?.comparison?.tfidf;
  const w2v   = result?.comparison?.word2vec;

  const aggMetric = (data, metric) => {
    if (!data) return 0;
    const vals = Object.values(data.report).map((m) => m[metric]);
    return vals.reduce((a, b) => a + b, 0) / vals.length;
  };

  return (
    <div>
      <h2 className="section-title">Train Model</h2>
      <p className="subtitle">
        Trains Naive Bayes, Logistic Regression, SVM, and Random Forest using both
        <strong> TF-IDF</strong> and <strong> Word2Vec (CBOW)</strong>, then compares their performance side by side.
      </p>

      <button className="btn-primary" onClick={train} disabled={loading} style={{ marginTop: 16 }}>
        {loading ? "Training… (may take a minute or two)" : "🔄 Train & Compare"}
      </button>

      {error && <p className="error-msg">{error}</p>}

      {result && (
        <>
          {/* ── Summary banner ── */}
          <div className="train-result" style={{ marginTop: 24 }}>
            <div className="train-summary">
              <span>Active model: <strong>{result.best_model}</strong> (TF-IDF)</span>
              <span>F1 Score: <strong>{(result.f1_score * 100).toFixed(2)}%</strong></span>
            </div>
          </div>

          {/* ── Head-to-head comparison table ── */}
          {tfidf && w2v && (
            <div style={{ marginTop: 28 }}>
              <h3 style={{ marginBottom: 12, fontSize: 16, color: "#1e293b" }}>
                ⚔️ TF-IDF vs Word2Vec — Head-to-Head
              </h3>
              <div style={{ overflowX: "auto" }}>
                <table style={{
                  width: "100%", borderCollapse: "collapse", fontSize: 14,
                  border: "1px solid #e2e8f0", borderRadius: 8, overflow: "hidden",
                }}>
                  <thead>
                    <tr style={{ background: "#f1f5f9" }}>
                      <th style={{ padding: "10px 12px", textAlign: "left" }}>Metric (avg)</th>
                      <th style={{ padding: "10px 12px", textAlign: "center", color: "#3b82f6" }}>📊 TF-IDF</th>
                      <th style={{ padding: "10px 12px", textAlign: "center", color: "#8b5cf6" }}>🧠 Word2Vec</th>
                    </tr>
                  </thead>
                  <tbody>
                    {METRICS.map((m) => (
                      <ComparisonRow
                        key={m}
                        label={m.charAt(0).toUpperCase() + m.slice(1)}
                        tfidfVal={aggMetric(tfidf, m)}
                        w2vVal={aggMetric(w2v, m)}
                      />
                    ))}
                    <tr style={{ background: "#f8fafc", borderTop: "2px solid #e2e8f0" }}>
                      <td style={{ padding: "8px 12px", fontWeight: 700 }}>Best Model</td>
                      <td style={{ padding: "8px 12px", textAlign: "center", color: "#3b82f6", fontWeight: 600 }}>{tfidf.best_model}</td>
                      <td style={{ padding: "8px 12px", textAlign: "center", color: "#8b5cf6", fontWeight: 600 }}>{w2v.best_model}</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* ── Side-by-side detailed cards ── */}
          <div style={{ display: "flex", gap: 20, marginTop: 28, flexWrap: "wrap" }}>
            <VectorizerCard title="📊 TF-IDF" color="#3b82f6" data={tfidf} />
            <VectorizerCard title="🧠 Word2Vec (CBOW)" color="#8b5cf6" data={w2v} />
          </div>
        </>
      )}
    </div>
  );
}

import React, { useState } from "react";
import "./App.css";

const typeOptions = ["L", "M", "H"];

function App() {
  const [formData, setFormData] = useState({
    product_id: "",
    type: typeOptions[0],
    air_temperature: "",
    process_temperature: "",
    rotational_speed: "",
    torque: "",
    tool_wear: ""
  });

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setResult(null);

    try {
      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ...formData,
          air_temperature: parseFloat(formData.air_temperature),
          process_temperature: parseFloat(formData.process_temperature),
          rotational_speed: parseFloat(formData.rotational_speed),
          torque: parseFloat(formData.torque),
          tool_wear: parseFloat(formData.tool_wear)
        })
      });

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || "Error predicting failure");
      }

      const data = await response.json();
      setResult(data);
    } catch (error) {
      setResult({ message: error.message, isError: true });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <div className="layout">

        {/* LEFT SIDE: FEATURE IMPORTANCE */}
        <div className="card insights-card">
          <div className="card-header">
            <h2>Model Insights</h2>
            <p className="insights-desc">
              Sensor impact on failure probability.
            </p>
          </div>
          <div className="image-wrapper">
            <img
              src="/feature_importance.png"
              alt="Feature Importance Chart"
              className="importance-img"
            />
          </div>
          <div className="model-info">
             <small>Model Type: LightGBM Classifier</small>
          </div>
        </div>

        {/* RIGHT SIDE: FORM + RESULT */}
        <div className="main-content">
          <div className="card">
            <header className="header">
              <h1>Machine Failure Prediction</h1>
              <p>Input operational metrics to analyze machine health</p>
            </header>

            <form onSubmit={handleSubmit} className="form-grid">
              <div className="form-group full-width">
                <label>Product ID</label>
                <input
                  type="text"
                  name="product_id"
                  placeholder="e.g. M14860"
                  value={formData.product_id}
                  onChange={handleChange}
                  required
                />
              </div>

              <div className="form-group">
                <label>Machine Type</label>
                <select name="type" value={formData.type} onChange={handleChange}>
                  {typeOptions.map((t) => (
                    <option key={t} value={t}>{t}</option>
                  ))}
                </select>
              </div>

              {["air_temperature", "process_temperature", "rotational_speed", "torque", "tool_wear"].map((key) => (
                <div className="form-group" key={key}>
                  <label>{key.replace("_", " ").toUpperCase()}</label>
                  <input
                    type="number"
                    name={key}
                    value={formData[key]}
                    onChange={handleChange}
                    step="any"
                    required
                    placeholder="0.00"
                  />
                </div>
              ))}

              <button type="submit" className="submit-btn" disabled={loading}>
                {loading ? "Processing..." : "Analyze System Health"}
              </button>
            </form>

            {result && (
              <div
                className={`result-card ${
                  result.prediction ? "failure" : "success"
                } ${result.isError ? "error" : ""}`}
              >
                <div className="result-header">
                  <h3>{result.isError ? "Error" : "Analysis Report"}</h3>
                  <span className="status-badge">
                    {result.prediction ? "Failure Detected" : "Healthy"}
                  </span>
                </div>

                <p className="result-message">{result.message}</p>

                {result.prediction !== undefined && (
                  <div className="probability-container">
                    <div className="prob-text">
                      <span>Confidence Score</span>
                      <span>{(result.probability * 100).toFixed(1)}%</span>
                    </div>
                    <div className="progress-bar">
                      <div
                        className="progress-fill"
                        style={{ width: `${result.probability * 100}%` }}
                      />
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>

      </div>
    </div>
  );
}

export default App;
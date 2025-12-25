import { useState } from "react";
import "./App.css";

const typeOptions = ["L", "M", "H"];

function App() {
  const [formData, setFormData] = useState({
    product_id: "", type: "L", air_temperature: "",
    process_temperature: "", rotational_speed: "",
    torque: "", tool_wear: ""
  });

  const [result, setResult] = useState(null);
  const [batchResults, setBatchResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);

  const delay = (ms) => new Promise(res => setTimeout(res, ms));

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({ ...prev, [name]: value }));
  };

  const handleSingleSubmit = async (e) => {
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
      const data = await response.json();
      setResult(data);
    } catch (error) {
      alert("Single prediction failed: " + error.message);
    } finally {
      setLoading(false);
    }
  };

  const handleFileChange = (e) => {
    setSelectedFile(e.target.files[0]);
    setBatchResults([]);
  };

  const handleStreamPredict = async () => {
    if (!selectedFile) return;
    setLoading(true);
    setBatchResults([]); 

    const reader = new FileReader();
    reader.onload = async (event) => {
      try {
        const text = event.target.result;
        const lines = text.split(/\r?\n/).filter(line => line.trim() !== "");
        const rows = lines.slice(1); 

        // One-by-one streaming loop
        for (let i = 0; i < rows.length; i++) {
          const rowData = rows[i].split(",").map(item => item.trim());
          if (rowData.length < 7) continue;

          const singleRowPayload = [{
            product_id: rowData[0], 
            type: rowData[1],
            air_temperature: Number(rowData[2]), 
            process_temperature: Number(rowData[3]),
            rotational_speed: Number(rowData[4]), 
            torque: Number(rowData[5]),
            tool_wear: Number(rowData[6])
          }];

          const response = await fetch("http://127.0.0.1:8000/batch_predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(singleRowPayload)
          });

          if (!response.ok) throw new Error(`Failed at row ${i + 1}`);
          
          const newResult = await response.json();
          setBatchResults(prev => [...prev, ...newResult]);
          
          // 1 second delay per row
          await delay(1000);
        }
      } catch (error) {
        alert("Streaming failed: " + error.message);
      } finally {
        setLoading(false);
      }
    };
    reader.readAsText(selectedFile);
  };

  return (
    <div className="dashboard-container">
      <header className="header">
        <h1>Machine Failure Prediction</h1>
      </header>

      <div className="dashboard-grid">
        {/* LEFT: FEATURE IMPORTANCE */}
        <section className="card side-panel">
          <h3>Model Insights</h3>
          <div className="image-container">
            <img src="http://127.0.0.1:8000/static/feature_importance.png" alt="Importance Graph" className="importance-plot" />
          </div>
          <p className="description">Key sensor drivers for failure predictions.</p>
        </section>

        {/* MIDDLE: SINGLE PREDICTION */}
        <section className="card main-panel">
          <h3>Single Prediction</h3>
          <form onSubmit={handleSingleSubmit} className="prediction-form">
            <div className="single-form-grid">
              <label>Product ID</label>
              <input name="product_id" placeholder="L47230" onChange={handleChange} required />
              
              <label>Type</label>
              <select name="type" onChange={handleChange}>
                {typeOptions.map(t => <option key={t} value={t}>{t}</option>)}
              </select>
              
              <label>Air Temp (K)</label>
              <input type="number" step="0.1" name="air_temperature" onChange={handleChange} required />
              
              <label>Process Temp (K)</label>
              <input type="number" step="0.1" name="process_temperature" onChange={handleChange} required />
              
              <label>Speed (RPM)</label>
              <input type="number" name="rotational_speed" onChange={handleChange} required />
              
              <label>Torque (Nm)</label>
              <input type="number" step="0.1" name="torque" onChange={handleChange} required />
              
              <label>Tool Wear (min)</label>
              <input type="number" name="tool_wear" onChange={handleChange} required />
            </div>
            <button type="submit" className="btn-primary" disabled={loading}>Run Single</button>
          </form>
          {result && (
            <div className={`result-alert ${result.prediction === 1 ? "danger" : "success"}`}>
              {result.message} ({(result.probability * 100).toFixed(2)}%)
            </div>
          )}
        </section>

        {/* RIGHT: BATCH PREDICTION */}
        <section className="card side-panel">
          <h3>Batch Streaming</h3>
          <div className="file-upload-area">
            <input type="file" accept=".csv" onChange={handleFileChange} id="csv-upload" />
            <label htmlFor="csv-upload" className="file-label">
              {selectedFile ? selectedFile.name : "Upload CSV File"}
            </label>
          </div>
          <button className="btn-batch" onClick={handleStreamPredict} disabled={!selectedFile || loading}>
            {loading ? "Streaming..." : "Start Prediction Stream"}
          </button>
          
          <div className="mini-table-wrapper">
            <table className="results-table">
              <thead>
                <tr>
                  <th>Record Number</th>
                  <th>Status</th>
                  <th>Failure Probability</th>
                </tr>
              </thead>
              <tbody>
                {batchResults.map((res, i) => (
                  <tr key={i} className={res.prediction === 1 ? "row-fail" : "row-pass"}>
                    <td>{i + 1}</td>
                    <td>{res.message}</td>
                    <td>{(res.probability * 100).toFixed(2)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      </div>
    </div>
  );
}

export default App;
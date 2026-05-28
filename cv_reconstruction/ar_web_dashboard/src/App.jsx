import { useState } from 'react';
import './index.css';

function App() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImage(file);
      setPreview(URL.createObjectURL(file));
      setResult(null);
    }
  };

  const handleUpload = async () => {
    if (!image) return;
    
    setLoading(true);
    const formData = new FormData();
    formData.append('image', image);

    try {
      const response = await fetch('http://localhost:8000/api_identify_location', {
        method: 'POST',
        body: formData,
      });
      
      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error('Error uploading image:', error);
      setResult({ error: "Failed to connect to API server. Make sure the FastAPI backend is running on port 8000." });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <header className="header">
        <div className="logo-glow"></div>
        <h1>CV Spatial Computing</h1>
        <p>Visual Positioning System (VPS) Testing Dashboard</p>
      </header>

      <main className="glass-panel">
        <div className="upload-section">
          <label className="file-upload">
            <input type="file" accept="image/*" onChange={handleFileChange} />
            <span className="upload-icon">📷</span>
            <span className="upload-text">{image ? image.name : "Select Campus Image"}</span>
          </label>
          
          <button 
            className={`analyze-btn ${!image || loading ? 'disabled' : ''}`}
            onClick={handleUpload}
            disabled={!image || loading}
          >
            {loading ? "Analyzing..." : "Locate via VPS"}
          </button>
        </div>

        <div className="content-grid">
          <div className="preview-card">
            <h3>Camera Feed Preview</h3>
            <div className="image-container">
              {preview ? (
                <img src={preview} alt="Preview" className="preview-image" />
              ) : (
                <div className="empty-state">No image selected</div>
              )}
            </div>
          </div>

          <div className="result-card">
            <h3>VPS Output</h3>
            <div className="data-container">
              {loading && <div className="loader"></div>}
              {!loading && !result && <div className="empty-state">Awaiting API response...</div>}
              
              {!loading && result && result.status === "success" && (
                <div className="results">
                  <div className="metric">
                    <span className="label">Location ID:</span>
                    <span className="value highlight">{result.location_id}</span>
                  </div>
                  
                  <div className="metric">
                    <span className="label">Translation (xyz):</span>
                    <div className="matrix">
                      [{result.translation.map(v => v.toFixed(2)).join(', ')}]
                    </div>
                  </div>
                  
                  <div className="metric">
                    <span className="label">Rotation Matrix:</span>
                    <div className="matrix grid">
                      {result.rotation.map((row, i) => (
                        <div key={i}>[{row.map(v => v.toFixed(2)).join(', ')}]</div>
                      ))}
                    </div>
                  </div>

                  {result.image_base64 && (
                    <div className="metric">
                      <span className="label">Object Detection (YOLOv8):</span>
                      <img 
                        src={`data:image/jpeg;base64,${result.image_base64}`} 
                        alt="YOLO Output" 
                        className="output-image"
                      />
                    </div>
                  )}
                </div>
              )}

              {!loading && result && result.error && (
                <div className="error-msg">{result.error}</div>
              )}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;

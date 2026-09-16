import { useState } from "react";
import { UploadCloud, FileText, CheckCircle, ShieldAlert, ArrowRight } from "lucide-react";
import { Link } from "react-router-dom";

export default function Upload() {
  const [selectedFile, setSelectedFile] = useState(null);

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0]);
    }
  };

  return (
    <div className="page-container">
      <div className="page-header">
        <h1 className="page-title">Dataset Ingestion & Diagnostics</h1>
        <p className="page-subtitle">
          Upload tabular CSV data to automatically infer column schemas, detect data leakage, and inspect quality diagnostics.
        </p>
      </div>

      <div className="card-grid">
        <div className="studio-card upload-box">
          <div className="dropzone">
            <UploadCloud size={48} className="upload-icon" />
            <h3 className="dropzone-title">Drag & drop your dataset here</h3>
            <p className="dropzone-subtitle">Supports CSV files up to 50MB</p>
            <input 
              type="file" 
              accept=".csv" 
              id="csv-upload" 
              className="file-input-hidden" 
              onChange={handleFileChange}
            />
            <label htmlFor="csv-upload" className="btn btn-primary mt-4">
              Browse CSV File
            </label>
            {selectedFile && (
              <div className="selected-file-badge">
                <FileText size={16} />
                <span>{selectedFile.name} ({(selectedFile.size / 1024).toFixed(1)} KB)</span>
              </div>
            )}
          </div>
        </div>

        <div className="studio-card">
          <h2 className="card-heading">Phase 1 Pipeline Checklist</h2>
          <div className="checklist">
            <div className="checklist-item">
              <CheckCircle className="text-success" size={20} />
              <div>
                <strong>SHA-256 Hashing & Versioning</strong>
                <p>Detects duplicate uploads and tracks dataset versions over time.</p>
              </div>
            </div>
            <div className="checklist-item">
              <CheckCircle className="text-success" size={20} />
              <div>
                <strong>Automatic Schema Inference</strong>
                <p>Identifies numerical, categorical, and datetime types, null percentages, and cardinality.</p>
              </div>
            </div>
            <div className="checklist-item">
              <ShieldAlert className="text-warning" size={20} />
              <div>
                <strong>Rule-Based Diagnostics Engine</strong>
                <p>Flags class imbalances, high-cardinality features, and potential target leakage.</p>
              </div>
            </div>
          </div>

          <div className="card-actions mt-6">
            <Link to="/pipeline" className="btn btn-secondary flex items-center gap-2">
              <span>Go to Pipeline Studio</span>
              <ArrowRight size={16} />
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}

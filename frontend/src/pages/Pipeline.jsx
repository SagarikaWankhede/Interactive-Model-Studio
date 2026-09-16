import { Sliders, Cpu, Code2, ArrowRight } from "lucide-react";
import { Link } from "react-router-dom";

export default function Pipeline() {
  return (
    <div className="page-container">
      <div className="page-header">
        <h1 className="page-title">Pipeline Studio & Model Builder</h1>
        <p className="page-subtitle">
          Construct reproducible preprocessing recipes, configure machine learning estimators, and generate clean Python code.
        </p>
      </div>

      <div className="features-grid">
        <div className="studio-card">
          <div className="card-icon-header">
            <Sliders className="text-primary" size={24} />
            <h2 className="card-heading">Preprocessing Recipe</h2>
          </div>
          <p className="card-description">
            Configure column-level transformations including SimpleImputer, OneHotEncoder, StandardScaler, and train/test splits.
          </p>
          <div className="status-tag">Phase 1 Integration Ready</div>
        </div>

        <div className="studio-card">
          <div className="card-icon-header">
            <Cpu className="text-accent" size={24} />
            <h2 className="card-heading">Model Configuration</h2>
          </div>
          <p className="card-description">
            Tune hyperparameters dynamically for Random Forest, Logistic Regression, XGBoost, and Decision Trees with instant validation.
          </p>
          <div className="status-tag">Phase 2 Integration Ready</div>
        </div>

        <div className="studio-card">
          <div className="card-icon-header">
            <Code2 className="text-success" size={24} />
            <h2 className="card-heading">Python Codegen (Differentiator #3)</h2>
          </div>
          <p className="card-description">
            Export the assembled visual pipeline directly into standalone, executable Python scripts using standard scikit-learn.
          </p>
          <div className="status-tag">Phase 4 Flagship Feature</div>
        </div>
      </div>

      <div className="mt-8 flex justify-end">
        <Link to="/dashboard" className="btn btn-primary flex items-center gap-2">
          <span>View Experiments in Dashboard</span>
          <ArrowRight size={16} />
        </Link>
      </div>
    </div>
  );
}

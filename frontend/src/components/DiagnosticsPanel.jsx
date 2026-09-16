import { useState } from "react";
import { 
  AlertTriangle, 
  AlertOctagon, 
  CheckCircle2, 
  Lightbulb, 
  Filter,
  ShieldCheck
} from "lucide-react";

export default function DiagnosticsPanel({ diagnostics = [] }) {
  const [filter, setFilter] = useState("all"); // 'all' | 'red' | 'yellow'

  const filtered = diagnostics.filter((diag) => {
    if (filter === "all") return true;
    return diag.severity === filter;
  });

  const redCount = diagnostics.filter((d) => d.severity === "red").length;
  const yellowCount = diagnostics.filter((d) => d.severity === "yellow").length;

  return (
    <div className="diagnostics-container">
      <div className="diagnostics-header">
        <div className="flex items-center gap-3">
          <div className="diagnostics-title-icon">
            {redCount > 0 ? (
              <AlertOctagon className="text-danger" size={24} />
            ) : yellowCount > 0 ? (
              <AlertTriangle className="text-warning" size={24} />
            ) : (
              <ShieldCheck className="text-success" size={24} />
            )}
          </div>
          <div>
            <h3 className="diagnostics-title">Data Quality & Leakage Diagnostics</h3>
            <p className="diagnostics-subtitle">
              Automated checks for target leakage, high cardinality, severe class imbalance, and null ratios.
            </p>
          </div>
        </div>

        <div className="diagnostics-filters">
          <button 
            className={`diag-filter-btn ${filter === "all" ? "active" : ""}`}
            onClick={() => setFilter("all")}
          >
            All ({diagnostics.length})
          </button>
          <button 
            className={`diag-filter-btn ${filter === "red" ? "active red" : ""}`}
            onClick={() => setFilter("red")}
          >
            <span className="dot red-dot"></span>
            Critical ({redCount})
          </button>
          <button 
            className={`diag-filter-btn ${filter === "yellow" ? "active yellow" : ""}`}
            onClick={() => setFilter("yellow")}
          >
            <span className="dot yellow-dot"></span>
            Warnings ({yellowCount})
          </button>
        </div>
      </div>

      {filtered.length === 0 ? (
        <div className="empty-diagnostics">
          <CheckCircle2 size={36} className="text-success mb-2" />
          <p className="font-semibold text-gray-700">No diagnostic warnings for the selected filter!</p>
          <p className="text-sm text-gray-500">Your dataset meets baseline quality thresholds.</p>
        </div>
      ) : (
        <div className="diagnostics-list">
          {filtered.map((diag) => {
            const isRed = diag.severity === "red";
            return (
              <div 
                key={diag.id || diag.title} 
                className={`diag-card ${isRed ? "diag-card-red" : "diag-card-yellow"}`}
              >
                <div className="diag-card-icon">
                  {isRed ? (
                    <AlertOctagon size={20} className="text-danger" />
                  ) : (
                    <AlertTriangle size={20} className="text-warning" />
                  )}
                </div>

                <div className="diag-card-content">
                  <div className="diag-card-top">
                    <span className="diag-card-title">{diag.title}</span>
                    {diag.column && (
                      <span className="diag-column-tag">
                        Column: <code>{diag.column}</code>
                      </span>
                    )}
                  </div>

                  <p className="diag-description">{diag.description}</p>

                  {diag.suggested_fix && (
                    <div className="diag-suggestion">
                      <Lightbulb size={16} className="suggestion-icon" />
                      <span><strong>Suggested Action:</strong> {diag.suggested_fix}</span>
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

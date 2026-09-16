import { Table, CheckCircle2, XCircle } from "lucide-react";

export default function ConfusionMatrixChart({ experiments = [] }) {
  if (!experiments || experiments.length === 0) {
    return <div className="text-muted text-sm text-center py-4">No experiments selected for confusion matrix comparison.</div>;
  }

  return (
    <div className="confusion-matrices-container">
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {experiments.map((exp) => {
          const matrix = exp.confusion_matrix || [
            [950, 50],
            [60, 340],
          ];

          // matrix: [[TN, FP], [FN, TP]]
          const tn = matrix[0]?.[0] ?? 0;
          const fp = matrix[0]?.[1] ?? 0;
          const fn = matrix[1]?.[0] ?? 0;
          const tp = matrix[1]?.[1] ?? 0;
          const total = tn + fp + fn + tp || 1;

          const getBgStyle = (val, isCorrect) => {
            const ratio = val / total;
            const opacity = Math.min(0.9, Math.max(0.12, ratio * 1.8));
            if (isCorrect) {
              return { backgroundColor: `rgba(16, 185, 129, ${opacity})` };
            }
            return { backgroundColor: `rgba(239, 68, 68, ${opacity})` };
          };

          return (
            <div key={exp.id} className="matrix-card">
              <div className="matrix-header">
                <strong className="text-sm block">{exp.model_name}</strong>
                <span className="text-xs text-muted">Accuracy: {((exp.metrics?.accuracy || 0) * 100).toFixed(1)}%</span>
              </div>

              <div className="matrix-grid-wrapper">
                <div className="matrix-axis-y">
                  <span className="axis-label-vertical">Actual</span>
                </div>

                <div className="matrix-content">
                  <div className="matrix-axis-x">
                    <span className="axis-label">Predicted Negative</span>
                    <span className="axis-label">Predicted Positive</span>
                  </div>

                  <div className="matrix-grid">
                    {/* Row 1: Actual Negative */}
                    <div className="matrix-row">
                      <span className="row-label">Actual Neg</span>
                      <div className="matrix-cell" style={getBgStyle(tn, true)}>
                        <span className="cell-type">TN (True Neg)</span>
                        <strong className="cell-val">{tn.toLocaleString()}</strong>
                        <span className="cell-pct">{((tn / total) * 100).toFixed(1)}%</span>
                      </div>
                      <div className="matrix-cell" style={getBgStyle(fp, false)}>
                        <span className="cell-type">FP (False Pos)</span>
                        <strong className="cell-val text-danger">{fp.toLocaleString()}</strong>
                        <span className="cell-pct">{((fp / total) * 100).toFixed(1)}%</span>
                      </div>
                    </div>

                    {/* Row 2: Actual Positive */}
                    <div className="matrix-row">
                      <span className="row-label">Actual Pos</span>
                      <div className="matrix-cell" style={getBgStyle(fn, false)}>
                        <span className="cell-type">FN (False Neg)</span>
                        <strong className="cell-val text-danger">{fn.toLocaleString()}</strong>
                        <span className="cell-pct">{((fn / total) * 100).toFixed(1)}%</span>
                      </div>
                      <div className="matrix-cell" style={getBgStyle(tp, true)}>
                        <span className="cell-type">TP (True Pos)</span>
                        <strong className="cell-val">{tp.toLocaleString()}</strong>
                        <span className="cell-pct">{((tp / total) * 100).toFixed(1)}%</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              <div className="matrix-footer">
                <span className="text-xs text-muted">Total Test Samples: {total.toLocaleString()}</span>
                <span className="text-xs font-semibold text-emerald-700">
                  Correct: {(((tn + tp) / total) * 100).toFixed(1)}%
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

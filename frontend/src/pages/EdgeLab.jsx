import { useCallback, useEffect, useMemo, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import {
  Cpu, Download, Loader2, CheckCircle2, AlertTriangle, AlertCircle, Zap, ShieldCheck, Inbox, ScanSearch,
  RefreshCw, Check, X, MonitorSmartphone, Workflow,
} from "lucide-react";
import api from "../api/client";
import { createSession, predict } from "../services/edgeRuntime";

const kb = (b) => (b < 1024 ? `${b} B` : `${(b / 1024).toFixed(1)} KB`);
const pct = (v) => `${(v * 100).toFixed(1)}%`;
const METRICS = {
  classification: [["accuracy", "Accuracy"], ["precision_macro", "Precision"], ["recall_macro", "Recall"], ["f1_macro", "F1"]],
  regression: [["rmse", "RMSE"], ["mae", "MAE"], ["r2", "R²"]],
};

// ---------------------------------------------------------------------------
// Deploy tab
// ---------------------------------------------------------------------------
function DeployTab({ runId, threshold, setThreshold }) {
  const [info, setInfo] = useState(null);
  const [quantize, setQuantize] = useState(false);
  const [converting, setConverting] = useState(false);
  const [error, setError] = useState(null);
  const [runtime, setRuntime] = useState(null);
  const [loadingModel, setLoadingModel] = useState(false);
  const [values, setValues] = useState({});
  const [prediction, setPrediction] = useState(null);
  const [history, setHistory] = useState([]);
  const [autoQueue, setAutoQueue] = useState(false);
  const [queued, setQueued] = useState(null);

  const spec = info?.spec;
  const resetInputs = useCallback((s) => setValues(Object.fromEntries((s?.inputs || []).map((i) => [i.name, i.example ?? ""]))), []);

  // The parent remounts this tab per run (key={runId}), so state starts fresh.
  useEffect(() => {
    if (!runId) return;
    api.getBrowserExport(runId).then((r) => { setInfo(r); resetInputs(r.spec); }).catch(() => {});
  }, [runId, resetInputs]);

  const convert = async () => {
    setConverting(true); setError(null); setRuntime(null); setPrediction(null);
    try {
      const r = await api.convertForBrowser(runId, { quantize });
      setInfo(r); resetInputs(r.spec);
    } catch (e) {
      setError(e.message);
    } finally {
      setConverting(false);
    }
  };

  const loadModel = async () => {
    setLoadingModel(true); setError(null);
    try {
      const t0 = performance.now();
      const bytes = await api.fetchOnnxModel(runId);
      const downloadMs = performance.now() - t0;
      const rt = await createSession(bytes);
      setRuntime({ ...rt, downloadMs, bytes: bytes.byteLength });
    } catch (e) {
      setError(`Could not start the in-browser runtime: ${e.message}`);
    } finally {
      setLoadingModel(false);
    }
  };

  const sendToQueue = async (p, features) => {
    const res = await api.addEdgeCase({
      run_id: runId, source: "browser", features,
      predicted_label: p.label, confidence: p.confidence, probabilities: p.probabilities,
    });
    setQueued(res.id);
  };

  const runPrediction = async () => {
    setError(null); setQueued(null);
    try {
      const raw = Object.fromEntries(spec.inputs.map((i) => [i.name, i.type === "numeric" && values[i.name] !== "" ? Number(values[i.name]) : values[i.name]]));
      const p = await predict(runtime, spec, raw);
      setPrediction({ ...p, raw });
      setHistory((h) => [{ ...p, raw, at: new Date() }, ...h].slice(0, 8));
      if (autoQueue && p.confidence !== undefined && p.confidence < threshold) await sendToQueue(p, raw);
    } catch (e) {
      setError(`Prediction failed: ${e.message}`);
    }
  };

  if (!runId) return <p className="st-muted">Choose a trained run above.</p>;
  const lowConfidence = prediction?.confidence !== undefined && prediction.confidence < threshold;

  return (
    <div className="el-grid">
      <section className="studio-card">
        <h2 className="card-heading"><Cpu size={17} /> 1 · Optimize for the edge</h2>
        <div className="el-options">
          <label className="st-check"><input type="checkbox" checked readOnly disabled /> ONNX export <span className="st-muted">— portable model format</span></label>
          <label className="st-check"><input type="checkbox" checked={quantize} onChange={(e) => setQuantize(e.target.checked)} /> INT8 quantization <span className="st-muted">— applied only where it helps</span></label>
          <label className="st-check"><input type="checkbox" checked readOnly disabled /> Compression <span className="st-muted">— gzip transfer size</span></label>
          <label className="st-check"><input type="checkbox" checked readOnly disabled /> Browser deployment <span className="st-muted">— onnxruntime-web (WebAssembly)</span></label>
        </div>
        <button className="btn btn-primary" onClick={convert} disabled={converting}>
          {converting ? <Loader2 size={15} className="animate-spin" /> : <Zap size={15} />} {info ? "Re-convert" : "Convert for browser"}
        </button>

        {info && (
          <div className="el-report">
            <div className="el-stats">
              <div><span>Model file</span><strong>{kb(info.size_bytes)}</strong></div>
              <div><span>Over the network (gzip)</span><strong>{kb(info.gzip_bytes)}</strong></div>
              <div><span>Serving</span><strong>{info.serving === "int8" ? "INT8" : "FP32"}</strong></div>
            </div>
            {info.parity && (
              <div className={`el-parity ${info.parity.passed ? "ok" : info.parity.passed === false ? "bad" : ""}`}>
                {info.parity.passed ? <ShieldCheck size={15} /> : <AlertTriangle size={15} />}
                <span>
                  <strong>Parity check:</strong>{" "}
                  {info.parity.value === null
                    ? `unavailable (${info.parity.message || "onnxruntime missing"})`
                    : `${info.parity.metric} with the original Python model = ${info.parity.metric === "label agreement" ? pct(info.parity.value) : info.parity.value} on ${info.parity.rows} rows.`}
                </span>
              </div>
            )}
            {info.quantization && <p className="st-muted el-quant">{info.quantization.message}</p>}
            <div className="el-downloads">
              <a className="btn btn-secondary btn-sm" href={api.onnxModelUrl(runId)}><Download size={14} /> model.onnx</a>
              <a className="btn btn-secondary btn-sm" href={api.onnxSpecUrl(runId)}><Download size={14} /> preprocessing.json</a>
            </div>
          </div>
        )}
        {error && <div className="st-banner error"><AlertCircle size={15} /> {error}</div>}
      </section>

      <section className="studio-card">
        <h2 className="card-heading"><MonitorSmartphone size={17} /> 2 · Run in this browser</h2>
        {!info ? (
          <p className="st-muted">Convert the model first.</p>
        ) : !runtime ? (
          <>
            <p className="st-muted">Downloads <code>model.onnx</code> once, then every prediction is computed on this device. Your inputs are not sent to the server.</p>
            <button className="btn btn-primary" onClick={loadModel} disabled={loadingModel}>
              {loadingModel ? <Loader2 size={15} className="animate-spin" /> : <Download size={15} />} Load model in browser
            </button>
          </>
        ) : (
          <>
            <div className="el-runtime">
              <CheckCircle2 size={14} /> Loaded {kb(runtime.bytes)} · download {runtime.downloadMs.toFixed(0)} ms · session start {runtime.loadMs.toFixed(0)} ms
            </div>
            <div className="el-form">
              {spec.inputs.map((input) => (
                <label key={input.name} className="st-field">
                  <span className="st-field-label">{input.name}</span>
                  {input.type === "numeric" ? (
                    <input type="number" className="form-input" value={values[input.name] ?? ""} placeholder="missing" onChange={(e) => setValues({ ...values, [input.name]: e.target.value })} />
                  ) : (
                    <select className="form-select" value={values[input.name] ?? ""} onChange={(e) => setValues({ ...values, [input.name]: e.target.value })}>
                      <option value="">(missing)</option>
                      {input.options.map((o) => <option key={String(o)} value={o}>{String(o)}</option>)}
                    </select>
                  )}
                </label>
              ))}
            </div>
            <div className="el-predict-row">
              <button className="btn btn-primary" onClick={runPrediction}><Zap size={15} /> Predict locally</button>
              <button className="btn btn-secondary btn-sm" onClick={() => resetInputs(spec)}>Reset inputs</button>
            </div>

            {prediction && (
              <div className={`el-prediction ${lowConfidence ? "low" : ""}`}>
                {prediction.label !== undefined ? (
                  <>
                    <div className="el-pred-main">
                      <span>{spec.target_column}</span>
                      <strong>{prediction.label}</strong>
                      <em>{pct(prediction.confidence)} confident</em>
                    </div>
                    {Object.entries(prediction.probabilities).sort((a, b) => b[1] - a[1]).map(([cls, p]) => (
                      <div key={cls} className="el-prob"><span>{cls}</span><span className="dd-bar"><span style={{ width: pct(p) }} /></span><span>{pct(p)}</span></div>
                    ))}
                  </>
                ) : (
                  <div className="el-pred-main"><span>{spec.target_column}</span><strong>{prediction.value.toFixed(3)}</strong></div>
                )}
                <div className="st-muted">Inference took {prediction.inferenceMs.toFixed(2)} ms on this device (plus the one-time load above).</div>
                {lowConfidence && (
                  <div className="el-low">
                    <AlertTriangle size={14} /> Below the {pct(threshold)} confidence threshold — a good example for a human to label.
                    {queued ? (
                      <span className="el-queued"><Check size={13} /> Sent to the edge-case queue (#{queued})</span>
                    ) : (
                      <button className="btn btn-secondary btn-sm" onClick={() => sendToQueue(prediction, prediction.raw)}><Inbox size={13} /> Send to edge-case queue</button>
                    )}
                  </div>
                )}
              </div>
            )}

            {spec.task_type === "classification" && (
              <div className="el-threshold">
                <label className="st-field">
                  <span className="st-field-label">Low-confidence threshold: {pct(threshold)}</span>
                  <input type="range" min="0.5" max="0.95" step="0.05" value={threshold} onChange={(e) => setThreshold(Number(e.target.value))} />
                </label>
                <label className="st-check">
                  <input type="checkbox" checked={autoQueue} onChange={(e) => setAutoQueue(e.target.checked)} />
                  Automatically send low-confidence inputs to the server queue <span className="st-muted">(this uploads those inputs)</span>
                </label>
              </div>
            )}

            {history.length > 1 && (
              <details className="el-history">
                <summary>Recent local predictions ({history.length})</summary>
                <ul>{history.map((h, i) => <li key={i}>{h.label ?? h.value?.toFixed(3)} {h.confidence !== undefined && `· ${pct(h.confidence)}`} · {h.inferenceMs.toFixed(2)} ms</li>)}</ul>
              </details>
            )}
          </>
        )}
      </section>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Edge-case queue tab
// ---------------------------------------------------------------------------
function FeatureChips({ features }) {
  return (
    <div className="el-features">
      {Object.entries(features || {}).map(([k, v]) => (
        <span key={k}><b>{k}</b> {v === null || v === "" ? "∅" : typeof v === "number" ? Number(v.toFixed?.(3) ?? v) : String(v)}</span>
      ))}
    </div>
  );
}

function QueueTab({ runId, threshold, setThreshold, onNewRun }) {
  const [summary, setSummary] = useState(null);
  const [cases, setCases] = useState([]);
  const [scanInfo, setScanInfo] = useState(null);
  const [busy, setBusy] = useState(null);
  const [error, setError] = useState(null);
  const [drafts, setDrafts] = useState({});
  const [retrain, setRetrain] = useState(null);

  const refresh = useCallback(async () => {
    if (!runId) return;
    try {
      const [s, c] = await Promise.all([api.getActiveLearning(runId), api.listEdgeCases(runId)]);
      setSummary(s); setCases(c);
    } catch (e) {
      setError(e.message);
    }
  }, [runId]);

  // Remounted per run (key={runId}); just load.
  useEffect(() => {
    if (!runId) return;
    let cancelled = false;
    Promise.all([api.getActiveLearning(runId), api.listEdgeCases(runId)])
      .then(([s, c]) => { if (!cancelled) { setSummary(s); setCases(c); } })
      .catch((e) => { if (!cancelled) setError(e.message); });
    return () => { cancelled = true; };
  }, [runId]);

  const act = async (key, fn) => {
    setBusy(key); setError(null);
    try { await fn(); await refresh(); } catch (e) { setError(e.message); } finally { setBusy(null); }
  };

  if (!runId) return <p className="st-muted">Choose a trained run above.</p>;
  const pending = cases.filter((c) => c.status === "pending");
  const labeled = cases.filter((c) => c.status === "labeled");
  const applied = cases.filter((c) => c.status === "applied");
  const classes = summary?.classes || [];
  const taskMetrics = METRICS[summary?.run?.task_type] || METRICS.classification;

  return (
    <div className="el-queue">
      <section className="studio-card el-scan">
        <div>
          <h2 className="card-heading"><ScanSearch size={17} /> Find uncertain examples</h2>
          <p className="st-muted">
            Low-confidence predictions arrive here from the browser, or you can scan the training data. Rows the model is unsure about are often
            ambiguous, borderline or mislabeled — exactly where a human label helps most.
          </p>
        </div>
        <div className="el-scan-controls">
          <label className="st-field">
            <span className="st-field-label">Confidence below {pct(threshold)}</span>
            <input type="range" min="0.5" max="0.95" step="0.05" value={threshold} onChange={(e) => setThreshold(Number(e.target.value))} />
          </label>
          <button className="btn btn-secondary" disabled={!summary?.can_scan || busy} onClick={() => act("scan", async () => setScanInfo(await api.scanRun(runId, { threshold, limit: 25 })))}>
            {busy === "scan" ? <Loader2 size={15} className="animate-spin" /> : <ScanSearch size={15} />} Scan dataset
          </button>
        </div>
        {summary && !summary.can_scan && <p className="st-muted">Scanning needs a classifier with probabilities.</p>}
        {scanInfo && (
          <p className="el-scan-result">
            Scanned {scanInfo.scanned_rows.toLocaleString()} rows: {scanInfo.below_threshold} below the threshold, {scanInfo.queued} new added to the queue (most uncertain first, max 25 per scan).
          </p>
        )}
      </section>

      {error && <div className="st-banner error"><AlertCircle size={15} /> {error}</div>}

      <section className="studio-card">
        <div className="el-queue-head">
          <h2 className="card-heading"><Inbox size={17} /> Edge-case queue <span className="status-tag">{pending.length} pending</span></h2>
          <button className="btn btn-primary" disabled={!labeled.length || busy} onClick={() => act("retrain", async () => { const r = await api.retrainFromLabels(runId); setRetrain(r); })}>
            {busy === "retrain" ? <Loader2 size={15} className="animate-spin" /> : <RefreshCw size={15} />} Retrain with {labeled.length} label{labeled.length === 1 ? "" : "s"}
          </button>
        </div>

        {retrain && (
          <div className="el-retrain">
            <strong>New run #{retrain.new_run.run_id}</strong> trained on <code>{retrain.new_dataset.filename}</code> ({retrain.new_dataset.rows.toLocaleString()} rows):
            {" "}{retrain.applied} label(s) applied — {retrain.relabeled_rows} corrected an existing row,{" "}
            {retrain.applied - retrain.relabeled_rows - retrain.appended_rows} confirmed the existing label, {retrain.appended_rows} added a new row.
            <table className="dd-table">
              <thead><tr><th>Metric</th><th>Run #{retrain.previous_run.run_id}</th><th>Run #{retrain.new_run.run_id}</th></tr></thead>
              <tbody>
                {taskMetrics.map(([k, label]) => (
                  <tr key={k}><td>{label}</td><td>{retrain.previous_run.metrics[k]?.toFixed(4) ?? "—"}</td><td>{retrain.new_run.metrics[k]?.toFixed(4) ?? "—"}</td></tr>
                ))}
              </tbody>
            </table>
            <p className="st-muted">The data changed, so the test set changed too — treat this as a rough comparison, not a controlled experiment.</p>
            <button className="btn btn-secondary btn-sm" onClick={() => onNewRun(retrain.new_run.run_id)}>Switch to run #{retrain.new_run.run_id}</button>
          </div>
        )}

        {!pending.length && <p className="st-muted">Nothing waiting for review.</p>}
        <div className="el-cases">
          {pending.map((c) => (
            <div key={c.id} className="el-case">
              <div className="el-case-top">
                <span className="el-case-id">#{c.id}</span>
                <span className="el-case-src">{c.source === "scan" ? `dataset row ${c.row_index}` : c.source}</span>
                <span>Predicted <strong>{c.predicted_label ?? "—"}</strong></span>
                {c.confidence !== null && <span className="el-conf">{pct(c.confidence)}</span>}
                {c.dataset_label !== null && <span className="st-muted">label in data: <code>{c.dataset_label}</code></span>}
              </div>
              {c.probabilities && (
                <div className="el-case-probs">
                  {Object.entries(c.probabilities).map(([k, p]) => <span key={k}>{k}: {pct(p)}</span>)}
                </div>
              )}
              <FeatureChips features={c.features} />
              <div className="el-case-actions">
                <select className="form-select-sm" value={drafts[c.id] ?? c.dataset_label ?? c.predicted_label ?? ""} onChange={(e) => setDrafts({ ...drafts, [c.id]: e.target.value })} aria-label="Correct label">
                  {classes.map((cls) => <option key={cls} value={cls}>{cls}</option>)}
                </select>
                <button className="btn btn-primary btn-sm" disabled={!!busy} onClick={() => act(`l${c.id}`, () => api.labelEdgeCase(c.id, drafts[c.id] ?? c.dataset_label ?? c.predicted_label))}>
                  <Check size={13} /> Save label
                </button>
                <button className="btn btn-secondary btn-sm" disabled={!!busy} onClick={() => act(`d${c.id}`, () => api.dismissEdgeCase(c.id))}>
                  <X size={13} /> Dismiss
                </button>
              </div>
            </div>
          ))}
        </div>

        {labeled.length > 0 && (
          <details className="el-labeled" open>
            <summary>Labeled, ready for retraining ({labeled.length})</summary>
            <ul>{labeled.map((c) => <li key={c.id}>#{c.id}: predicted {c.predicted_label} → <strong>{c.corrected_label}</strong></li>)}</ul>
          </details>
        )}
        {applied.length > 0 && <p className="st-muted">{applied.length} example(s) already folded into retrained runs.</p>}
      </section>
    </div>
  );
}

// ---------------------------------------------------------------------------
export default function EdgeLab() {
  const [params, setParams] = useSearchParams();
  const [runs, setRuns] = useState([]);
  const [loadError, setLoadError] = useState(null);
  const [threshold, setThreshold] = useState(0.6);
  const runId = params.get("runId") ? Number(params.get("runId")) : null;
  const tab = params.get("tab") === "queue" ? "queue" : "deploy";

  useEffect(() => {
    api.getExperiments()
      .then((list) => {
        setRuns(list);
        if (!params.get("runId") && list.length) setParams({ runId: String(list[0].run_id), tab }, { replace: true });
      })
      .catch((e) => setLoadError(e.message));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const selected = useMemo(() => runs.find((r) => r.run_id === runId), [runs, runId]);
  const select = (id, t = tab) => setParams({ runId: String(id), tab: t });

  return (
    <div className="page-container">
      <div className="page-header">
        <h1 className="page-title">Edge Lab</h1>
        <p className="page-subtitle">Shrink a trained model, run it inside the browser, and turn its uncertain predictions into new training data.</p>
      </div>

      <div className="studio-card el-runbar">
        <label className="st-field">
          <span className="st-field-label">Trained run</span>
          <select className="form-select" value={runId ?? ""} onChange={(e) => select(Number(e.target.value))}>
            {runs.map((r) => (
              <option key={r.run_id} value={r.run_id}>
                #{r.run_id} · {r.model_name} · {r.dataset_name}
              </option>
            ))}
          </select>
        </label>
        {selected && (
          <div className="el-run-metrics">
            {Object.entries(selected.metrics || {}).filter(([k]) => ["accuracy", "f1_macro", "r2", "rmse"].includes(k)).map(([k, v]) => (
              <span key={k}>{k}: <strong>{typeof v === "number" ? v.toFixed(3) : "—"}</strong></span>
            ))}
            <Link to={`/studio?runId=${selected.run_id}`} className="st-link-btn"><Workflow size={13} /> Open graph</Link>
          </div>
        )}
      </div>
      {loadError && <div className="st-banner error">Can't load runs: {loadError}</div>}
      {!runs.length && !loadError && <p className="st-muted">No trained runs yet — train one in the <Link to="/studio">Visual Studio</Link>.</p>}

      <div className="el-tabs" role="tablist">
        <button role="tab" aria-selected={tab === "deploy"} className={`tab-btn ${tab === "deploy" ? "active" : ""}`} onClick={() => select(runId, "deploy")}>
          <MonitorSmartphone size={15} /> Browser deployment
        </button>
        <button role="tab" aria-selected={tab === "queue"} className={`tab-btn ${tab === "queue" ? "active" : ""}`} onClick={() => select(runId, "queue")}>
          <Inbox size={15} /> Edge-case queue &amp; retraining
        </button>
      </div>

      {tab === "deploy"
        ? <DeployTab key={runId} runId={runId} threshold={threshold} setThreshold={setThreshold} />
        : <QueueTab key={runId} runId={runId} threshold={threshold} setThreshold={setThreshold} onNewRun={(id) => {
            api.getExperiments().then(setRuns).catch(() => {});
            select(id, "queue");
          }} />}
    </div>
  );
}

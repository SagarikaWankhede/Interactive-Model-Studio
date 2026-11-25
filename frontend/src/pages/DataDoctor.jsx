import { useEffect, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { CheckCircle2, AlertTriangle, AlertOctagon, ChevronDown, ChevronRight, Loader2, Workflow, Stethoscope } from "lucide-react";
import api from "../api/client";

const STATUS = {
  ok: { icon: CheckCircle2, label: "OK", cls: "ok" },
  warn: { icon: AlertTriangle, label: "Check", cls: "warn" },
  critical: { icon: AlertOctagon, label: "Problem", cls: "critical" },
};

const fmt = (v) => (typeof v === "number" ? v.toLocaleString(undefined, { maximumFractionDigits: 2 }) : String(v ?? "—"));

function Bar({ pct, tone = "neutral" }) {
  return (
    <span className={`dd-bar ${tone}`}>
      <span style={{ width: `${Math.min(100, Math.max(0, pct))}%` }} />
    </span>
  );
}

function CheckDetails({ check }) {
  const d = check.details;
  if (check.id === "missing" && d.length) {
    return (
      <table className="dd-table">
        <thead><tr><th>Column</th><th>Missing</th><th className="dd-bar-col" /></tr></thead>
        <tbody>
          {d.map((r) => (
            <tr key={r.column}><td><code>{r.column}</code></td><td>{r.count.toLocaleString()} ({r.pct}%)</td><td><Bar pct={r.pct} tone={r.pct >= 30 ? "critical" : "warn"} /></td></tr>
          ))}
        </tbody>
      </table>
    );
  }
  if (check.id === "balance" && d?.length) {
    return (
      <table className="dd-table">
        <thead><tr><th>Class</th><th>Rows</th><th className="dd-bar-col" /></tr></thead>
        <tbody>
          {d.map((r) => <tr key={r.label}><td><code>{r.label}</code></td><td>{r.count.toLocaleString()} ({r.pct}%)</td><td><Bar pct={r.pct} /></td></tr>)}
        </tbody>
      </table>
    );
  }
  if (check.id === "outliers" && d.length) {
    return (
      <table className="dd-table">
        <thead><tr><th>Feature</th><th>Flagged</th><th>Median</th><th>Normal range</th><th>Most extreme values</th></tr></thead>
        <tbody>
          {d.map((r) => (
            <tr key={r.column}>
              <td><code>{r.column}</code></td>
              <td>{r.count} ({r.pct}%)</td>
              <td>{fmt(r.median)}</td>
              <td>{fmt(r.lower_fence)} … {fmt(r.upper_fence)}</td>
              <td className="dd-mono">{r.examples.map(fmt).join(", ")}</td>
            </tr>
          ))}
        </tbody>
      </table>
    );
  }
  if (check.id === "correlation" && (d.pairs.length || d.leakage.length)) {
    return (
      <table className="dd-table">
        <thead><tr><th>Features</th><th>|correlation|</th><th>Meaning</th></tr></thead>
        <tbody>
          {d.leakage.map((r) => <tr key={`l-${r.column}`} className="dd-row-critical"><td><code>{r.column}</code> ↔ target</td><td>{r.corr}</td><td>possible leakage</td></tr>)}
          {d.pairs.map((r) => <tr key={`${r.a}-${r.b}`}><td><code>{r.a}</code> ↔ <code>{r.b}</code></td><td>{r.corr}</td><td>redundant information</td></tr>)}
        </tbody>
      </table>
    );
  }
  if (check.id === "types" && d.length) {
    return (
      <table className="dd-table">
        <thead><tr><th>Column</th><th>Issue</th><th>Detail</th></tr></thead>
        <tbody>{d.map((r) => <tr key={`${r.column}-${r.issue}`}><td><code>{r.column}</code></td><td>{r.issue}</td><td>{r.detail}</td></tr>)}</tbody>
      </table>
    );
  }
  return null;
}

function CheckRow({ check }) {
  const [open, setOpen] = useState(check.status !== "ok");
  const s = STATUS[check.status];
  const Icon = s.icon;
  return (
    <div className={`dd-check ${s.cls}`}>
      <button className="dd-check-head" onClick={() => setOpen(!open)} aria-expanded={open}>
        {open ? <ChevronDown size={15} /> : <ChevronRight size={15} />}
        <span className="dd-check-label">{check.label}</span>
        <span className="dd-check-value">{check.value}</span>
        <span className={`dd-status ${s.cls}`}><Icon size={14} /> {s.label}</span>
      </button>
      {open && (
        <div className="dd-check-body">
          <p className="dd-summary">{check.summary}</p>
          <p className="dd-why"><strong>Why it matters:</strong> {check.why}</p>
          {check.suggestion && <p className="dd-suggest"><strong>Try:</strong> {check.suggestion}</p>}
          <CheckDetails check={check} />
        </div>
      )}
    </div>
  );
}

export default function DataDoctor() {
  const [params, setParams] = useSearchParams();
  const [datasets, setDatasets] = useState([]);
  const [fetched, setFetched] = useState({ key: null, report: null, error: null });
  const [loadError, setLoadError] = useState(null);
  const datasetId = params.get("datasetId") || "";
  const target = params.get("target") || "";
  const key = `${datasetId}|${target}`;
  // Derived: a request is in flight whenever the latest result is for a different selection.
  const loading = Boolean(datasetId) && fetched.key !== key;
  const report = loading ? null : fetched.report;
  const error = loadError || (loading ? null : fetched.error);

  useEffect(() => {
    api.getDatasets()
      .then((list) => {
        setDatasets(list);
        if (!datasetId && list.length) setParams({ datasetId: String(list[0].id) }, { replace: true });
      })
      .catch((e) => setLoadError(`Can't reach the backend: ${e.message}`));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!datasetId) return;
    let cancelled = false;
    api.getDatasetReport(datasetId, target)
      .then((r) => { if (!cancelled) setFetched({ key, report: r, error: null }); })
      .catch((e) => { if (!cancelled) setFetched({ key, report: null, error: e.message }); });
    return () => { cancelled = true; };
  }, [datasetId, target, key]);

  const counts = report ? report.checks.reduce((acc, c) => ({ ...acc, [c.status]: (acc[c.status] || 0) + 1 }), {}) : {};
  const scoreTone = report ? (report.health_score >= 85 ? "ok" : report.health_score >= 60 ? "warn" : "critical") : "ok";

  return (
    <div className="page-container">
      <div className="page-header">
        <h1 className="page-title"><Stethoscope size={24} className="dd-title-icon" /> Data Doctor</h1>
        <p className="page-subtitle">
          Many ML problems start in the data, not the model. This report flags issues to investigate — it never changes your data.
        </p>
      </div>

      <div className="dd-controls studio-card">
        <label className="st-field">
          <span className="st-field-label">Dataset</span>
          <select className="form-select" value={datasetId} onChange={(e) => setParams({ datasetId: e.target.value })}>
            {datasets.map((d) => <option key={d.id} value={d.id}>{d.filename || d.name}</option>)}
          </select>
        </label>
        <label className="st-field">
          <span className="st-field-label">Target column</span>
          <select className="form-select" value={report?.target_column || target} onChange={(e) => setParams({ datasetId, target: e.target.value })}>
            {(report?.all_columns || []).map((c) => <option key={c} value={c}>{c}</option>)}
          </select>
        </label>
        {datasetId && (
          <Link className="btn btn-secondary btn-sm dd-open-studio" to="/studio">
            <Workflow size={14} /> Open Visual Studio
          </Link>
        )}
      </div>

      {error && <div className="st-banner error">{error}</div>}
      {!datasets.length && !error && <p className="st-muted">No datasets yet — <Link to="/upload">upload a CSV</Link>.</p>}
      {loading && <p className="st-muted"><Loader2 size={14} className="animate-spin" /> Examining the data…</p>}

      {report && !loading && (
        <div className="dd-report">
          <div className="dd-report-head">
            <div>
              <div className="dd-eyebrow">Dataset report</div>
              <div className="dd-filename">{report.filename}</div>
              <div className="dd-meta">
                {report.rows.toLocaleString()} rows · {report.columns} columns · target <code>{report.target_column}</code> ({report.task_type})
              </div>
            </div>
            <div className={`dd-score ${scoreTone}`} title="100 minus 25 per problem and 8 per warning">
              <strong>{report.health_score}</strong>
              <span>health score</span>
            </div>
          </div>
          <div className="dd-counts">
            <span className="dd-status critical"><AlertOctagon size={13} /> {counts.critical || 0} problems</span>
            <span className="dd-status warn"><AlertTriangle size={13} /> {counts.warn || 0} to check</span>
            <span className="dd-status ok"><CheckCircle2 size={13} /> {counts.ok || 0} OK</span>
          </div>
          <div className="dd-checks">
            {report.checks.map((c) => <CheckRow key={`${report.dataset_id}-${report.target_column}-${c.id}`} check={c} />)}
          </div>

          <details className="dd-columns">
            <summary>Column overview ({report.column_summary.length})</summary>
            <table className="dd-table">
              <thead><tr><th>Column</th><th>Type</th><th>Unique</th><th>Missing</th></tr></thead>
              <tbody>
                {report.column_summary.map((c) => (
                  <tr key={c.name}><td><code>{c.name}</code></td><td>{c.type}</td><td>{c.unique.toLocaleString()}</td><td>{c.missing_pct}%</td></tr>
                ))}
              </tbody>
            </table>
          </details>
        </div>
      )}
    </div>
  );
}

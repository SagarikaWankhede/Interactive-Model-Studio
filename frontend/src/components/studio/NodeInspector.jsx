import { Link } from "react-router-dom";
import { Trash2, Stethoscope, AlertCircle, AlertTriangle } from "lucide-react";
import { ALGORITHMS, NODE_TYPES, algorithmById, defaultHyperparams, sklearnName } from "./nodeCatalog";

function ColumnPicker({ schema, target, value = [], onChange, filter }) {
  const cols = schema.filter((c) => c.name !== target && (!filter || filter(c)));
  const toggle = (name) => onChange(value.includes(name) ? value.filter((c) => c !== name) : [...value, name]);
  const pick = (type) => onChange([...new Set([...value, ...cols.filter((c) => c.type === type).map((c) => c.name)])]);

  if (!schema.length) return <p className="st-muted">Pick a dataset in the CSV Dataset node first.</p>;
  return (
    <div>
      <div className="st-chip-actions">
        <button type="button" onClick={() => pick("numerical")}>+ numeric</button>
        <button type="button" onClick={() => pick("categorical")}>+ categorical</button>
        <button type="button" onClick={() => onChange([])}>clear</button>
      </div>
      <div className="st-chips">
        {cols.map((c) => (
          <button
            type="button"
            key={c.name}
            className={`st-chip ${value.includes(c.name) ? "on" : ""}`}
            onClick={() => toggle(c.name)}
            aria-pressed={value.includes(c.name)}
            title={`${c.type}${c.null_count ? ` · ${c.null_count} missing` : ""}`}
          >
            {c.name}
            <span className={`st-chip-type ${c.type}`}>{c.type === "numerical" ? "123" : "abc"}</span>
            {c.null_count > 0 && <span className="st-chip-missing">{c.null_pct}%∅</span>}
          </button>
        ))}
      </div>
      {value.filter((v) => !cols.some((c) => c.name === v)).length > 0 && (
        <p className="st-muted">Also selected (not in this dataset): {value.filter((v) => !cols.some((c) => c.name === v)).join(", ")}</p>
      )}
    </div>
  );
}

function Field({ label, children, hint }) {
  return (
    <label className="st-field">
      <span className="st-field-label">{label}</span>
      {children}
      {hint && <span className="st-field-hint">{hint}</span>}
    </label>
  );
}

export default function NodeInspector({ node, datasets, schema, issues, onChange, onDelete }) {
  if (!node) {
    return (
      <div className="st-inspector st-inspector-empty">
        <p><strong>Select a node</strong> to configure it.</p>
        <p className="st-muted">Drag nodes from the palette (or click them to append), then connect the bottom dot of one node to the top dot of the next.</p>
      </div>
    );
  }

  const { kind, params } = node.data;
  const meta = NODE_TYPES[kind];
  const set = (patch) => onChange(node.id, { ...params, ...patch });
  const target = kind === "dataset" ? params.target_column : null;

  return (
    <div className="st-inspector">
      <div className="st-inspector-head" style={{ "--node-color": meta.color }}>
        <div>
          <div className="st-inspector-title">{meta.label}</div>
          <code className="st-node-sklearn">{sklearnName(kind, params)}</code>
        </div>
        <button className="st-icon-btn danger" onClick={() => onDelete(node.id)} title="Delete node" aria-label="Delete node">
          <Trash2 size={15} />
        </button>
      </div>
      <p className="st-muted">{meta.hint}</p>

      {issues?.errors?.map((m) => (
        <div key={m} className="st-issue error"><AlertCircle size={14} /> {m}</div>
      ))}
      {issues?.warnings?.map((m) => (
        <div key={m} className="st-issue warning"><AlertTriangle size={14} /> {m}</div>
      ))}

      {kind === "dataset" && (
        <>
          <Field label="Dataset">
            <select
              className="form-select"
              value={params.dataset_id ?? ""}
              onChange={(e) => {
                const ds = datasets.find((d) => String(d.id) === e.target.value);
                set({ dataset_id: ds ? ds.id : "", filename: ds ? ds.filename || ds.name : "", target_column: ds?.target_column || "" });
              }}
            >
              <option value="">— choose —</option>
              {datasets.map((d) => (
                <option key={d.id} value={d.id}>{d.filename || d.name} ({d.row_count} rows)</option>
              ))}
            </select>
          </Field>
          {datasets.length === 0 && (
            <p className="st-muted">No datasets on the server yet. <Link to="/upload">Upload a CSV</Link> first.</p>
          )}
          <Field label="Target column (what to predict)">
            <select className="form-select" value={params.target_column || ""} onChange={(e) => set({ target_column: e.target.value })}>
              <option value="">— choose —</option>
              {schema.map((c) => <option key={c.name} value={c.name}>{c.name}</option>)}
            </select>
          </Field>
          <Field label="Task" hint="Auto: text or ≤10 distinct values → classification, otherwise regression.">
            <select className="form-select" value={params.task_type || "auto"} onChange={(e) => set({ task_type: e.target.value })}>
              <option value="auto">Auto-detect</option>
              <option value="classification">Classification</option>
              <option value="regression">Regression</option>
            </select>
          </Field>
          {params.dataset_id && (
            <Link className="st-link-btn" to={`/doctor?datasetId=${params.dataset_id}${params.target_column ? `&target=${encodeURIComponent(params.target_column)}` : ""}`}>
              <Stethoscope size={14} /> Check this data with the Data Doctor
            </Link>
          )}
        </>
      )}

      {kind === "impute" && (
        <>
          <Field label="Strategy" hint="mean/median: numeric only · most_frequent: any column">
            <select className="form-select" value={params.strategy} onChange={(e) => set({ strategy: e.target.value })}>
              <option value="mean">mean</option>
              <option value="median">median</option>
              <option value="most_frequent">most_frequent</option>
              <option value="constant">constant</option>
            </select>
          </Field>
          {params.strategy === "constant" && (
            <Field label="Fill value">
              <input className="form-input" value={params.fill_value ?? 0} onChange={(e) => set({ fill_value: e.target.value === "" || Number.isNaN(Number(e.target.value)) ? e.target.value : Number(e.target.value) })} />
            </Field>
          )}
          <Field label="Columns"><ColumnPicker schema={schema} target={target} value={params.columns} onChange={(columns) => set({ columns })} filter={params.strategy === "mean" || params.strategy === "median" ? (c) => c.type === "numerical" : null} /></Field>
        </>
      )}

      {kind === "scale" && (
        <>
          <Field label="Method">
            <select className="form-select" value={params.method} onChange={(e) => set({ method: e.target.value })}>
              <option value="standard">standard — mean 0, std 1</option>
              <option value="minmax">minmax — squeeze into 0…1</option>
              <option value="robust">robust — median/IQR, outlier-resistant</option>
            </select>
          </Field>
          <Field label="Columns"><ColumnPicker schema={schema} value={params.columns} onChange={(columns) => set({ columns })} filter={(c) => c.type === "numerical"} /></Field>
        </>
      )}

      {kind === "encode" && (
        <>
          <Field label="Method">
            <select className="form-select" value={params.method} onChange={(e) => set({ method: e.target.value })}>
              <option value="onehot">onehot — one 0/1 column per category</option>
              <option value="ordinal">ordinal — one integer per category</option>
            </select>
          </Field>
          <Field label="Columns"><ColumnPicker schema={schema} value={params.columns} onChange={(columns) => set({ columns })} /></Field>
        </>
      )}

      {kind === "drop_columns" && (
        <Field label="Columns to remove"><ColumnPicker schema={schema} value={params.columns} onChange={(columns) => set({ columns })} /></Field>
      )}

      {kind === "split" && (
        <>
          <Field label={`Test size: ${Math.round((params.test_size ?? 0.2) * 100)}%`}>
            <input type="range" min="0.1" max="0.4" step="0.05" value={params.test_size ?? 0.2} onChange={(e) => set({ test_size: Number(e.target.value) })} />
          </Field>
          <Field label="Random seed" hint="Same seed → same split → reproducible results.">
            <input type="number" className="form-input" value={params.random_state ?? 42} onChange={(e) => set({ random_state: Number(e.target.value) || 0 })} />
          </Field>
          <label className="st-check">
            <input type="checkbox" checked={!!params.stratify} onChange={(e) => set({ stratify: e.target.checked })} />
            Stratify (keep class proportions equal in train and test)
          </label>
          <p className="st-callout">
            <strong>Why split first?</strong> The test rows must stay unseen. So the data is split before anything else, and every
            preprocessing step learns from the training rows only (e.g. the imputer's median), then reuses that on the test rows.
            Keep this node directly under the dataset so the graph reads in the same order as the code.
          </p>
        </>
      )}

      {kind === "model" && (
        <>
          <Field label="Algorithm">
            <select className="form-select" value={params.algorithm} onChange={(e) => set({ algorithm: e.target.value, hyperparameters: defaultHyperparams(e.target.value) })}>
              {["classification", "regression"].map((task) => (
                <optgroup key={task} label={task === "classification" ? "Classification" : "Regression"}>
                  {ALGORITHMS.filter((a) => a.task === task).map((a) => <option key={a.id} value={a.id}>{a.label}</option>)}
                </optgroup>
              ))}
            </select>
          </Field>
          {algorithmById(params.algorithm)?.note && <p className="st-muted">{algorithmById(params.algorithm).note}</p>}
          {(algorithmById(params.algorithm)?.params || []).map((p) => (
            <Field key={p.name} label={p.name}>
              <input
                type="number"
                className="form-input"
                min={p.min}
                step={p.step || 1}
                placeholder={p.placeholder}
                value={params.hyperparameters?.[p.name] ?? ""}
                onChange={(e) => set({ hyperparameters: { ...params.hyperparameters, [p.name]: e.target.value === "" ? "" : Number(e.target.value) } })}
              />
            </Field>
          ))}
        </>
      )}

      {kind === "evaluate" && (
        <p className="st-muted">Classification: accuracy, precision, recall, F1 (macro). Regression: RMSE, MAE, R². Computed on the test rows only.</p>
      )}
    </div>
  );
}

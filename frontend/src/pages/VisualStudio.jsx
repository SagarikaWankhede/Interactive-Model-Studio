import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import ReactFlow, { Background, Controls, MiniMap, ReactFlowProvider, addEdge, useEdgesState, useNodesState } from "reactflow";
import "reactflow/dist/style.css";
import { Play, Sparkles, Eraser, LayoutList, Loader2, AlertCircle, AlertTriangle, Rocket, Stethoscope, X, WifiOff } from "lucide-react";
import api from "../api/client";
import MLNode from "../components/studio/MLNode";
import { NODE_ICONS, NodeIssuesContext } from "../components/studio/studioShared";
import NodeInspector from "../components/studio/NodeInspector";
import CodePanel from "../components/studio/CodePanel";
import { NODE_TYPES, PALETTE_GROUPS, makeEdge, makeNode, starterPipeline } from "../components/studio/nodeCatalog";

const nodeTypes = { mlNode: MLNode };
const STORAGE_KEY = "ims.visualStudio.graph.v1";
const ROW = 105;
const EMPTY_COMPILE = { code: "", blocks: [], errors: [], warnings: [], order: [], ok: false };
const INSERT_BEFORE_MODEL = new Set(["impute", "scale", "encode", "drop_columns", "split"]);

const METRIC_LABELS = {
  accuracy: "Accuracy", precision_macro: "Precision", recall_macro: "Recall", f1_macro: "F1",
  rmse: "RMSE", mae: "MAE", r2: "R²",
};

function loadSaved() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY));
    if (saved?.nodes?.length) return saved;
  } catch {
    /* ignore corrupt storage */
  }
  return { nodes: [], edges: [] };
}

const toPayload = (nodes, edges) => ({
  nodes: nodes.map((n) => ({ id: n.id, data: { kind: n.data.kind, params: n.data.params } })),
  edges: edges.map((e) => ({ id: e.id, source: e.source, target: e.target })),
});

function formatMetric(key, v) {
  if (v === null || v === undefined) return "—";
  if (["accuracy", "precision_macro", "recall_macro", "f1_macro"].includes(key)) return `${(v * 100).toFixed(1)}%`;
  return Number(v).toFixed(3);
}

function Studio() {
  const [searchParams] = useSearchParams();
  const [initial] = useState(loadSaved);
  const [nodes, setNodes, onNodesChange] = useNodesState(initial.nodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initial.edges);
  const [selectedId, setSelectedId] = useState(null);
  const [rf, setRf] = useState(null);

  const [datasets, setDatasets] = useState([]);
  const [backendDown, setBackendDown] = useState(false);
  const [compiledState, setCompiled] = useState(EMPTY_COMPILE);
  const [compiling, setCompiling] = useState(false);
  const [training, setTraining] = useState(false);
  const [result, setResult] = useState(null);
  const [trainError, setTrainError] = useState(null);
  const compileSeq = useRef(0);

  // ---- data loading ---------------------------------------------------------
  useEffect(() => {
    api.getDatasets()
      .then((list) => { setDatasets(list); setBackendDown(false); })
      .catch(() => setBackendDown(true));
  }, []);

  useEffect(() => {
    const runId = searchParams.get("runId");
    if (!runId) return;
    api.getRunGraph(runId)
      .then(({ graph }) => {
        setNodes(graph.nodes.map((n) => ({ ...n, type: "mlNode", position: n.position || { x: 40, y: 20 } })));
        setEdges(graph.edges.map((e) => makeEdge(e.source, e.target)));
      })
      .catch(() => { /* run not from the visual editor — keep current canvas */ });
  }, [searchParams, setNodes, setEdges]);

  // ---- live compile (debounced) --------------------------------------------
  const signature = useMemo(
    () => JSON.stringify(toPayload(nodes, edges)),
    [nodes, edges],
  );
  // Positions change while dragging but don't affect code — only recompile on structure/params.
  const structural = useMemo(() => {
    const p = JSON.parse(signature);
    return JSON.stringify({ n: p.nodes, e: p.edges.map((e) => [e.source, e.target]) });
  }, [signature]);

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ nodes, edges }));
  }, [nodes, edges]);

  useEffect(() => {
    const payload = JSON.parse(structural);
    const seq = ++compileSeq.current;
    if (!payload.n.length) return;
    const t = setTimeout(() => {
      setCompiling(true);
      api.compileGraph(JSON.parse(signature))
        .then((res) => { if (seq === compileSeq.current) { setCompiled(res); setBackendDown(false); } })
        .catch(() => { if (seq === compileSeq.current) setBackendDown(true); })
        .finally(() => { if (seq === compileSeq.current) setCompiling(false); });
    }, 300);
    return () => clearTimeout(t);
    // `signature` is read inside but `structural` is the intended trigger.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [structural]);

  // An empty canvas has nothing to compile; ignore any stale result.
  const compiled = nodes.length ? compiledState : EMPTY_COMPILE;

  const issues = useMemo(() => {
    const map = {};
    const add = (list, key) => list.forEach((i) => {
      if (!i.node_id) return;
      map[i.node_id] ??= { errors: [], warnings: [] };
      map[i.node_id][key].push(i.message);
    });
    add(compiled.errors || [], "errors");
    add(compiled.warnings || [], "warnings");
    return map;
  }, [compiled]);
  const globalErrors = (compiled.errors || []).filter((e) => !e.node_id);

  // ---- graph helpers --------------------------------------------------------
  const datasetNode = nodes.find((n) => n.data.kind === "dataset");
  const activeDataset = datasets.find((d) => String(d.id) === String(datasetNode?.data.params.dataset_id));
  const schema = activeDataset?.schema || [];
  const selectedNode = nodes.find((n) => n.id === selectedId) || null;
  // Selection is driven by `selectedId` so the canvas and the code panel stay in sync.
  const displayNodes = useMemo(() => nodes.map((n) => (n.selected === (n.id === selectedId) ? n : { ...n, selected: n.id === selectedId })), [nodes, selectedId]);

  const onConnect = useCallback((conn) => {
    if (conn.source === conn.target) return;
    // Keep the pipeline a single chain: a new link replaces the old output/input.
    setEdges((eds) => addEdge(makeEdge(conn.source, conn.target), eds.filter((e) => e.source !== conn.source && e.target !== conn.target)));
  }, [setEdges]);

  const updateParams = useCallback((id, params) => {
    setNodes((nds) => nds.map((n) => (n.id === id ? { ...n, data: { ...n.data, params } } : n)));
  }, [setNodes]);

  const deleteNode = useCallback((id) => {
    const incoming = edges.find((e) => e.target === id);
    const outgoing = edges.find((e) => e.source === id);
    setNodes((nds) => nds.filter((n) => n.id !== id));
    setEdges((eds) => {
      const rest = eds.filter((e) => e.source !== id && e.target !== id);
      // Re-link the neighbours so deleting a middle step keeps the chain intact.
      return incoming && outgoing ? [...rest, makeEdge(incoming.source, outgoing.target)] : rest;
    });
    setSelectedId(null);
  }, [edges, setNodes, setEdges]);

  const addNodeSmart = useCallback((kind) => {
    const chain = compiled.order?.length ? compiled.order : [];
    const byId = Object.fromEntries(nodes.map((n) => [n.id, n]));
    const chainNodes = chain.map((id) => byId[id]).filter(Boolean);
    const modelIdx = chainNodes.findIndex((n) => n.data.kind === "model");

    // The engine splits first, so a new Split goes right under the dataset; other steps go just above the model.
    const insertAt = kind === "split" && chainNodes[0]?.data.kind === "dataset" && chainNodes.length > 1 ? 1 : modelIdx;
    if (INSERT_BEFORE_MODEL.has(kind) && insertAt > 0) {
      const next = chainNodes[insertAt];
      const prev = chainNodes[insertAt - 1];
      const node = makeNode(kind, { ...next.position });
      const shift = new Set(chainNodes.slice(insertAt).map((n) => n.id));
      setNodes((nds) => [...nds.map((n) => (shift.has(n.id) ? { ...n, position: { ...n.position, y: n.position.y + ROW } } : n)), node]);
      setEdges((eds) => [...eds.filter((e) => !(e.source === prev.id && e.target === next.id)), makeEdge(prev.id, node.id), makeEdge(node.id, next.id)]);
      setSelectedId(node.id);
      return;
    }
    const tail = chainNodes[chainNodes.length - 1] || nodes[nodes.length - 1];
    const position = tail ? { x: tail.position.x, y: tail.position.y + ROW } : { x: 40, y: 20 };
    const node = makeNode(kind, position);
    setNodes((nds) => [...nds, node]);
    if (tail && tail.data.kind !== "evaluate" && kind !== "dataset") setEdges((eds) => [...eds, makeEdge(tail.id, node.id)]);
    setSelectedId(node.id);
  }, [compiled.order, nodes, setNodes, setEdges]);

  const onDrop = useCallback((event) => {
    event.preventDefault();
    const kind = event.dataTransfer.getData("application/x-ml-node");
    if (!kind || !rf) return;
    const node = makeNode(kind, rf.screenToFlowPosition({ x: event.clientX - 110, y: event.clientY - 30 }));
    setNodes((nds) => [...nds, node]);
    setSelectedId(node.id);
  }, [rf, setNodes]);

  const autoLayout = () => {
    const order = compiled.order || [];
    let other = 0;
    setNodes((nds) => nds.map((n) => {
      const i = order.indexOf(n.id);
      if (i >= 0) return { ...n, position: { x: 40, y: 20 + i * ROW } };
      other += 1;
      return { ...n, position: { x: 340, y: 20 + (other - 1) * ROW } };
    }));
    setTimeout(() => rf?.fitView({ padding: 0.15 }), 50);
  };

  const loadStarter = () => {
    const ds = activeDataset || datasets[0];
    if (!ds) return;
    const target = datasetNode?.data.params.target_column || ds.target_column;
    const { nodes: n, edges: e } = starterPipeline(ds, target);
    setNodes(n);
    setEdges(e);
    setSelectedId(n[0].id);
    setResult(null);
    setTimeout(() => rf?.fitView({ padding: 0.15 }), 50);
  };

  const clearAll = () => {
    setNodes([]);
    setEdges([]);
    setSelectedId(null);
    setResult(null);
  };

  const train = async () => {
    setTraining(true);
    setTrainError(null);
    try {
      const res = await api.runGraph(toPayload(nodes, edges));
      setResult(res.run);
    } catch (err) {
      setTrainError(err.detail?.errors ? err.detail.errors.map((e) => e.message).join(" ") : err.message);
    } finally {
      setTraining(false);
    }
  };

  const metricKeys = result
    ? (result.task_type === "regression" ? ["rmse", "mae", "r2"] : ["accuracy", "precision_macro", "recall_macro", "f1_macro"])
    : [];

  return (
    <div className="st-page">
      <div className="st-header">
        <div>
          <h1 className="page-title">Visual ML Studio</h1>
          <p className="page-subtitle">Build the pipeline as a graph. The Python on the right is exactly what runs when you press Train.</p>
        </div>
        <div className="st-header-actions">
          <button className="btn btn-secondary btn-sm" onClick={loadStarter} disabled={!datasets.length} title="Build a sensible pipeline from the dataset's columns">
            <Sparkles size={14} /> Starter pipeline
          </button>
          <button className="btn btn-secondary btn-sm" onClick={autoLayout} disabled={!nodes.length}><LayoutList size={14} /> Tidy</button>
          <button className="btn btn-secondary btn-sm" onClick={clearAll} disabled={!nodes.length}><Eraser size={14} /> Clear</button>
          <button className="btn btn-primary" onClick={train} disabled={!compiled.ok || training || compiling} title={compiled.ok ? "Train on the server" : "Fix the errors first"}>
            {training ? <Loader2 size={15} className="animate-spin" /> : <Play size={15} />} {training ? "Training…" : "Train"}
          </button>
        </div>
      </div>

      {backendDown && (
        <div className="st-banner error"><WifiOff size={15} /> Can't reach the backend at the API URL — start it with <code>uvicorn backend.main:app --reload</code>.</div>
      )}
      {globalErrors.map((e) => (
        <div key={e.message} className="st-banner warning"><AlertTriangle size={15} /> {e.message}</div>
      ))}
      {trainError && (
        <div className="st-banner error"><AlertCircle size={15} /> {trainError}<button className="st-banner-x" onClick={() => setTrainError(null)} aria-label="Dismiss"><X size={14} /></button></div>
      )}

      {result && (
        <div className="st-result">
          <div className="st-result-title">
            <strong>Run #{result.run_id}</strong> · {result.algorithm} · {result.train_time_sec?.toFixed(2)}s
          </div>
          <div className="st-result-metrics">
            {metricKeys.map((k) => (
              <div key={k} className="st-metric">
                <span>{METRIC_LABELS[k]}</span>
                <strong>{formatMetric(k, result.metrics?.[k])}</strong>
              </div>
            ))}
          </div>
          <div className="st-result-actions">
            <Link className="btn btn-secondary btn-sm" to={`/edge?runId=${result.run_id}`}><Rocket size={14} /> Optimize &amp; run in browser</Link>
            <Link className="btn btn-secondary btn-sm" to={`/edge?runId=${result.run_id}&tab=queue`}>Edge-case queue</Link>
            <button className="st-banner-x" onClick={() => setResult(null)} aria-label="Dismiss"><X size={14} /></button>
          </div>
          {result.task_type === "classification" && <p className="st-muted st-result-note">Tip: on imbalanced data, accuracy can look great while the model ignores the rare class — compare it with recall and F1.</p>}
        </div>
      )}

      <div className="st-workspace">
        <aside className="st-left">
          <div className="st-palette">
            {PALETTE_GROUPS.map((group) => (
              <div key={group} className="st-palette-group">
                <div className="st-palette-heading">{group}</div>
                {Object.entries(NODE_TYPES).filter(([, m]) => m.group === group).map(([kind, meta]) => {
                  const Icon = NODE_ICONS[kind];
                  return (
                    <button
                      key={kind}
                      className="st-palette-item"
                      style={{ "--node-color": meta.color }}
                      draggable
                      onDragStart={(e) => { e.dataTransfer.setData("application/x-ml-node", kind); e.dataTransfer.effectAllowed = "move"; }}
                      onClick={() => addNodeSmart(kind)}
                      title={`${meta.hint}\nDrag onto the canvas, or click to add it to the chain.`}
                    >
                      <Icon size={14} /> {meta.label}
                    </button>
                  );
                })}
              </div>
            ))}
          </div>
          <NodeInspector
            node={selectedNode}
            datasets={datasets}
            schema={schema}
            issues={selectedNode ? issues[selectedNode.id] : null}
            onChange={updateParams}
            onDelete={deleteNode}
          />
        </aside>

        <section className="st-canvas" onDragOver={(e) => { e.preventDefault(); e.dataTransfer.dropEffect = "move"; }} onDrop={onDrop}>
          <NodeIssuesContext.Provider value={issues}>
            <ReactFlow
              nodes={displayNodes}
              edges={edges}
              nodeTypes={nodeTypes}
              onNodesChange={onNodesChange}
              onEdgesChange={onEdgesChange}
              onConnect={onConnect}
              onInit={setRf}
              onNodeClick={(_, n) => setSelectedId(n.id)}
              onPaneClick={() => setSelectedId(null)}
              onNodesDelete={(deleted) => deleted.some((n) => n.id === selectedId) && setSelectedId(null)}
              defaultEdgeOptions={{ type: "smoothstep" }}
              fitView
              fitViewOptions={{ padding: 0.15, maxZoom: 1 }}
              proOptions={{ hideAttribution: true }}
            >
              <Background gap={18} size={1} />
              <Controls showInteractive={false} />
              <MiniMap pannable zoomable style={{ width: 110, height: 80 }} nodeColor={(n) => NODE_TYPES[n.data?.kind]?.color || "#94a3b8"} />
            </ReactFlow>
          </NodeIssuesContext.Provider>
          {!nodes.length && (
            <div className="st-canvas-empty">
              <p><strong>Empty canvas.</strong></p>
              <p>Click <em>Starter pipeline</em>, or drag a <em>CSV Dataset</em> node here and build from scratch.</p>
              {!datasets.length && !backendDown && <p><Link to="/upload">Upload a dataset</Link> first.</p>}
            </div>
          )}
          {datasetNode && activeDataset && (
            <Link className="st-canvas-doctor" to={`/doctor?datasetId=${activeDataset.id}&target=${encodeURIComponent(datasetNode.data.params.target_column || "")}`}>
              <Stethoscope size={13} /> Data Doctor
            </Link>
          )}
        </section>

        <section className="st-right">
          <CodePanel
            code={compiled.code}
            blocks={compiled.blocks}
            selectedId={selectedId}
            onSelectNode={setSelectedId}
            compiling={compiling}
          />
        </section>
      </div>
    </div>
  );
}

export default function VisualStudio() {
  return (
    <ReactFlowProvider>
      <Studio />
    </ReactFlowProvider>
  );
}

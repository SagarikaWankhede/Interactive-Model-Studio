import { memo, useContext } from "react";
import { Handle, Position } from "reactflow";
import { AlertCircle, AlertTriangle } from "lucide-react";
import { NODE_TYPES, describeNode, sklearnName } from "./nodeCatalog";
import { NODE_ICONS, NodeIssuesContext } from "./studioShared";

function MLNode({ id, data, selected }) {
  const issues = useContext(NodeIssuesContext)[id];
  const meta = NODE_TYPES[data.kind];
  const Icon = NODE_ICONS[data.kind];
  const hasError = issues?.errors?.length > 0;
  const hasWarning = !hasError && issues?.warnings?.length > 0;
  const tooltip = [...(issues?.errors || []), ...(issues?.warnings || [])].join("\n");

  return (
    <div
      className={`st-node ${selected ? "is-selected" : ""} ${hasError ? "has-error" : ""} ${hasWarning ? "has-warning" : ""}`}
      style={{ "--node-color": meta.color }}
      title={tooltip || meta.hint}
    >
      {data.kind !== "dataset" && <Handle type="target" position={Position.Top} className="st-handle" />}
      <div className="st-node-icon">
        <Icon size={16} />
      </div>
      <div className="st-node-body">
        <div className="st-node-title">
          {meta.label}
          {hasError && <AlertCircle size={13} className="st-node-flag error" aria-label="Has errors" />}
          {hasWarning && <AlertTriangle size={13} className="st-node-flag warning" aria-label="Has warnings" />}
        </div>
        <div className="st-node-detail">{describeNode(data.kind, data.params)}</div>
        <code className="st-node-sklearn">{sklearnName(data.kind, data.params)}</code>
      </div>
      {data.kind !== "evaluate" && <Handle type="source" position={Position.Bottom} className="st-handle" />}
    </div>
  );
}

export default memo(MLNode);

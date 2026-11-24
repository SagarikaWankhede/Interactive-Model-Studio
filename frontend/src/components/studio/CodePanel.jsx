import { useEffect, useMemo, useRef, useState } from "react";
import { Copy, Check, Download, Loader2 } from "lucide-react";

const PY_KEYWORDS = /\b(import|from|as|def|return|if|else|for|in|print|True|False|None)\b/g;

/** Tiny Python highlighter: comments, strings, keywords. Returns React nodes. */
function highlight(line) {
  const hash = line.indexOf("#");
  const quoteBefore = hash >= 0 ? (line.slice(0, hash).match(/"/g) || []).length % 2 === 1 : false;
  const codePart = hash >= 0 && !quoteBefore ? line.slice(0, hash) : line;
  const comment = hash >= 0 && !quoteBefore ? line.slice(hash) : "";

  const parts = [];
  const re = /("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')/g;
  let last = 0;
  let m;
  while ((m = re.exec(codePart))) {
    if (m.index > last) parts.push({ t: "code", v: codePart.slice(last, m.index) });
    parts.push({ t: "str", v: m[0] });
    last = m.index + m[0].length;
  }
  if (last < codePart.length) parts.push({ t: "code", v: codePart.slice(last) });

  const out = parts.map((p, i) => {
    if (p.t === "str") return <span key={i} className="py-str">{p.v}</span>;
    const segs = p.v.split(PY_KEYWORDS);
    return segs.map((s, j) => (j % 2 === 1 ? <span key={`${i}-${j}`} className="py-kw">{s}</span> : s));
  });
  if (comment) out.push(<span key="c" className="py-comment">{comment}</span>);
  return out;
}

export default function CodePanel({ code, blocks, selectedId, onSelectNode, compiling, filename = "train_model.py" }) {
  const [copied, setCopied] = useState(false);
  const scrollRef = useRef(null);
  const lines = useMemo(() => (code || "").split("\n"), [code]);

  const lineOwner = useMemo(() => {
    const owner = {};
    (blocks || []).forEach((b) => {
      for (let l = b.start; l <= b.end; l++) owner[l] = b.node_id;
    });
    return owner;
  }, [blocks]);

  const active = (blocks || []).find((b) => b.node_id === selectedId);

  useEffect(() => {
    if (!active || !scrollRef.current) return;
    const el = scrollRef.current.querySelector(`[data-line="${active.start}"]`);
    el?.scrollIntoView({ block: "center", behavior: "smooth" });
  }, [active]);

  const copy = () => {
    navigator.clipboard.writeText(code).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  };

  const download = () => {
    const url = URL.createObjectURL(new Blob([code], { type: "text/x-python" }));
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="st-code">
      <div className="st-code-bar">
        <span className="st-code-file">{filename}</span>
        {compiling && <Loader2 size={13} className="animate-spin" aria-label="Updating" />}
        <span className="st-code-spacer" />
        <button className="st-code-btn" onClick={copy} disabled={!code}>
          {copied ? <Check size={13} /> : <Copy size={13} />} {copied ? "Copied" : "Copy"}
        </button>
        <button className="st-code-btn" onClick={download} disabled={!code}>
          <Download size={13} /> Export .py
        </button>
      </div>
      <div className="st-code-scroll" ref={scrollRef}>
        {!code ? (
          <p className="st-code-empty">Add a CSV Dataset node to start generating code.</p>
        ) : (
          <pre>
            {lines.map((line, i) => {
              const n = i + 1;
              const owner = lineOwner[n];
              const isActive = owner && owner === selectedId;
              return (
                <div
                  key={n}
                  data-line={n}
                  className={`st-code-line ${owner ? "owned" : ""} ${isActive ? "active" : ""}`}
                  onClick={owner ? () => onSelectNode(owner) : undefined}
                >
                  <span className="st-code-ln">{n}</span>
                  <span className="st-code-text">{highlight(line)}</span>
                </div>
              );
            })}
          </pre>
        )}
      </div>
    </div>
  );
}

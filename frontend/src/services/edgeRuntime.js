/**
 * Edge runtime: run an exported model entirely in the browser.
 *
 * The backend exports two things per run:
 *   - model.onnx            the estimator (numeric feature vector -> prediction)
 *   - preprocessing spec    the fitted recipe steps as plain numbers
 *
 * `preprocessRow` replays the recipe in JavaScript, then `predict` runs the
 * ONNX model with onnxruntime-web (WebAssembly). No request reaches the server.
 */

const isMissing = (v) =>
  v === null || v === undefined || (typeof v === "string" && v.trim() === "") || (typeof v === "number" && Number.isNaN(v));

const isNumericLike = (v) => typeof v === "number" || (typeof v === "string" && v.trim() !== "" && !Number.isNaN(Number(v)));

/** Does a raw input value match a category learned by an encoder? */
function sameCategory(value, category) {
  if (category === null) return isMissing(value);
  if (isMissing(value)) return false;
  if (isNumericLike(value) && isNumericLike(category)) return Number(value) === Number(category);
  return String(value) === String(category);
}

/**
 * Turn one raw row ({column: value}) into the model's Float32 feature vector.
 * Mirrors ml_engine.preprocessing.apply_fitted_transformers.
 */
export function preprocessRow(spec, raw) {
  const row = {};
  for (const input of spec.inputs) {
    const v = raw[input.name];
    if (input.type === "numeric") row[input.name] = isMissing(v) ? NaN : Number(v);
    else row[input.name] = isMissing(v) ? null : v;
  }

  for (const step of spec.steps) {
    const cols = step.columns;
    if (step.step === "drop_columns") {
      cols.forEach((c) => delete row[c]);
    } else if (step.step === "impute") {
      cols.forEach((c, i) => {
        if (c in row && isMissing(row[c])) row[c] = step.values[i];
      });
    } else if (step.step === "scale") {
      cols.forEach((c, i) => {
        if (c in row) row[c] = Number(row[c]) * step.mul[i] + step.add[i];
      });
    } else if (step.step === "encode" && step.method === "onehot") {
      let k = 0;
      cols.forEach((c, i) => {
        const value = row[c];
        for (const cat of step.categories[i]) {
          row[step.output_names[k++]] = sameCategory(value, cat) ? 1 : 0;
        }
        delete row[c];
      });
    } else if (step.step === "encode") {
      cols.forEach((c, i) => {
        // OrdinalEncoder: missing -> NaN (encoded_missing_value), unknown -> -1
        if (isMissing(row[c])) {
          row[c] = NaN;
          return;
        }
        row[c] = step.categories[i].findIndex((cat) => cat !== null && sameCategory(row[c], cat));
      });
    }
  }

  // Align to training feature order; columns that never appeared become 0.
  return Float32Array.from(spec.features, (f) => (f in row ? Number(row[f]) : 0));
}

let ortPromise = null;

async function loadOrt() {
  if (!ortPromise) {
    ortPromise = (async () => {
      const ort = await import("onnxruntime-web/wasm");
      const wasmUrl = (await import("onnxruntime-web/ort-wasm-simd-threaded.wasm?url")).default;
      ort.env.wasm.wasmPaths = { wasm: wasmUrl };
      // Multi-threading needs cross-origin isolation; a single thread is plenty for tabular models.
      ort.env.wasm.numThreads = 1;
      return ort;
    })();
  }
  return ortPromise;
}

/** Create an inference session from the model bytes. */
export async function createSession(modelBytes) {
  const ort = await loadOrt();
  const t0 = performance.now();
  const session = await ort.InferenceSession.create(new Uint8Array(modelBytes), { executionProviders: ["wasm"] });
  return { ort, session, loadMs: performance.now() - t0 };
}

/**
 * Predict one row. Returns { label, confidence, probabilities, value, inferenceMs, features }.
 */
export async function predict({ ort, session }, spec, raw) {
  const features = preprocessRow(spec, raw);
  const tensor = new ort.Tensor("float32", features, [1, features.length]);
  const feeds = { [spec.onnx.input_name]: tensor };
  const t0 = performance.now();

  if (spec.task_type === "classification" && spec.onnx.probabilities_output) {
    const out = await session.run(feeds, [spec.onnx.probabilities_output]);
    const inferenceMs = performance.now() - t0;
    const probs = Array.from(out[spec.onnx.probabilities_output].data);
    let best = 0;
    probs.forEach((p, i) => { if (p > probs[best]) best = i; });
    const probabilities = Object.fromEntries(spec.classes.map((c, i) => [String(c), probs[i]]));
    return { label: String(spec.classes[best]), confidence: probs[best], probabilities, inferenceMs, features };
  }

  const out = await session.run(feeds);
  const inferenceMs = performance.now() - t0;
  const value = Number(out[spec.onnx.outputs[0]].data[0]);
  return { value, inferenceMs, features };
}

# app/app.py  — multi-model API for KMeans / Hier / DBSCAN
import json, os, joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

HERE = os.path.dirname(__file__)
MODEL_DIR = os.path.join(HERE, "model")

def _load_pair(name: str):
    """Load one model + meta pair and normalize types for use."""
    art_path  = os.path.join(MODEL_DIR, f"pipeline_{name}.joblib")
    meta_path = os.path.join(MODEL_DIR, f"meta_{name}.json")
    if not (os.path.exists(art_path) and os.path.exists(meta_path)):
        return None

    art  = joblib.load(art_path)
    meta = json.load(open(meta_path, "r", encoding="utf-8"))

    entry = {
        "model_type": meta["model_type"],         # "kmeans" | "hier" | "dbscan"
        "features":   meta["feature_names"],      # ordered list of feature names
        "model_meta": meta.get("model_meta", {}),
        "preproc":    art["preproc"],
        "model":      art["model"],
        # Convert centroids dict keys to int and values to np.array
        "centroids":  {int(k): np.array(v) for k, v in art["centroids"].items()},
        "radii":      None,
    }
    if art.get("radii") is not None:
        entry["radii"] = {int(k): float(v) for k, v in art["radii"].items()}
    return entry

# Discover and load all available models
MODELS = {}
for name in ["kmeans", "hier", "dbscan"]:
    pair = _load_pair(name)
    if pair is not None:
        MODELS[name] = pair

if not MODELS:
    raise RuntimeError(
        "No models found in app/model/. Run your notebook export (8B) to create "
        "pipeline_kmeans.joblib/meta_kmeans.json etc."
    )

DEFAULT_MODEL = "kmeans" if "kmeans" in MODELS else list(MODELS.keys())[0]
app = Flask(__name__)

# ---------- helpers ----------
def _as_row(payload: dict, features: list[str]) -> pd.DataFrame:
    """Build a single-row DataFrame following the exact feature order."""
    # We allow missing fields (imputer will handle), but fill with None for clarity.
    data = {f: payload.get(f, None) for f in features}
    return pd.DataFrame([data])

def _predict_cluster(entry: dict, X_std: np.ndarray) -> tuple[int, dict]:
    """Return (label, extras). Extras may include distances for non-kmeans."""
    if entry["model_type"] == "kmeans":
        # Use trained kmeans' own predict
        label = int(entry["model"].predict(X_std)[0])
        return label, {"method": "kmeans_predict"}

    # For Hier/DBSCAN we do nearest-centroid in the trained space
    cents = entry["centroids"]
    if not cents:
        # Degenerate case; should not happen if there are clusters
        return 0, {"method": "nearest_centroid", "distances": {}}

    dists = {c: float(np.linalg.norm(X_std[0] - mu)) for c, mu in cents.items()}
    label = min(dists, key=dists.get)

    if entry["model_type"] == "dbscan" and entry["radii"] is not None:
        # classify as noise if farther than learned 95th-percentile radius
        if dists[label] > entry["radii"][label]:
            label = -1

    return label, {"method": "nearest_centroid", "distances": dists}

def _top_factors(entry: dict, X_std: np.ndarray, label: int, k: int = 3):
    """Return k biggest standardized differences from the chosen centroid."""
    if label == -1:
        return []
    mu = entry["centroids"][label]
    delta = np.abs(X_std[0] - mu)
    order = np.argsort(delta)[::-1]
    feats = entry["features"]
    return [{"feature": feats[i], "z_diff": float(delta[i])} for i in order[:k]]

# ---------- routes ----------
@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "available_models": list(MODELS.keys()),
        "default": DEFAULT_MODEL,
    })

@app.get("/metadata")
def metadata():
    # Summaries for each loaded model
    out = {}
    for name, entry in MODELS.items():
        out[name] = {
            "model_type": entry["model_type"],
            "n_features": len(entry["features"]),
            "features_head": entry["features"][:5],
            "model_meta": entry["model_meta"],
        }
    return jsonify(out)

@app.get("/features")
def features():
    # Return ordered feature names for a model (default or requested)
    name = request.args.get("model") or DEFAULT_MODEL
    if name not in MODELS:
        return jsonify({"error": f"Unknown model '{name}'. Available: {list(MODELS.keys())}"}), 400
    return jsonify({"model": name, "features": MODELS[name]["features"]})

@app.post("/predict")
def predict():
    # Choose model: JSON "model" -> query param "model" -> default
    try:
        payload = request.get_json(force=True, silent=False)
        if not isinstance(payload, dict):
            return jsonify({"error": "JSON body must be an object"}), 400
    except Exception as e:
        return jsonify({"error": f"Invalid JSON: {e}"}), 400

    chosen = payload.get("model") or request.args.get("model") or DEFAULT_MODEL
    if chosen not in MODELS:
        return jsonify({"error": f"Unknown model '{chosen}'. Available: {list(MODELS.keys())}"}), 400

    entry = MODELS[chosen]
    feats = entry["features"]

    # Build input row
    X_df = _as_row(payload, feats)

    # Transform using the trained preprocessor
    try:
        X_std = entry["preproc"].transform(X_df)
    except Exception as e:
        return jsonify({"error": f"Preprocessing failed: {e}"}), 400

    # Predict cluster
    label, extras = _predict_cluster(entry, X_std)

    # Explain top k factors
    reasons = _top_factors(entry, X_std, label, k=3)

    # Report which features were missing (helpful for clients)
    missing = [f for f in feats if payload.get(f, None) is None]

    resp = {
        "model": chosen,
        "model_type": entry["model_type"],
        "cluster": int(label) if label != -1 else "noise",
        "country": payload.get("country"),
        "missing_features": missing,         # informational; imputer handles them
        "reasons": reasons,
    }
    if "distances" in extras:
        resp["distances"] = {str(k): float(v) for k, v in extras["distances"].items()}
    return jsonify(resp)

if __name__ == "__main__":
    # Run dev server
    app.run(host="0.0.0.0", port=5000, debug=True)




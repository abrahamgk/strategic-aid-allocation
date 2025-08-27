import argparse, json, joblib, os
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "app", "model")
os.makedirs(MODEL_DIR, exist_ok=True)

def choose_kmeans(X_std, k_min=2, k_max=10, random_state=42):
    best_k, best_s = None, -1
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=20, random_state=random_state)
        labels = km.fit_predict(X_std)
        s = silhouette_score(X_std, labels)
        if s > best_s:
            best_k, best_s = k, s
    final = KMeans(n_clusters=best_k, n_init=50, random_state=random_state).fit(X_std)
    labels = final.labels_
    return final, labels, {"best_k": best_k, "silhouette": float(best_s)}

def choose_hier(X_std, k_min=2, k_max=10):
    best_k, best_s = None, -1
    for k in range(k_min, k_max + 1):
        ac = AgglomerativeClustering(n_clusters=k, linkage="ward")
        labels = ac.fit_predict(X_std)
        s = silhouette_score(X_std, labels)
        if s > best_s:
            best_k, best_s = k, s
    final = AgglomerativeClustering(n_clusters=best_k, linkage="ward")
    labels = final.fit_predict(X_std)
    return final, labels, {"best_k": best_k, "silhouette": float(best_s)}

def choose_dbscan(X_std, min_samples=5, q_low=0.80, q_high=0.97, n_q=18):
    nn = NearestNeighbors(n_neighbors=min_samples).fit(X_std)
    kdist = np.sort(nn.kneighbors(X_std)[0][:, -1])
    eps_grid = np.quantile(kdist, np.linspace(q_low, q_high, n_q))
    best_model, best_labels, best_row = None, None, None
    for eps in eps_grid:
        model = DBSCAN(eps=float(eps), min_samples=min_samples).fit(X_std)
        labels = model.labels_
        n_cl = len(set(labels)) - (1 if -1 in labels else 0)
        noise = (labels == -1).mean()
        sil = np.nan
        if n_cl >= 2 and noise < 0.95 and np.any(labels != -1):
            m = labels != -1
            sil = silhouette_score(X_std[m], labels[m])
        row = (float(eps), n_cl, float(noise), None if np.isnan(sil) else float(sil))
        if best_row is None:
            best_row, best_model, best_labels = row, model, labels
        else:
            curr_key = (row[3] if row[3] is not None else -1, row[1], -row[2])
            best_key = (best_row[3] if best_row[3] is not None else -1, best_row[1], -best_row[2])
            if curr_key > best_key:
                best_row, best_model, best_labels = row, model, labels
    meta = {"eps": best_row[0], "n_clusters": best_row[1], "noise_frac": best_row[2], "silhouette": best_row[3]}
    return best_model, best_labels, meta

def build_preproc():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

def compute_centroids(X_std, labels):
    uniq = sorted([c for c in set(labels) if c != -1])
    return {c: X_std[labels == c].mean(axis=0) for c in uniq}

def compute_radii(X_std, labels, centroids, q=0.95):
    radii = {}
    for c, mu in centroids.items():
        pts = X_std[labels == c]
        d = np.linalg.norm(pts - mu, axis=1)
        radii[c] = float(np.quantile(d, q))
    return radii

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to training CSV")
    ap.add_argument("--model", choices=["kmeans", "hier", "dbscan"], default="kmeans")
    ap.add_argument("--id-col", default="country")
    ap.add_argument("--exclude", nargs="*", default=[], help="Extra columns to exclude")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    exclude_cols = [args.id_col] + args.exclude
    X = df.drop(columns=[c for c in exclude_cols if c in df.columns], errors="ignore")
    feature_names = X.select_dtypes(include=np.number).columns.tolist()

    preproc = build_preproc()
    X_std = preproc.fit_transform(X[feature_names])

    if args.model == "kmeans":
        model, labels, model_meta = choose_kmeans(X_std)
    elif args.model == "hier":
        model, labels, model_meta = choose_hier(X_std)
    else:
        model, labels, model_meta = choose_dbscan(X_std)

    centroids = compute_centroids(X_std, labels)
    radii = compute_radii(X_std, labels, centroids) if args.model == "dbscan" else None

    artifact = {
        "preproc": preproc,
        "model": model,
        "centroids": centroids,
        "radii": radii,
        "labels_train": labels,
    }
    meta = {
        "model_type": args.model,
        "feature_names": feature_names,
        "model_meta": model_meta,
    }

    joblib.dump(artifact, os.path.join(MODEL_DIR, "pipeline.joblib"))
    with open(os.path.join(MODEL_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("Saved:", os.path.abspath(os.path.join(MODEL_DIR, "pipeline.joblib")))
    print("Meta:", json.dumps(meta, indent=2))

if __name__ == "__main__":
    main()

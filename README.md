Trillion-Dollar Transformation: Finding the “Black Holes” of Need

Data + clustering + simple rules for strategic aid allocation

What this repo does

This project turns country-level socio-economic data into an allocation map you can explain on one slide:

Clean & standardize public indicators (mortality, fertility, life expectancy, income, GDP, trade, inflation, health spend).

Discover attractors with clustering (the “black hole” of Actual Need vs the “white hole” of Stable Prosperity).

Explain the split with a tiny, transparent model (decision tree or logistic regression).

Allocate only within need using two auditable rules you can blend:

Priority Index (magnitude of need)

Inverse-Radius (coherence with the need centroid)

Outputs include aligned cluster maps, cluster “personas” (medians), feature importances, and per-country allocation tables.

⚠️ This is an analytical prototype, not policy advice. It’s designed to be simple, transparent, and easy to adapt.

Repo structure
.
├─ notebooks/
│  └─ Strategic_Aid_Allocation.ipynb        # main, end-to-end analysis
├─ src/
│  ├─ clustering_utils.py                    # alignment, palette, label picking, DBSCAN sweep
│  ├─ allocation.py                          # Priority Index & Inverse-Radius allocation
│  ├─ plotting.py                            # EDA, diagnostics, personas, side-by-side panels
│  └─ modeling.py                            # decision tree / logistic regression wrappers
├─ data/
│  ├─ raw/                                   # place the original CSV here (not tracked)
│  └─ processed/                             # standardized arrays or cached artifacts
├─ reports/
│  ├─ figures/                               # exported figures (optional)
│  └─ tables/                                # CSV outputs (allocations, rankings)
├─ requirements.txt
└─ README.md


If you’re only using the notebook, you can ignore src/—the notebook contains everything needed.

🗂️ Data

Source: Public country-level dataset from Kaggle (e.g., “Country Development” style datasets).
Download the CSV and place it at data/raw/countries.csv.

Expected columns (case-insensitive; underscores allowed)
Column	Meaning (units)
country	Country name
child_mort	Under-5 deaths per 1,000 live births
total_fer	Total fertility (births per woman)
life_expec	Life expectancy at birth (years)
income	Net income per person (USD)
gdpp	GDP per capita (USD)
health	Health spending (% of GDP)
inflation	Annual inflation (%)
exports	Exports per capita (% of GDP per capita)
imports	Imports per capita (% of GDP per capita)
Preprocessing

No imputation needed if the dataset has no missing values.

Winsorize outliers on income and gdpp using Tukey’s IQR fences.

Standardize all numeric features (z-score) before clustering.

Quickstart
1) Environment
# Python 3.10+ recommended
python -m venv .venv
source .venv/bin/activate   # (Windows: .venv\Scripts\activate)

pip install -r requirements.txt


Minimal install:

pip install numpy pandas matplotlib seaborn scikit-learn scipy adjustText joblib

2) Data

Place your CSV at:

data/raw/countries.csv

3) Run

Notebook: open notebooks/Strategic_Aid_Allocation.ipynb and run all cells.

Outputs: generated in reports/tables/ (CSV) and displayed inline (figures).
You can optionally save figures to reports/figures/.

What the notebook does

EDA: KDEs, top/bottom bars, correlation heatmap; quick statistical checks (correlations + Welch t-tests).

Clustering (auditable):

Standardize features; choose K using elbow/silhouette diagnostics.

Fit K-Means and Ward (Agglomerative) for the chosen K.

Optional DBSCAN search over sensible grids; pick runs that yield clean, low-noise groups.
For side-by-side comparison, DBSCAN clusters are mapped to the nearest K-Means centroid (noise stays -1).

Label semantics (data-driven): infer which numeric cluster is “Actual Need” vs “Stable Prosperity” using a simple need score:
(+ child_mort, + total_fer, + inflation) − (life_expec + income + gdpp).

Visualization: side-by-side cluster maps (PCA(2) for display only), ellipses, centroids, and consistent labels across panels.

Personas: bar “cards” (medians in original units) for each cluster and algorithm.

Transparency: train a tiny Decision Tree (or Logistic Regression) to predict the cluster; show feature importances / standardized coefficients.

Allocation (Need only):

Method A — Priority Index (magnitude): equal-weight z-score average after flipping “good” metrics so bigger = worse.

Method B — Inverse-Radius (coherence): weight ∝ 1 / (distance + ε) to the Need centroid in standardized space.

Blend (e.g., 70% B + 30% A), apply floors/caps, renormalize.

Saves CSVs:
table_top15_global_priority.csv,
table_top10_within_cluster.csv,
table_allocation_proposal.csv.

Key parameters (where to tweak)

Choosing K: Use elbow + silhouette plots; K=2 is a sensible default for “Need vs Prosperity,” but set any K as required.

NAME_MAP inference: automatic, based on mean need score per cluster.

Label selections in plots: pick_labels_spread ensures diverse, well-spaced annotations (policy seeds + extremes + coverage).

DBSCAN search: the grid over (eps, min_samples) uses k-distance quantiles + knee; we prioritize clean, low-noise, well-separated outcomes.

Allocation guardrails: MIN_PER_COUNTRY, MAX_PER_COUNTRY, and A/B blending weights.

All parameters live near the top of the relevant notebook sections (and in src/ if you prefer scripts).

Re-use the artifacts (optional)

Utilities are included to export clustering pipelines (preprocessing + model, centroids, and per-cluster radii for DBSCAN) with joblib.
Artifacts can be loaded later to label new rows or to compute centroid distances.

Outputs you’ll get

Figures: EDA KDEs, correlation heatmap, top/bottom bars, K-Means/Ward diagnostics, side-by-side cluster maps, personas, feature importances, allocation plots.

Tables (CSV):

reports/tables/table_top15_global_priority.csv

reports/tables/table_top10_within_cluster.csv

reports/tables/table_allocation_proposal.csv

Extending the analysis

More than two clusters: set K>2; label multiple “need” attractors; allocate within those clusters with weights.

Fairness / policy constraints: add floors by region, caps per program, or eligibility filters (governance/readiness).

Time series: repeat by year; track movement toward/away from attractors.

Alternative representations: try contrastive embeddings, graph Laplacians, or causal feature sets—allocation recipes remain the same.

Questions or suggestions?
Open an issue or reach out: gkabraham33@gmail.com

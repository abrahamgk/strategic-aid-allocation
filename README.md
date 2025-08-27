Trillion-Dollar Transformation: Finding the “Black Holes” of Need

Data + clustering + simple rules for strategic aid allocation

What this repo does

Turn country-level socio-economic data into an allocation map you can explain on one slide:

Clean & standardize public indicators (mortality, fertility, life expectancy, income, GDP, trade, inflation, health spend).

Discover attractors with clustering (the “black hole” of Actual Need vs the “white hole” of Stable Prosperity).

Explain the split with a tiny, transparent model (decision tree or logistic regression).

Allocate only within need using two auditable rules you can blend:

Priority Index → magnitude of need.

Inverse-Radius → coherence with the need centroid.

Outputs: aligned cluster maps, cluster “personas” (medians), feature importances, and per-country allocation tables.

⚠️ Prototype, not policy advice. The goal is transparency and easy adaptation.

Repo structure
.
├─ notebooks/
│  └─ Strategic_Aid_Allocation.ipynb      # end-to-end analysis
├─ src/
│  ├─ clustering_utils.py                  # alignment, label mapping, DBSCAN sweep
│  ├─ allocation.py                        # Priority Index & Inverse-Radius rules
│  ├─ plotting.py                          # EDA, diagnostics, personas, side-by-side panels
│  └─ modeling.py                          # decision tree / logistic regression wrappers
├─ data/
│  ├─ raw/                                 # put original CSV here (not tracked)
│  └─ processed/                           # cached artifacts (optional)
├─ reports/
│  ├─ figures/                             # exported figures (optional)
│  └─ tables/                              # CSV outputs (allocations, rankings)
├─ requirements.txt
└─ README.md


Using only the notebook? You can ignore src/. The notebook contains everything you need.

Data

Source: public country-level dataset from Kaggle (e.g., “Country Development” style).
Place the CSV at: data/raw/countries.csv.

Expected columns
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

Winsorize outliers for income and gdpp with Tukey’s IQR fences.

Z-score standardize all numeric features before clustering.

Quickstart
1) Environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
# or minimal
pip install numpy pandas matplotlib seaborn scikit-learn scipy adjustText joblib

2) Data
# put your CSV here
data/raw/countries.csv

3) Run

Open notebooks/Strategic_Aid_Allocation.ipynb and run all cells.

Outputs appear in reports/tables/ (CSV) and display inline (figures).
Optionally save figures to reports/figures/.

What the notebook does

EDA: KDEs, top/bottom bars, correlation heatmap; Welch t-tests + correlations.

Clustering (auditable):

Standardize features; pick K via elbow + silhouette.

Fit K-Means and Ward (Agglomerative).

Optional DBSCAN grid search (k-distance quantiles + knee); select clean, low-noise runs.
For side-by-side visuals, map DBSCAN clusters to the nearest K-Means centroid (keep noise as -1).

Label semantics (data-driven): infer which numeric cluster is Actual Need vs Stable Prosperity via a simple need score
(+ child_mort + total_fer + inflation) − (life_expec + income + gdpp).

Visualization: aligned cluster maps (PCA(2) for display only), ellipses, centroids, consistent labels.

Personas: bar “cards” (medians, original units) per cluster and algorithm.

Transparency: tiny Decision Tree (or Logistic Regression) to explain the split (feature importances / standardized coefficients).

Allocation (Need only):

Method A — Priority Index (magnitude): equal-weight z-score average after flipping “good” metrics so bigger = worse.

Method B — Inverse-Radius (coherence): weight ∝ 1 / (distance + ε) to the Need centroid.

Blend (e.g., 70% B + 30% A), apply floors/caps, renormalize.

Saves:

reports/tables/table_top15_global_priority.csv

reports/tables/table_top10_within_cluster.csv

reports/tables/table_allocation_proposal.csv

Key parameters to tweak

K (number of clusters): choose via elbow + silhouette (K=2 is a sensible default for Need vs Prosperity).

DBSCAN search: eps/min_samples grid from k-distance quantiles + knee; prefer clean, low-noise, high-silhouette outcomes.

Label inference: mean need score per cluster determines “Actual Need” vs “Stable Prosperity”.

Plot labeling: pick_labels_spread yields diverse, well-spaced country annotations.

Allocation guardrails: min/max per country, and A/B blending weights.

All are surfaced in the notebook (and mirrored in src/ if you script it).

Re-use artifacts (optional)

Utilities let you export preprocessing + clustering with joblib (including centroids).
Load later to label new rows and compute centroid distances.

Outputs you’ll get

Figures: KDEs, top/bottom bars, correlation heatmap, K-Means/Ward diagnostics, side-by-side cluster maps, personas, feature importances, allocation plots.

Tables (CSV):

reports/tables/table_top15_global_priority.csv

reports/tables/table_top10_within_cluster.csv

reports/tables/table_allocation_proposal.csv

Extend the analysis

>2 clusters: set K>2; label multiple “need” attractors; allocate within each need cluster with weights.

Fairness / policy constraints: add regional floors, per-program caps, readiness filters.

Time series: repeat by year; track movement toward/away from attractors.

Alternative reps: contrastive embeddings, graph Laplacians, or causal features—the allocation recipes remain the same.

Questions / suggestions?
Open an issue or reach me at gkabraham33@gmail.com
.

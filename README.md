# Trillion-Dollar Transformation: Finding the “Black Holes” of Need

Data + clustering + simple rules for strategic aid allocation

## What this repo does

Turn country-level socio-economic data into an allocation map you can explain on one slide:

Clean & standardize public indicators (mortality, fertility, life expectancy, income, GDP, trade, inflation, health spend).

Discover attractors with clustering (the “black hole” of Actual Need vs the “white hole” of Stable Prosperity).

Explain the split with a tiny, transparent model (decision tree or logistic regression).

Allocate only within need using two auditable rules you can blend:

Priority Index → magnitude of need

Inverse-Radius → coherence with the need centroid

Outputs: aligned cluster maps, cluster “personas” (medians), feature importances, and per-country allocation tables.

⚠️ Prototype, not policy advice. The goal is transparency and easy adaptation.

## Live demo & code

Interactive dashboard (Tableau Public):
https://public.tableau.com/app/profile/abraham.g.k/viz/StrategicAidAllocation_17545416983020/Dashboard1

GitHub repository (this project):
https://github.com/abrahamgk/strategic-aid-allocation


## Data

Source: public country-level dataset from Kaggle (e.g., “Country Development” style).
Place the CSV at: data/raw/countries.csv.

## Expected columns

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

## Preprocessing

Winsorize outliers for income and gdpp with Tukey’s IQR fences.
Z-score standardize all numeric features before clustering.

## Key parameters to tweak

K (number of clusters): choose via elbow + silhouette (K=2 is a sensible default for Need vs Prosperity).
DBSCAN search: eps/min_samples grid from k-distance quantiles + knee; prefer clean, low-noise, high-silhouette outcomes.
Label inference: mean need score per cluster determines “Actual Need” vs “Stable Prosperity”.
Plot labeling: pick_labels_spread yields diverse, well-spaced country annotations.
Allocation guardrails: min/max per country, and A/B blending weights.
All are surfaced in the notebook (and mirrored in src/ if you script it).

## Questions / suggestions?

Open an issue or reach me at gkabraham33@gmail.com


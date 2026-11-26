<div align="center">

# Project Sentinel

### Financial Transaction Intelligence & Risk Analytics

![Python](https://img.shields.io/badge/PYTHON-3.10%2B-3776ab?style=for-the-badge) ![SQL](https://img.shields.io/badge/SQL-SQLITE-003b57?style=for-the-badge) ![ML](https://img.shields.io/badge/ML-SCIKIT--LEARN-f7931e?style=for-the-badge) ![Dashboard](https://img.shields.io/badge/DASHBOARD-STREAMLIT-c53030?style=for-the-badge) ![Status](https://img.shields.io/badge/STATUS-PORTFOLIO-0f766e?style=for-the-badge)

**Payment operations, settlement reconciliation and synthetic risk modelling on a reproducible dataset.**

[Data dictionary](data/README.md) · [SQL queries](sql/analysis.sql) · [Dashboard source](app.py) · [Tests](tests/test_sentinel.py)

</div>

---

## What this project does

I developed Sentinel to connect payment operations problems with quantitative analysis. It investigates provider approval and failure rates, processing latency, settlement discrepancies and the ability of machine learning models to distinguish simulated risk labels.

The default generator creates **100,000 fictional payment attempts** using a fixed random seed. This is a design target, not a claim that a dataset has already been published. No real transactions, customer identifiers or confidential employer information are used.

## Analytical workstreams

| Workstream | Implementation |
| --- | --- |
| Payment operations | Approval, failure and pending rates; attempted volume and daily activity |
| Provider benchmarking | Fictional PSP-level rates, median and 95th percentile latency |
| Settlement reconciliation | Expected versus actual settlement; signed and absolute discrepancies |
| Risk modelling | Random Forest classifier and Isolation Forest anomaly scoring |
| Evaluation | Chronological 80/20 holdout; ROC-AUC, average precision, precision, recall, F1 and confusion matrix |
| Business intelligence | Streamlit dashboard with provider and direction filters and CSV export |

## Architecture

```mermaid
flowchart LR
 A[Synthetic generator] --> B[Validation]
 B --> C[(SQLite)]
 B --> D[Operational analytics]
 B --> E[Temporal model evaluation]
 D --> F[Streamlit dashboard]
 E --> F
 D --> G[CSV outputs]
```

## Quick start

Python 3.10+ recommended.

```bash
python -m venv .venv
# Activate your virtual environment.
pip install -r requirements.txt
python -m sentinel.pipeline --rows 100000 --seed 42
streamlit run app.py
```

The pipeline generates a local CSV and SQLite database, daily and provider summaries, reconciliation exceptions and model metrics. Generated data and outputs are excluded from Git. The dashboard independently generates the same default synthetic dataset.

## Repository structure

```text
sentinel/
  generate.py    Fictional transaction generator
  analysis.py    Validation and operational metrics
  model.py       Temporal model evaluation
  pipeline.py    Reproducible data and analysis workflow
sql/             Example analytical SQL
data/            Data dictionary and generated dataset location
outputs/         Generated research outputs
tests/           Automated validation
app.py           Streamlit dashboard
```

## Validation and limitations

```bash
python -m unittest discover -s tests -v
```

Approval and failure denominators are all attempted transactions, including pending attempts. Volumes are gross attempted KES, not revenue. Settlement amounts are deliberately simplified and do not model actual provider fee contracts. Model performance on synthetic labels measures simulation behavior only, not real fraud detection capability. No AML compliance conclusions or production decisions should be drawn from the simulated dataset.

## Project status

- [x] Implemented reproducible generator, SQL database pipeline and operational metrics
- [x] Implemented Random Forest and Isolation Forest evaluation
- [x] Implemented interactive dashboard and automated test definitions
- [ ] Verify CI, execute full-size pipeline and publish verified sample outputs
- [ ] Add exploratory notebook and hosted dashboard

**Author:** [Samuel Ndungu Njoroge](https://github.com/samuel6183) · [LinkedIn](https://www.linkedin.com/in/samuel-ndungu-6800b8194/)

## Research notebook and generated results

[Open exploratory notebook](notebooks/sentinel_analysis.ipynb) · [Research workflow runs](https://github.com/samuel6183/project-sentinel/actions/workflows/research.yml)

The full research workflow runs the 100,000-row pipeline, executes the notebook, generates charts and an executive summary, and publishes downloadable workflow artifacts after successful completion. GitHub artifact downloads may require sign-in and expire after 90 days. No generated metrics are claimed until execution succeeds.

To generate the report locally after the pipeline:

```bash
python -m sentinel.report
```

For dashboard hosting on Streamlit Community Cloud, select this public repository, branch main and entry point app.py. Hosting requires an authorized Streamlit account; no hosted dashboard is claimed yet.

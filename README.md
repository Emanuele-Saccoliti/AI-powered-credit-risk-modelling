# AI-Powered Credit Risk Modelling

An end-to-end, notebook-based credit-risk project built on the Home Credit Default Risk dataset. It combines interpretable Probability of Default modelling, a nonlinear challenger, bank-style validation and stress testing, and a governed AI Early-Warning System based on controlled external information.

The project is designed as a reproducible portfolio and learning artifact. It is not a production lending, pricing, regulatory-capital, or automated decisioning system.


## Table of Contents

- [Project objectives](#project-objectives)
- [Architecture](#architecture)
- [Key capabilities](#key-capabilities)
- [Repository structure](#repository-structure)
- [Getting started](#getting-started)
- [Home Credit data](#home-credit-data)
- [Configuration](#configuration)
- [AI Early-Warning System](#ai-early-warning-system)
- [Validation and governance](#validation-and-governance)
- [Limitations](#limitations)

## Project objectives

- Build calibrated and interpretable borrower-level PD models.
- Compare a Classical Logistic benchmark, an interpretable ElasticNet Logistic model, and a nonlinear LightGBM challenger.
- Engineer behavioural risk drivers from the complete Home Credit relational dataset.
- Validate discrimination, calibration, stability, and segment performance using bank-style diagnostics.
- Measure PD sensitivity under adverse and severe behavioural stress scenarios.
- Extract emerging portfolio-risk signals from raw external documents while keeping the AI EWS separate from calibrated PD.

## Architecture

### Quantitative credit-risk workflow

```text
Home Credit application and behavioural tables
                         ↓
          Schema, key, and leakage audits
                         ↓
       Borrower-level relational aggregation
                         ↓
         Interpretable feature engineering
                         ↓
 Classical Logistic → ElasticNet → LightGBM
                         ↓
       Out-of-fold probability calibration
                         ↓
 Validation → PSI monitoring → Stress testing
                         ↓
          Coefficients, SHAP, model card
```

### External-information workflow

```text
Controlled web sources / NewsAPI / local documents
                         ↓
                Unstructured raw text
                         ↓
              Structured LLM extraction
                         ↓
            Pydantic and application checks
                         ↓
             Deterministic EWS aggregation
                         ↓
      LOW / MEDIUM / HIGH / INSUFFICIENT_EVIDENCE
```

The quantitative PD and AI EWS are displayed together for monitoring but are never blended into a single score.

## Key capabilities

### Behavioural credit-risk data

The notebook can aggregate the following Home Credit tables to one row per `SK_ID_CURR`:

- `application_train.csv` and `application_test.csv`;
- `bureau.csv` and `bureau_balance.csv`;
- `previous_application.csv`;
- `installments_payments.csv`;
- `POS_CASH_balance.csv`;
- `credit_card_balance.csv`.

Engineered drivers cover affordability, leverage, bureau debt, delinquency, payment shortfall, repayment behaviour, previous refusals, revolving utilization, and economically interpretable interactions.

### PD model development

- **Classical Logistic Regression:** transparent benchmark.
- **ElasticNet Logistic Regression:** regularized and interpretable candidate model.
- **LightGBM:** nonlinear challenger for interactions and nonlinear effects.
- All imputers, encoders, scalers, tuning, and calibration remain inside leakage-safe training pipelines.
- Data are split into 60% training, 20% validation, and 20% test populations.

### Model validation

The notebook reports:

- ROC-AUC;
- Gini;
- Kolmogorov-Smirnov statistic;
- PR-AUC;
- Brier score;
- observed versus predicted PD;
- calibration gaps and calibration curves;
- performance by borrower segment;
- PSI diagnostics for features and calibrated PD.

The champion is selected using validation data. The test set is reserved for final evaluation.

### Behavioural stress testing

The scenario engine includes `BASELINE`, `ADVERSE`, and `SEVERE` scenarios across:

- affordability and income;
- annuity burden and leverage;
- installment lateness and bureau overdue measures;
- repayment behaviour;
- revolving credit utilization;
- previous-application refusal ratios.

Outputs include mean and median PD, P90/P95/P99 PD, absolute and relative uplift, segment-level impact, and borrower migration into higher risk buckets.

### Explainability

- ElasticNet coefficients and odds ratios;
- LightGBM global SHAP importance;
- borrower-level SHAP reason codes;
- explicit model assumptions, intended use, limitations, and model card.

## Repository structure
```
The project intentionally keeps the implementation in a single notebook.

## Getting started

### 1. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd AI-powered-CreditRisk-Modelling
```

### 2. Create a Python environment

Python 3.12 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip jupyterlab
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Open the notebook

```bash
jupyter lab v2.ipynb
```

Run every cell from top to bottom. The first code cell installs missing runtime packages, including pandas, scikit-learn, LightGBM, SHAP, DuckDB, Pydantic, OpenAI, Kaggle, and Requests.

For a faster technical run:

```bash
export HC_FAST_MODE=1
export HC_MAX_ROWS=30000
jupyter lab v2.ipynb
```

## Home Credit data

The dataset is not included in this repository. Download access is governed by the [Home Credit Default Risk competition](https://www.kaggle.com/competitions/home-credit-default-risk) on Kaggle.

The notebook searches for data in:

1. `HOME_CREDIT_DATA_DIR`;
2. the local `data/` directory;
3. the repository directory;
4. `/kaggle/input/home-credit-default-risk`.

If `application_train.csv` is missing and `HC_AUTO_DOWNLOAD=1`, the notebook attempts:

```bash
python -m kaggle competitions download -c home-credit-default-risk -p data
```

Before using automatic download:

1. accept the competition rules on Kaggle;
2. configure Kaggle credentials using the official Kaggle CLI method;
3. never commit `kaggle.json` or credential values to the repository.

Alternatively, point the notebook to an existing dataset directory:

```bash
export HOME_CREDIT_DATA_DIR=/absolute/path/to/home-credit-default-risk
```

## Configuration

The main runtime controls are environment variables.

### Data and modelling

| Variable | Purpose | Default |
|---|---|---:|
| `HOME_CREDIT_DATA_DIR` | Directory containing Home Credit CSV files | `data` |
| `HC_AUTO_DOWNLOAD` | Attempt Kaggle download if data are missing | `1` |
| `HC_MAX_ROWS` | Maximum application rows; `0` uses all rows | `120000` |
| `HC_FAST_MODE` | Reduce tuning and SHAP workloads | `0` |
| `HC_DEEP_RELATIONAL_AUDIT` | Run full duplicate-key checks on large tables | `0` |
| `HC_DUCKDB_THREADS` | DuckDB execution threads | `4` |
| `HC_N_JOBS` | Parallel model-training jobs | `-1` |
| `HC_SHAP_ROWS` | Maximum SHAP sample size | `600` |

### External evidence and AI EWS

| Variable | Purpose | Default |
|---|---|---:|
| `EWS_EVIDENCE_MODE` | `auto`, `file`, `openai_web`, `newsapi`, `combined`, or `demo` | `auto` |
| `EWS_EVIDENCE_PATH` | CSV, JSON, or JSONL raw-evidence file | empty |
| `EWS_WEB_SEARCH_QUERY` | Query for controlled OpenAI Web Search | configured query |
| `EWS_ALLOWED_WEB_DOMAINS` | Comma-separated web-search allowlist | official-source list |
| `EWS_ENABLE_OPENAI_WEB_IN_AUTO` | Opt in to web search in `auto` mode | `0` |
| `EWS_NEWS_QUERY` | NewsAPI search query | configured query |
| `EWS_NEWSAPI_DOMAINS` | Optional comma-separated NewsAPI domains | empty |
| `EWS_EVIDENCE_TEXT_LIMIT` | Maximum raw-text characters per item | `5000` |
| `RUN_LLM_EWS` | Run structured LLM extraction | `0` |
| `OPENAI_EWS_MODEL` | OpenAI model used for extraction | notebook configuration |
| `EWS_MEDIUM_THRESHOLD` | Medium EWS threshold | `1.50` |
| `EWS_HIGH_THRESHOLD` | High EWS threshold | `3.00` |
| `EWS_MIN_COVERAGE` | Minimum factor coverage | `0.50` |

Secrets must be supplied through the environment:

```bash
export OPENAI_API_KEY=<YOUR_OPENAI_API_KEY>
export NEWS_API_KEY=<YOUR_NEWS_API_KEY>
```

Do not add real keys to the notebook, README, shell scripts, or version control.

## AI Early-Warning System

### Evidence modes

| Mode | Behaviour |
|---|---|
| `file` | Loads raw evidence from a local CSV, JSON, or JSONL file. |
| `openai_web` | Searches only configured domains and revalidates every returned URL locally. |
| `newsapi` | Uses NewsAPI directly through `NEWS_API_KEY`, with optional domain filtering. |
| `combined` | Combines every available real source and deduplicates by URL or evidence ID. |
| `auto` | Uses local evidence and NewsAPI when available; OpenAI Web Search requires explicit opt-in. |
| `demo` | Uses clearly labelled synthetic raw documents for architecture testing only. |

The canonical evidence schema contains:

```text
evidence_id, timestamp, source, title, raw_text, url, retrieval_method, is_demo
```

Upstream evidence is intentionally not pre-classified. The LLM interprets raw titles and text and returns exactly one structured signal for each governed factor:

- liquidity stress;
- refinancing pressure;
- household affordability stress;
- employment risk;
- housing-market risk;
- sector deterioration.

Pydantic and application checks reject invalid structures, duplicated factors, missing governed factors, and invented evidence IDs. Transparent rules then aggregate supported signals into the final EWS.

If the LLM is disabled, unavailable, or fails, the notebook does not fabricate a fallback classification. It reports `NOT_RUN` or a failure status instead.

## Validation and governance

Key governance controls include:

- target-free behavioural aggregation;
- fold-safe preprocessing, tuning, and calibration;
- validation-only model selection;
- immutable source features during stress testing;
- source provenance and retrieval audits;
- local URL validation after controlled web retrieval;
- prompt-injection controls that treat external text as untrusted data;
- strict separation between quantitative PD and AI EWS;
- demo evidence marked as non-live and non-production;
- explicit implementation checklist and model card.

> **The AI overlay is a monitoring signal and is not part of the calibrated PD model.**

## Limitations

- Home Credit `TARGET` is a competition payment-difficulty proxy, not a documented regulatory default definition.
- The dataset does not provide a defensible calendar alignment between borrower observations and macroeconomic conditions.
- Stress scenarios are transparent sensitivity tests, not Point-in-Time forecasts or historically calibrated recession models.
- PSI is descriptive when a dated production monitoring population is unavailable.
- Logistic coefficients and SHAP values describe model associations, not causal effects.
- External evidence requires independent controls for source quality, licensing, retention, deduplication, and model evaluation.
- Demo documents are synthetic and must never be presented as live risk evidence.
- The AI EWS has not demonstrated incremental borrower-level predictive value and therefore remains outside calibrated PD.

## Responsible use

This repository is intended for research, education, and portfolio demonstration. Any real-world credit use would require independent validation, regulatory and legal review, data-governance approval, fairness testing, security controls, and ongoing model-risk management.

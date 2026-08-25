AeroRecover AI

Explainable Flight-Delay Propagation Decision Support

AeroRecover AI is a research-grade decision-support prototype for predicting whether delay attributed to a late inbound aircraft will propagate to its next flight. It combines flight-level XGBoost predictions, aircraft-rotation graphs, SHAP explanations, multi-hop chain tracing, operational prioritization, a historical Streamlit replay, and downloadable decision reports.

Scope: Historical decision support and research demonstration. AeroRecover AI is not a live airline operations, dispatch, or safety-critical system.

At a Glance

Item

Result

Raw 2023 BTS flight records

6,847,899

Aircraft-rotation samples

4,796,459

Model features

24

Locked-test ROC-AUC

0.9914

Locked-test average precision

0.9030

Locked-test F1

0.8128

Multi-hop chain-start F1

0.7089

System Overview

flowchart TD
    A[BTS flight records] --> B[Aircraft rotations]
    B --> C[Full-enhanced XGBoost]
    C --> D[Scored rotation graph]
    C --> E[SHAP explanations]
    D --> F[Multi-hop tracing]
    E --> G[Decision support]
    F --> G
    G --> H[Historical replay and PDF report]

The model scores one physically connected aircraft-rotation edge at a time. Alerted edges are then connected chronologically to represent predicted downstream propagation chains. A separate decision layer combines model likelihood, graph impact, and turnaround urgency into a P1–P4 operational priority and human-reviewable recommendations.

Research Design

Dataset

The project uses the 2023 U.S. Bureau of Transportation Statistics On-Time Performance dataset.

Stage

Records

Raw flight records

6,847,899

Valid flights after eligibility rules

6,743,404

Physically connected rotation samples

4,796,459

Chronological Split

Random splitting was not used. Model development and evaluation follow operational time order.

Split

Period

Samples

Purpose

Training

January–August 2023

3,159,311

Parameter learning

Validation

September–October 2023

832,022

Model and threshold selection

Locked test

November–December 2023

805,126

One-pass final evaluation

Prediction Target

The positive class is defined as:

IS_DELAY_PROPAGATED = 1

when the downstream flight contains at least 15 minutes of BTS-reported LATE_AIRCRAFT_DELAY.

The predicted probability belongs to a single consecutive aircraft-rotation edge. It is neither a causal effect nor a calibrated joint probability for an entire multi-flight chain.

Full-Enhanced Prediction Model

The production research model is an XGBoost classifier using 24 leakage-audited features. Its validation-selected threshold, 0.47, was frozen before the locked November–December test was scored.

Feature groups include:

aircraft-rotation state and previous-flight delay;

planned turnaround capacity and remaining buffer;

carrier, destination, distance, and flight context;

cyclical schedule, calendar, and weekend context;

operational interaction features derived only from information available in the constructed rotation record.

<details>
<summary>Show the exact 24-feature contract</summary>

PREV_DEST, PREV_DELAY_LEVEL, DEST, OP_UNIQUE_CARRIER, ROTATION_POSITION, PREV_ARR_DELAY, PREV_ARR_MIN, PREV_CRS_ARR_MIN, PLANNED_TURNAROUND, TURN_BUFFER, PREV_DELAY_RATIO, HAS_BUFFER, IS_SHORT_TURN, PREV_DELAYED, DISTANCE, CRS_DEP_MIN_SAFE, CRS_DEP_TIME_SIN, CRS_DEP_TIME_COS, DAY_OF_WEEK, MONTH, IS_WEEKEND, DELAY_EXCESS_OVER_TURN, AVAILABLE_BUFFER_RATIO, and PREV_DELAY_SHORT_TURN_INTERACTION.

</details>

Locked-Test Performance

Results below come from the untouched November–December 2023 test period using the frozen 0.47 threshold.

Metric

Result

Samples

805,126

Positive propagation events

49,038

Alert rate

6.71%

Accuracy

0.9760

Precision

0.7751

Recall

0.8544

F1

0.8128

ROC-AUC

0.9914

Average precision / PR-AUC

0.9030

Log loss

0.0543

Brier score

0.0164

Confusion-matrix counts: TN = 743,928, FP = 12,160, FN = 7,140, and TP = 41,898.

Aircraft-Rotation Graph and Multi-Hop Evaluation

Each flight is represented as a vertex. A directed edge connects two consecutive flights only when they are operated by the same aircraft and the preceding destination matches the downstream origin.

Locked-test graph result

Value

Physical edges

832,421

Eligible edges

808,124

Scored edges

805,126

Chain-start precision

0.6743

Chain-start recall

0.7472

Chain-start F1

0.7089

Exact matched chain-length rate

0.8896

Graph metrics evaluate the start and extent of predicted propagation sequences. They are stricter than flight-level classification metrics and should not be interpreted interchangeably.

Explainable Decision Support

For a selected historical rotation, AeroRecover AI provides:

edge-level propagation probability;

likelihood, downstream impact, and urgency assessment;

P1 Critical to P4 Normal operational priority;

local SHAP risk-increasing and risk-decreasing contributions;

predicted downstream rotation chain;

recommendation owner, timing, objective, and feasibility note;

downloadable PDF decision report.

SHAP values explain contributions to the model's raw score. They are not causal effects or direct percentage-point changes in probability.

Historical Replay Dashboard

The Streamlit interface includes historical date, carrier, route, graph-length, and risk filters; animated propagation-map playback; decision cards; local explanations; domino-chain visualization; structured recommendations; and PDF report generation.

Map movement is coordinate interpolation used to communicate graph order. It does not represent recorded aircraft trajectories.

Project Structure

app/                         Streamlit application and replay page
models/                      Frozen model artifacts and contracts
results/                     Evaluation outputs and paper figures
reports/                     Audit, reproducibility, and metric reports
src/analysis/                Rotation-dataset construction
src/features/                Shared enhanced feature engineering
src/models/                  Training and final-model evaluation
src/evaluation/              Flight, graph, policy, and error analysis
src/graph/                   Edge scoring and multi-hop tracing
src/decision_support/        Assessment and recommendation logic
src/explainability/          Local and global SHAP analysis
src/reporting/               PDF decision-report generation
src/visualization/           Replay maps and scientific figures
tests/                       Feature-contract and integration tests

Quick Start

1. Clone and create an environment

git clone <repository-url>
cd AeroRecover-AI
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

2. Run the dashboard

python -m streamlit run app\streamlit_app.py

3. Run the automated tests

python -m pytest -q

Large BTS source files, processed datasets, and generated model/report artifacts may be excluded from version control. If an artifact is absent, reproduce it using the corresponding module under src/ before launching dependent workflows.

Reproducibility Controls

chronological training, validation, and locked-test periods;

threshold selection restricted to validation data;

a frozen 24-feature model contract;

leakage and missing-value audits;

consistent feature engineering across training, scoring, SHAP, graph, replay, and reporting paths;

saved metric, graph, priority, error-analysis, and paper-figure manifests;

automated tests for feature parity and inference integration.

Limitations

The final model uses BTS operational records and does not include live crew, gate, maintenance, airport-resource, or passenger-connection status.

Weather is not part of the final 24-feature model.

Historical replay is not a live forecasting environment.

Recommendations have not been validated in a real airline operations-control trial.

Predictive associations, probabilities, and SHAP values must not be interpreted as causal effects.

Resource feasibility must be confirmed by qualified operational personnel using current information.

Future Work

Planned research directions include multi-flight history, airport congestion and weather context, resource-constrained recovery optimization, richer graph-learning methods, probability recalibration under distribution shift, and validation with real operational stakeholders.

Responsible Use

AeroRecover AI supports research and historical decision analysis. It must not issue automatic operational commands or replace qualified decisions involving safety, dispatch, crew legality, maintenance, airport capacity, regulation, or network recovery.

Citation

An IEEE-style manuscript accompanies this project. Add the final manuscript to the repository documentation or release assets and cite that version when reusing the methodology or results
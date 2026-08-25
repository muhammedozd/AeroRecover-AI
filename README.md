AeroRecover AI

Explainable decision support for flight-delay propagation across aircraft rotations.

AeroRecover AI predicts whether delay from an inbound aircraft will propagate to its next flight, traces the resulting risk through aircraft-rotation chains, and presents explainable operational recommendations.

Research prototype based on historical data. It is not a live airline operations or safety-critical system.

Highlights

6.8 million 2023 BTS flight records processed

4.8 million physically connected aircraft-rotation samples

Chronological train, validation, and locked-test design

Leakage-audited 24-feature XGBoost model

SHAP-based local and global explanations

Multi-hop aircraft-rotation graph analysis

P1–P4 operational prioritization

Historical Streamlit replay and downloadable PDF reports

Workflow

BTS flight records
        ↓
Aircraft-rotation construction
        ↓
XGBoost propagation prediction
        ↓
SHAP explanation + multi-hop graph tracing
        ↓
Operational priority and recommendations
        ↓
Historical Streamlit replay and PDF report

Each prediction applies to one pair of consecutive flights operated by the same aircraft. Alerted edges are connected chronologically to reveal possible downstream propagation chains.

Data and Evaluation Design

The project uses the 2023 U.S. Bureau of Transportation Statistics On-Time Performance dataset. Random splitting was not used.

Dataset stage

Period

Samples

Raw flight records

Jan–Dec 2023

6,847,899

Rotation samples

Jan–Dec 2023

4,796,459

Training

Jan–Aug 2023

3,159,311

Validation

Sep–Oct 2023

832,022

Locked test

Nov–Dec 2023

805,126

The positive class represents at least 15 minutes of BTS-reported late-aircraft delay on the downstream flight. The decision threshold was selected on validation data and frozen at 0.47 before final testing.

Locked-Test Performance

Metric

Result

Accuracy

0.9760

Precision

0.7751

Recall

0.8544

F1 score

0.8128

ROC-AUC

0.9914

PR-AUC

0.9030

Brier score

0.0164

These results were obtained on the untouched November–December 2023 test period using the frozen enhanced model and threshold.

Multi-Hop Graph Evaluation

Flights are graph nodes, while consecutive flights operated by the same aircraft form directed edges.

Metric

Result

Chain-start precision

0.6743

Chain-start recall

0.7472

Chain-start F1

0.7089

Exact matched chain-length rate

0.8896

Graph evaluation is stricter than flight-level classification because it measures whether the system identifies both the beginning and the extent of a propagation sequence.

Decision-Support Interface

The Streamlit dashboard provides:

historical flight, airline, route, and risk filters;

animated predicted propagation maps;

likelihood, impact, urgency, and P1–P4 priority cards;

local SHAP risk explanations;

multi-hop domino-chain visualization;

recommendation owner, timing, objective, and feasibility notes;

downloadable PDF decision reports.

Recommendations are advisory and require confirmation by qualified operational personnel.

Project Structure

app/                    Streamlit application
src/analysis/           Rotation construction
src/features/           Feature engineering
src/models/             Model training and contracts
src/evaluation/         Flight, graph, policy, and error evaluation
src/graph/              Graph scoring and multi-hop tracing
src/decision_support/   Priority and recommendation logic
src/explainability/     SHAP explanations
src/reporting/          PDF report generation
src/visualization/      Maps and scientific figures
tests/                  Automated tests

Installation

git clone <repository-url>
cd AeroRecover-AI
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

Run the dashboard:

python -m streamlit run app\streamlit_app.py

Run the tests:

python -m pytest -q

Limitations

The final model does not use live crew, gate, maintenance, passenger-connection, or airport-resource data.

Weather is not included in the final model.

The replay interface visualizes historical model output, not recorded trajectories or live forecasts.

Probabilities and SHAP values describe predictive associations, not causal effects.

Recommendations have not yet been validated in a live operations-control trial.

Future Work

Future development may include airport congestion and weather context, multi-flight history, resource-constrained recovery optimization, richer graph-learning methods, and validation with airline or airport operational stakeholders.

Responsible Use

AeroRecover AI is intended for research and human-in-the-loop decision support. It must not replace qualified decisions involving safety, dispatch, crew legality, maintenance, regulation, or network recovery.
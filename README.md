# AeroRecover AI

### Explainable Flight Delay Propagation Decision Support

AeroRecover AI is a research prototype that predicts whether an inbound aircraft delay will propagate to its next flight, traces that risk through aircraft-rotation graphs, and generates explainable operational recommendations.

> Historical decision-support prototype—not a live airline operations system.

## What It Does

```text
BTS Flight Data
      ↓
Aircraft Rotation Dataset
      ↓
XGBoost Propagation Prediction
      ↓
SHAP Explanation
      ↓
Multi-hop Rotation Graph
      ↓
Operational Priority and Recommendations
      ↓
Streamlit Replay and PDF Report
```

The system combines flight-level prediction, graph-based domino-chain analysis, explainability, and operational decision support in a single pipeline.

## Dataset

The project uses the 2023 U.S. Bureau of Transportation Statistics On-Time Performance dataset.

| Item               |     Value |
| ------------------ | --------: |
| Raw flight records | 6,847,899 |
| Valid flights      | 6,743,404 |
| Rotation samples   | 4,796,459 |
| Monthly files      |        12 |

### Chronological split

| Split       | Period            |   Samples |
| ----------- | ----------------- | --------: |
| Train       | January–August    | 3,159,311 |
| Validation  | September–October |   832,022 |
| Locked test | November–December |   805,126 |

Random splitting was not used.

## Prediction Task

The model predicts:

```text
IS_DELAY_PROPAGATED = 1
```

when the target flight has at least 15 minutes of BTS-reported late-aircraft delay.

Each probability applies to one consecutive aircraft-rotation edge. It is not a causal estimate or a calibrated probability for an entire multi-flight chain.

## Model Performance

### Validation baseline comparison

| Model                                | Precision | Recall |     F1 | ROC-AUC | PR-AUC |
| ------------------------------------ | --------: | -----: | -----: | ------: | -----: |
| Logistic Regression                  |    0.8318 | 0.7425 | 0.7846 |  0.9813 | 0.8347 |
| XGBoost — threshold 0.50             |    0.8062 | 0.8208 | 0.8134 |  0.9880 | 0.8798 |
| XGBoost — operational threshold 0.46 |    0.7907 | 0.8422 | 0.8157 |  0.9880 | 0.8798 |

### Locked final test

| Metric      |  Result |
| ----------- | ------: |
| Samples     | 805,126 |
| Precision   |  0.7503 |
| Recall      |  0.8422 |
| F1          |  0.7936 |
| ROC-AUC     |  0.9882 |
| PR-AUC      |  0.8543 |
| Brier score |  0.0189 |

The model, features, and `0.46` operational threshold were fixed before the November–December test period was evaluated.

## Graph Propagation Analysis

Flights are nodes, while consecutive flights operated by the same aircraft form directed edges.

| Graph result               |     Value |
| -------------------------- | --------: |
| Flight nodes               | 6,743,404 |
| Physical tail edges        | 4,975,883 |
| Propagation-eligible edges | 4,823,761 |
| Scored validation edges    |   832,022 |
| Edge-score match rate      |    99.50% |

### Multi-hop evaluation

| Metric                  | Result |
| ----------------------- | -----: |
| Matched chain starts    | 11,039 |
| Precision               | 0.7124 |
| Recall                  | 0.7471 |
| F1                      | 0.7293 |
| Exact chain-length rate | 0.9019 |
| Edge-count MAE          | 0.1105 |

## Explainable Decision Support

For each selected historical flight, AeroRecover AI produces:

* Propagation probability
* Likelihood, impact, and urgency assessment
* `P1–P4` operational priority
* Local SHAP explanation
* Predicted downstream chain
* Human-reviewable recommendations
* Responsible operational unit and timing
* Downloadable PDF report

Recommendations are advisory outputs, not automatic operational commands.

## Historical Replay Dashboard

The Streamlit interface provides:

* Historical flight and route filters
* Geographic propagation map
* Animated multi-hop replay
* Operational assessment cards
* Local SHAP analysis
* Structured recommendations
* PDF report generation

Map animations represent model-predicted graph progression, not recorded aircraft trajectories.

## Project Structure

```text
app/                     Streamlit application
models/                  Saved XGBoost model
results/                 Evaluation outputs
src/analysis/            Rotation dataset construction
src/models/              Model training
src/evaluation/          Model, graph and DSS evaluation
src/graph/               Flight graph and multi-hop analysis
src/decision_support/    Priority and recommendation logic
src/explainability/      Local SHAP analysis
src/reporting/           PDF report generation
src/visualization/       Propagation-map utilities
```

## Installation

```powershell
git clone <repository-url>
cd AeroRecover-AI

python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Run the dashboard:

```powershell
python -m streamlit run app\streamlit_app.py
```

## Limitations

* The current version uses BTS operational data only.
* Weather, crew, gate, maintenance, and live resource data are not included.
* Historical replay is not a live forecasting environment.
* Recommendations have not been validated in a real airline operational trial.
* Predictive associations and SHAP values must not be interpreted as causal effects.

## Future Work

Future versions may investigate weather and congestion features, multi-flight history, graph neural networks, live operational integration, and resource-constrained recovery optimization.

## Responsible Use

AeroRecover AI is intended for research and decision-support demonstration. Real operational decisions require qualified human review and current safety, fleet, crew, airport, maintenance, and regulatory information.

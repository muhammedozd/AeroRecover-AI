"""Integration contracts for the promoted full-enhanced rotation model."""

import hashlib
from pathlib import Path
import unittest

import pandas as pd

from src.decision_support.assessment_service import build_decision_report
from src.decision_support.contracts import FlightDecisionInput
from src.features.rotation_features import build_rotation_model_features
from src.models.rotation_model_contract import (
    FEATURE_COLUMNS, MODEL_THRESHOLD, MODEL_VERSION, load_model_pipeline,
    validate_feature_contract,
)
from src.reporting.decision_report_pdf import build_decision_report_pdf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_MODEL = PROJECT_ROOT / "models" / "xgboost_propagation_2023_time_split.pkl"
LOCKED_METRICS = PROJECT_ROOT / "results" / "full_enhanced_final_test_metrics.json"
SCORED_EDGES = PROJECT_ROOT / "data" / "processed" / "graph" / "scored_tail_edges_2023_validation_full_enhanced.parquet"


def file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def sample_raw_row():
    return pd.DataFrame([{
        "FL_DATE": "2023-10-01", "CRS_DEP_TIME": 1430, "PREV_DEST": "ATL",
        "PREV_DELAY_LEVEL": "Minor", "DEST": "ORD", "OP_UNIQUE_CARRIER": "DL",
        "ROTATION_POSITION": 2, "PREV_ARR_DELAY": 20.0, "PREV_ARR_MIN": 800.0,
        "PREV_CRS_ARR_MIN": 780.0, "PLANNED_TURNAROUND": 50.0,
        "TURN_BUFFER": 30.0, "PREV_DELAY_RATIO": 0.4, "HAS_BUFFER": 1,
        "IS_SHORT_TURN": 0, "PREV_DELAYED": 1, "DISTANCE": 600.0,
    }])


class FullEnhancedIntegrationTests(unittest.TestCase):
    def test_model_contract_and_probability(self):
        self.assertEqual(MODEL_THRESHOLD, 0.47)
        self.assertEqual(MODEL_VERSION, "xgboost_2023_full_enhanced")
        self.assertEqual(len(FEATURE_COLUMNS), 24)
        self.assertEqual(len(set(FEATURE_COLUMNS)), 24)
        validate_feature_contract(FEATURE_COLUMNS)
        with self.assertRaises(ValueError):
            validate_feature_contract(FEATURE_COLUMNS[:-1])
        with self.assertRaises(ValueError):
            validate_feature_contract([*FEATURE_COLUMNS, "DEP_DELAY"])
        features, _ = build_rotation_model_features(sample_raw_row())
        probability = float(load_model_pipeline().predict_proba(features)[0, 1])
        self.assertTrue(0.0 <= probability <= 1.0)

    def test_flight_decision_input_and_pdf(self):
        decision_input = FlightDecisionInput(
            propagation_probability=0.82, previous_arrival_delay=30.0,
            turn_buffer=15.0, previous_delay_ratio=0.6,
            planned_turnaround=50.0, downstream_edge_count=2,
        )
        report = build_decision_report(decision_input)
        chain = pd.DataFrame([{
            "SOURCE_FLIGHT_ID": "SOURCE", "TARGET_FLIGHT_ID": "TARGET",
            "CONNECTION_AIRPORT": "ATL", "PROPAGATION_PROBABILITY": 0.82,
            "PLANNED_CONNECTION_MINUTES": 50.0,
        }])
        pdf = build_decision_report_pdf(
            flight_id="SOURCE", decision_input=decision_input, decision_report=report,
            predicted_chain=chain, cumulative_chain_score=0.82,
        )
        self.assertTrue(pdf.startswith(b"%PDF"))
        with self.assertRaises(ValueError):
            FlightDecisionInput(
                propagation_probability=1.1, previous_arrival_delay=0.0,
                turn_buffer=10.0, previous_delay_ratio=0.0,
                planned_turnaround=40.0, downstream_edge_count=0,
            )

    def test_graph_ids_and_protected_artifacts(self):
        edges = pd.read_parquet(SCORED_EDGES, columns=["SOURCE_FLIGHT_ID", "TARGET_FLIGHT_ID"])
        self.assertFalse(edges["SOURCE_FLIGHT_ID"].duplicated().any())
        self.assertFalse(edges["TARGET_FLIGHT_ID"].duplicated().any())
        self.assertFalse(edges.duplicated(["SOURCE_FLIGHT_ID", "TARGET_FLIGHT_ID"]).any())
        self.assertEqual(file_sha256(BASELINE_MODEL), "ED5102B63C1B0F469C0D6F45A909AF7A8F1197EEED43AA1529E69802CC6B0290")
        self.assertEqual(file_sha256(LOCKED_METRICS), "6DD9E4D8CC803B8C566B127175ACB69D6B59A15D2AD5C53B8FF20A342AAE29E8")


if __name__ == "__main__":
    unittest.main()

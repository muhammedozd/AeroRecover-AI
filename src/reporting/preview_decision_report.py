"""Generate a local preview of the decision report PDF."""

from pathlib import Path

import pandas as pd

from src.decision_support.assessment_service import (
    build_decision_report,
)
from src.decision_support.contracts import (
    FlightDecisionInput,
)
from src.reporting.decision_report_pdf import (
    build_decision_report_pdf,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_PATH = (
    PROJECT_ROOT
    / "output"
    / "pdf"
    / "aerorecover_decision_report_preview.pdf"
)


def main() -> None:
    decision_input = FlightDecisionInput(
        propagation_probability=0.82,
        previous_arrival_delay=40.0,
        turn_buffer=12.0,
        previous_delay_ratio=0.73,
        planned_turnaround=55.0,
        downstream_edge_count=3,
    )

    decision_report = build_decision_report(
        decision_input
    )

    pdf_bytes = build_decision_report_pdf(
        flight_id=(
            "SYNTHETIC_PREVIEW_FLIGHT"
        ),
        decision_input=decision_input,
        decision_report=decision_report,
        predicted_chain=pd.DataFrame([
            {
                "SOURCE_FLIGHT_ID": "SYNTHETIC_SOURCE_FLIGHT",
                "TARGET_FLIGHT_ID": "SYNTHETIC_TARGET_FLIGHT",
                "CONNECTION_AIRPORT": "N/A",
                "PROPAGATION_PROBABILITY": 0.82,
                "PLANNED_CONNECTION_MINUTES": 35.0,
            }
        ]),
        cumulative_chain_score=0.82,
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_bytes(
        pdf_bytes
    )

    print(
        "PDF preview saved:",
        OUTPUT_PATH,
    )

    print(
        "PDF size:",
        f"{len(pdf_bytes):,} bytes",
    )


if __name__ == "__main__":
    main()

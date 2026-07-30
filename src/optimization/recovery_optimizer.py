"""Flight recovery optimization utilities."""

from typing import Any, Literal
from src.analysis.threshold_analysis import (
    calculate_operational_risk_score,
)

from src.decision_support.recommendation_engine import (
    generate_recommendations,
)



RiskLevel = Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]


def determine_risk_level(
    operational_risk_score: int,
) -> RiskLevel:
    if not 0 <= operational_risk_score <= 11:
        raise ValueError(
            "Operational risk score must be between 0 and 11."
        )

    if operational_risk_score <= 2:
        return "LOW"

    if operational_risk_score <= 5:
        return "MEDIUM"

    if operational_risk_score <= 8:
        return "HIGH"

    return "CRITICAL"

def identify_risk_factors(
    flight_data: dict[str, Any],
) -> list[str]:
    risk_factors: list[str] = []

    previous_arrival_delay = flight_data.get("PREV_ARR_DELAY")
    turn_buffer = flight_data.get("TURN_BUFFER")
    previous_delay_ratio = flight_data.get("PREV_DELAY_RATIO")
    planned_turnaround = flight_data.get("PLANNED_TURNAROUND")

    if previous_arrival_delay is not None:
        if previous_arrival_delay >= 60:
            risk_factors.append(
                "Previous arrival delay is critically high."
            )
        elif previous_arrival_delay >= 30:
            risk_factors.append(
                "Previous arrival delay is high."
            )
        elif previous_arrival_delay >= 15:
            risk_factors.append(
                "Previous arrival delay is moderately high."
            )

    if turn_buffer is not None:
        if turn_buffer < 0:
            risk_factors.append(
                "Turnaround buffer is critically insufficient."
            )
        elif turn_buffer < 10:
            risk_factors.append(
                "Turnaround buffer is severely insufficient."
            )
        elif turn_buffer < 20:
            risk_factors.append(
                "Turnaround buffer is insufficient."
            )
        elif turn_buffer < 30:
            risk_factors.append(
                "Turnaround buffer is limited."
            )

    if previous_delay_ratio is not None:
        if previous_delay_ratio >= 0.80:
            risk_factors.append(
                "Previous delay ratio is critically high."
            )
        elif previous_delay_ratio >= 0.60:
            risk_factors.append(
                "Previous delay ratio is very high."
            )
        elif previous_delay_ratio >= 0.40:
            risk_factors.append(
                "Previous delay ratio is high."
            )
        elif previous_delay_ratio >= 0.20:
            risk_factors.append(
                "Previous delay ratio is moderately high."
            )

    if planned_turnaround is not None:
        if planned_turnaround < 30:
            risk_factors.append(
                "Planned turnaround time is critically short."
            )
        elif planned_turnaround < 60:
            risk_factors.append(
                "Planned turnaround time is short."
            )
        elif planned_turnaround >= 180:
            risk_factors.append(
                "Planned turnaround time is unusually long."
            )

    return risk_factors



def generate_flight_risk_report(
    flight_data: dict[str, Any],
) -> None:

    score = calculate_operational_risk_score(
        flight_data
    )

    risk_level = determine_risk_level(
        score
    )

    risk_factors = identify_risk_factors(
        flight_data
    )

    recommendations = generate_recommendations(
    flight_data
)
    

    print("=" * 60)
    print("FLIGHT RISK ASSESSMENT")
    print("=" * 60)

    print(f"Operational Risk Score : {score} / 11")
    print(f"Risk Level             : {risk_level}")

    print("\nRisk Factors")
    print("-" * 30)

    for factor in risk_factors:
        print(f"- {factor}")

    print("\nRecommendations")
    print("-" * 30)

    for recommendation in recommendations:
        print(f"- Action   : {recommendation['action']}")
        print(f"  Reason   : {recommendation['reason']}")
        print(f"  Priority : {recommendation['priority']}")
        print()


def main():

   sample_flight = {
    "PREV_ARR_DELAY": 80,
    "TURN_BUFFER": 5,
    "PREV_DELAY_RATIO": 0.75,
    "PLANNED_TURNAROUND": 20,
}
   generate_flight_risk_report(
    sample_flight
)
   

if __name__ == "__main__":
    main()
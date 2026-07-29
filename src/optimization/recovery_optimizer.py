"""Flight recovery optimization utilities."""

from typing import Any, Literal


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

    if (
        previous_arrival_delay is not None
        and previous_arrival_delay >= 30
    ):
        risk_factors.append(
            "Previous arrival delay is high."
        )

    if (
        turn_buffer is not None
        and turn_buffer < 20
    ):
        risk_factors.append(
            "Turnaround buffer is insufficient."
        )

    if (
        previous_delay_ratio is not None
        and previous_delay_ratio >= 0.40
    ):
        risk_factors.append(
            "Previous delay ratio is high."
        )

    if (
        planned_turnaround is not None
        and planned_turnaround < 30
    ):
        risk_factors.append(
            "Planned turnaround time is critically short."
        )

    return risk_factors


sample_flight = {
    "PREV_ARR_DELAY": 45,
    "TURN_BUFFER": 8,
    "PREV_DELAY_RATIO": 0.65,
    "PLANNED_TURNAROUND": 25,
}

risk_factors = identify_risk_factors(sample_flight)

for factor in risk_factors:
    print(factor)
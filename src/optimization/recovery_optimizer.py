"""Flight recovery optimization utilities."""


from typing import Any, Literal

#Bu fonsksiyonum gelen olasılık değerlerine göre risk seviyesini belirler.

RiskLevel = Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]



def determine_risk_level(probability: float) -> RiskLevel:
    if not 0.0 <= probability <= 1.0:
        raise ValueError("Probability must be between 0.0 and 1.0.")

    if probability < 0.30:
        return "LOW"

    if probability < 0.60:
        return "MEDIUM"

    if probability < 0.80:
        return "HIGH"

    return "CRITICAL"

def identify_risk_factors(flight_data: dict[str, Any]) -> list[str]:
    risk_factors = []

    previous_arrival_delay = flight_data.get("PREV_ARR_DELAY", 0)
    turn_buffer = flight_data.get("TURN_BUFFER", 0)
    previous_delay_ratio = flight_data.get("PREV_DELAY_RATIO", 0.0)


    if previous_arrival_delay >= 30:
        risk_factors.append("Previous arrival delay is high.")

    if turn_buffer < 10:
        risk_factors.append("Turnaround buffer is insufficient.")

    if previous_delay_ratio >= 0.5:
        risk_factors.append("Previous delay ratio is high.")


    return risk_factors
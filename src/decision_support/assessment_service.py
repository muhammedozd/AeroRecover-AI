"""Flight decision assessment service."""

from src.decision_support.contracts import (
    FlightDecisionAssessment,
    FlightDecisionInput,
)
from src.decision_support.risk_classification import (
    determine_impact,
    determine_likelihood,
    determine_priority,
    determine_urgency,
)


def assess_flight(
    flight: FlightDecisionInput,
) -> FlightDecisionAssessment:
    
    likelihood = determine_likelihood(
        flight.propagation_probability
    )

    impact = determine_impact(
        flight.downstream_edge_count
    )

    urgency = determine_urgency(
        flight.turn_buffer
    )

    priority = determine_priority(
        likelihood=likelihood,
        impact=impact,
        urgency=urgency,
    )

    return FlightDecisionAssessment(
        likelihood=likelihood,
        impact=impact,
        urgency=urgency,
        priority=priority,
    )
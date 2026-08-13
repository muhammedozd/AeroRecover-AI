"""Operational recommendation engine."""


from src.decision_support.contracts import (
    FlightDecisionAssessment,
    FlightDecisionInput,
    ImpactLevel,
    PriorityLevel,
    Recommendation,
    UrgencyLevel,
)


def generate_recommendations(
    flight: FlightDecisionInput,
    assessment: FlightDecisionAssessment,
) -> list[Recommendation]:
    recommendations: list[Recommendation] = []

    if assessment.priority == PriorityLevel.P1_CRITICAL:
        recommendations.append(
            {
                "action": (
                    "Initiate an immediate operational "
                    "review of this aircraft rotation."
                ),
                "reason": (
                    "The combined likelihood, impact, and "
                    "urgency assessment is critical."
                ),
                "priority": PriorityLevel.P1_CRITICAL,
            }
        )

    elif assessment.priority == PriorityLevel.P2_HIGH:
        recommendations.append(
            {
                "action": (
                    "Prioritize this rotation for active "
                    "monitoring and contingency preparation."
                ),
                "reason": (
                    "The combined operational assessment "
                    "indicates high priority."
                ),
                "priority": PriorityLevel.P2_HIGH,
            }
        )

    if assessment.urgency == UrgencyLevel.IMMEDIATE:
        recommendations.append(
            {
                "action": (
                    "Coordinate turnaround teams before "
                    "the inbound aircraft arrives."
                ),
                "reason": (
                    "The available turnaround buffer is "
                    "less than 10 minutes."
                ),
                "priority": assessment.priority,
            }
        )

    elif assessment.urgency == UrgencyLevel.URGENT:
        recommendations.append(
            {
                "action": (
                    "Prepare ground handling resources for "
                    "a reduced turnaround window."
                ),
                "reason": (
                    "The available turnaround buffer is "
                    "between 10 and 19 minutes."
                ),
                "priority": assessment.priority,
            }
        )

    elif assessment.urgency == UrgencyLevel.WATCH:
        recommendations.append(
            {
                "action": (
                    "Monitor turnaround task completion "
                    "times closely."
                ),
                "reason": (
                    "The available turnaround buffer is "
                    "between 20 and 29 minutes."
                ),
                "priority": assessment.priority,
            }
        )

    if assessment.impact == ImpactLevel.NETWORK:
        recommendations.append(
            {
                "action": (
                    "Review downstream rotations that may "
                    "be exposed to propagation."
                ),
                "reason": (
                    "The predicted propagation path extends "
                    "across at least four graph edges."
                ),
                "priority": assessment.priority,
            }
        )

    elif assessment.impact == ImpactLevel.MULTI_HOP:
        recommendations.append(
            {
                "action": (
                    "Monitor the connected downstream "
                    "flight sequence."
                ),
                "reason": (
                    "The predicted propagation path covers "
                    "multiple graph edges."
                ),
                "priority": assessment.priority,
            }
        )

    if flight.previous_arrival_delay >= 60:
        recommendations.append(
            {
                "action": (
                    "Closely track the inbound aircraft and "
                    "its remaining turnaround window."
                ),
                "reason": (
                    "The previous flight arrived at least "
                    "60 minutes late."
                ),
                "priority": assessment.priority,
            }
        )

    elif flight.previous_arrival_delay >= 30:
        recommendations.append(
            {
                "action": (
                    "Prepare for a compressed turnaround "
                    "window."
                ),
                "reason": (
                    "The previous flight arrived between "
                    "30 and 59 minutes late."
                ),
                "priority": assessment.priority,
            }
        )

    if flight.previous_delay_ratio >= 0.60:
        recommendations.append(
            {
                "action": (
                    "Review whether the planned turnaround "
                    "can absorb the inbound delay."
                ),
                "reason": (
                    "The positive previous arrival delay "
                    "uses at least 60% of the planned "
                    "turnaround time."
                ),
                "priority": assessment.priority,
            }
        )

    if flight.planned_turnaround < 30:
        recommendations.append(
            {
                "action": (
                    "Prepare turnaround resources before "
                    "aircraft arrival."
                ),
                "reason": (
                    "The planned turnaround time is less "
                    "than 30 minutes."
                ),
                "priority": assessment.priority,
            }
        )

    if not recommendations:
        recommendations.append(
            {
                "action": (
                    "Continue routine operational "
                    "monitoring."
                ),
                "reason": (
                    "No elevated intervention condition "
                    "was identified."
                ),
                "priority": PriorityLevel.P4_NORMAL,
            }
        )

    return recommendations
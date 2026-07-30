"""Operational recommendation engine."""

from typing import Any, Literal, TypedDict


RecommendationPriority = Literal[
    "LOW",
    "MEDIUM",
    "HIGH",
    "CRITICAL",
]

class Recommendation(TypedDict):
    action: str
    reason: str
    priority: RecommendationPriority


def generate_recommendations(
    flight_data: dict[str, Any],
) -> list[Recommendation]:
    recommendations: list[Recommendation] = []

    previous_arrival_delay = flight_data.get(
        "PREV_ARR_DELAY"
    )

    if previous_arrival_delay is not None:
     if previous_arrival_delay >= 60:
            recommendations.append(

                 {
                    "action": (
                        "Prioritize this aircraft rotation "
                        "in operational monitoring."
                    ),
                    "reason": (
                        "The previous flight arrived at least "
                        "60 minutes late."
                    ),
                    "priority": "CRITICAL",
                }
                
            )


     elif previous_arrival_delay >= 30:
            recommendations.append(
                {
                    "action": (
                        "Prepare the turnaround team for a "
                        "reduced operational window."
                    ),
                    "reason": (
                        "The previous flight arrived between "
                        "30 and 59 minutes late."
                    ),
                    "priority": "HIGH",
                }
            )

     if previous_arrival_delay >= 15:
            recommendations.append(
                {
                    "action": (
                        "Closely monitor the inbound aircraft."
                    ),
                    "reason": (
                        "The previous flight arrived between "
                        "15 and 29 minutes late."
                    ),
                    "priority": "MEDIUM",
                }
            )


        turn_buffer = flight_data.get(
        "TURN_BUFFER"
    )

    if turn_buffer is not None:

        if turn_buffer < 10:
            recommendations.append(
                {
                    "action": (
                        "Prioritize turnaround activities and "
                        "prepare ground resources in advance."
                    ),
                    "reason": (
                        "The turnaround buffer is less than "
                        "10 minutes."
                    ),
                    "priority": "CRITICAL",
                }
            )

        elif turn_buffer < 20:
            recommendations.append(
                {
                    "action": (
                        "Coordinate ground handling teams "
                        "before aircraft arrival."
                    ),
                    "reason": (
                        "The turnaround buffer is between "
                        "10 and 19 minutes."
                    ),
                    "priority": "HIGH",
                }
            )

        elif turn_buffer < 30:
            recommendations.append(
                {
                    "action": (
                        "Monitor turnaround progress closely."
                    ),
                    "reason": (
                        "The turnaround buffer is between "
                        "20 and 29 minutes."
                    ),
                    "priority": "MEDIUM",
                }
            )

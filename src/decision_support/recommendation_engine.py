"""Operational recommendation engine."""


from src.decision_support.contracts import (
    FlightDecisionAssessment,
    FlightDecisionInput,
    ImpactLevel,
    PriorityLevel,
    Recommendation,
    UrgencyLevel,
)

def create_recommendation(
    *,
    action_code: str,
    action: str,
    reason: str,
    priority: PriorityLevel,
    owner: str,
    timing: str,
    objective: str,
    feasibility_note: str,
) -> Recommendation:

    return {
        "action_code": action_code,
        "action": action,
        "reason": reason,
        "priority": priority,
        "owner": owner,
        "timing": timing,
        "objective": objective,
        "feasibility_note": feasibility_note,
    }


def generate_recommendations(
    flight: FlightDecisionInput,
    assessment: FlightDecisionAssessment,
) -> list[Recommendation]:
    recommendations: list[Recommendation] = []

    if assessment.priority == PriorityLevel.P1_CRITICAL:
        recommendations.append(
            create_recommendation(
            action_code="ROTATION_CRITICAL_REVIEW",
            action=(
                "Initiate an immediate operational "
                "review of this aircraft rotation."
            ),
            reason=(
                "The combined likelihood, impact, and "
                "urgency assessment is critical."
            ),
            priority=PriorityLevel.P1_CRITICAL,
            owner="Operations Control",
            timing="Immediately",
            objective=(
                "Escalate the rotation for immediate "
                "human review."
            ),
            feasibility_note=(
                "The specific recovery action must be "
                "confirmed by an operations controller."
            ),
        )
    )

    elif assessment.priority == PriorityLevel.P2_HIGH:
        recommendations.append(
            create_recommendation(
            action_code="ROTATION_ACTIVE_MONITORING",
            action=(
                "Prioritize this rotation for active "
                "monitoring and contingency preparation."
            ),
            reason=(
                "The combined operational assessment "
                "indicates high priority."
            ),
            priority=PriorityLevel.P2_HIGH,
            owner="Operations Control",
            timing="Before the next departure",
            objective=(
                "Prepare the control team for possible "
                "delay propagation."
            ),
            feasibility_note=(
                "Contingency options depend on current "
                "fleet and station constraints."
            ),
        )
    )
    elif assessment.priority == PriorityLevel.P3_MONITOR:
        recommendations.append(
        create_recommendation(
            action_code="ROTATION_WATCHLIST",
            action=(
                "Place this rotation on the operational "
                "monitoring watchlist."
            ),
            reason=(
                "The combined assessment indicates a "
                "moderate level of operational attention."
            ),
            priority=PriorityLevel.P3_MONITOR,
            owner="Operations Control",
            timing="During routine monitoring cycles",
            objective=(
                "Maintain visibility without immediate "
                "operational escalation."
            ),
            feasibility_note=(
                "Monitoring frequency depends on current "
                "operations control workload."
            ),
        )
    )
    if assessment.urgency == UrgencyLevel.IMMEDIATE:
        recommendations.append(
        create_recommendation(
            action_code="TURNAROUND_TEAM_PRE_ALERT",
            action=(
                "Coordinate turnaround teams before "
                "the inbound aircraft arrives."
            ),
            reason=(
                "The available turnaround buffer is "
                "less than 10 minutes."
            ),
            priority=assessment.priority,
            owner="Station Operations",
            timing="Before inbound aircraft arrival",
            objective=(
                "Reduce turnaround preparation latency."
            ),
            feasibility_note=(
                "Requires confirmation of local team "
                "and equipment availability."
            ),
        )
    )

    elif assessment.urgency == UrgencyLevel.URGENT:
        recommendations.append(
        create_recommendation(
            action_code="GROUND_RESOURCE_PREPARATION",
            action=(
                "Prepare ground handling resources for "
                "a reduced turnaround window."
            ),
            reason=(
                "The available turnaround buffer is "
                "between 10 and 19 minutes."
            ),
            priority=assessment.priority,
            owner="Station Operations",
            timing="Before inbound aircraft arrival",
            objective=(
                "Improve readiness for a compressed "
                "turnaround window."
            ),
            feasibility_note=(
                "Resource assignments must be checked "
                "against current station workload."
            ),
        )
    )

    elif assessment.urgency == UrgencyLevel.WATCH:
        recommendations.append(
        create_recommendation(
            action_code="TURNAROUND_PROGRESS_MONITORING",
            action=(
                "Monitor turnaround task completion "
                "times closely."
            ),
            reason=(
                "The available turnaround buffer is "
                "between 20 and 29 minutes."
            ),
            priority=assessment.priority,
            owner="Turnaround Coordination",
            timing="During turnaround execution",
            objective=(
                "Detect emerging turnaround slippage "
                "before departure."
            ),
            feasibility_note=(
                "Requires access to current turnaround "
                "task status."
            ),
        )
    )

    if assessment.impact == ImpactLevel.NETWORK:
        recommendations.append(
        create_recommendation(
            action_code="NETWORK_ROTATION_REVIEW",
            action=(
                "Review downstream rotations that may "
                "be exposed to propagation."
            ),
            reason=(
                "The predicted propagation path extends "
                "across at least four graph edges."
            ),
            priority=assessment.priority,
            owner="Network Operations Control",
            timing="Before downstream departures",
            objective=(
                "Increase awareness of network-level "
                "propagation exposure."
            ),
            feasibility_note=(
                "Downstream actions require confirmation "
                "of current schedules and constraints."
            ),
        )
    )

    elif assessment.impact == ImpactLevel.MULTI_HOP:
        recommendations.append(
        create_recommendation(
            action_code="DOWNSTREAM_SEQUENCE_MONITORING",
            action=(
                "Monitor the connected downstream "
                "flight sequence."
            ),
            reason=(
                "The predicted propagation path covers "
                "multiple graph edges."
            ),
            priority=assessment.priority,
            owner="Network Operations Control",
            timing="During the connected rotation sequence",
            objective=(
                "Track whether risk is progressing through "
                "the predicted flight chain."
            ),
            feasibility_note=(
                "The predicted chain represents model "
                "exposure, not confirmed future disruption."
            ),
        )
    )
    if flight.previous_arrival_delay >= 60:
        recommendations.append(
        create_recommendation(
            action_code="INBOUND_DELAY_TRACKING",
            action=(
                "Closely track the inbound aircraft and "
                "its remaining turnaround window."
            ),
            reason=(
                "The previous flight arrived at least "
                "60 minutes late."
            ),
            priority=assessment.priority,
            owner="Operations Control",
            timing="Until completion of the next departure",
            objective=(
                "Maintain awareness of severe inbound "
                "delay exposure."
            ),
            feasibility_note=(
                "Updated arrival and turnaround status "
                "should be confirmed operationally."
            ),
        )
    )

    elif flight.previous_arrival_delay >= 30:
        recommendations.append(
        create_recommendation(
            action_code="COMPRESSED_TURNAROUND_PREPARATION",
            action=(
                "Prepare for a compressed turnaround "
                "window."
            ),
            reason=(
                "The previous flight arrived between "
                "30 and 59 minutes late."
            ),
            priority=assessment.priority,
            owner="Station Operations",
            timing="Before turnaround begins",
            objective=(
                "Increase readiness for reduced available "
                "turnaround time."
            ),
            feasibility_note=(
                "Actual preparation options depend on "
                "station resources and aircraft status."
            ),
        )
    )

    if flight.previous_delay_ratio >= 0.60:
        recommendations.append(
        create_recommendation(
            action_code="TURNAROUND_ABSORPTION_REVIEW",
            action=(
                "Review whether the planned turnaround "
                "can absorb the inbound delay."
            ),
            reason=(
                "The positive previous arrival delay "
                "uses at least 60% of the planned "
                "turnaround time."
            ),
            priority=assessment.priority,
            owner="Operations Control",
            timing="Before the next departure",
            objective=(
                "Assess whether the remaining schedule "
                "margin is operationally sufficient."
            ),
            feasibility_note=(
                "The assessment does not include live "
                "gate, crew, or ground-resource status."
            ),
        )
    )

    if flight.planned_turnaround < 30:
        recommendations.append(
        create_recommendation(
            action_code="SHORT_TURN_RESOURCE_READINESS",
            action=(
                "Prepare turnaround resources before "
                "aircraft arrival."
            ),
            reason=(
                "The planned turnaround time is less "
                "than 30 minutes."
            ),
            priority=assessment.priority,
            owner="Station Operations",
            timing="Before inbound aircraft arrival",
            objective=(
                "Increase readiness for a short planned "
                "turnaround."
            ),
            feasibility_note=(
                "Resource allocation requires confirmation "
                "of local availability and safety rules."
            ),
        )
    )

    if not recommendations:
        recommendations.append(
        create_recommendation(
            action_code="ROUTINE_OPERATIONAL_MONITORING",
            action=(
                "Continue routine operational monitoring."
            ),
            reason=(
                "No elevated intervention condition "
                "was identified."
            ),
            priority=assessment.priority,
            owner="Operations Control",
            timing="During routine monitoring cycles",
            objective=(
                "Maintain standard situational awareness."
            ),
            feasibility_note=(
                "Routine status does not guarantee that "
                "delay propagation will not occur."
            ),
        )
    )

    return recommendations
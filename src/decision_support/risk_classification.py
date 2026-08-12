"""Risk dimension classification for decision support."""

from src.decision_support.contracts import (
    UrgencyLevel,
    LikelihoodLevel,
    ImpactLevel,
    PriorityLevel,

)

def determine_urgency(
    turn_buffer: float,
) -> UrgencyLevel:
    if turn_buffer < 10:
        return UrgencyLevel.IMMEDIATE

    if turn_buffer < 20:
        return UrgencyLevel.URGENT

    if turn_buffer < 30:
        return UrgencyLevel.WATCH

    return UrgencyLevel.ROUTINE

def determine_likelihood(
    propagation_probability: float,
) -> LikelihoodLevel:
    if not 0.0 <= propagation_probability <= 1.0:
        raise ValueError(
            "Propagation probability must be between 0 and 1."
        )

    if propagation_probability < 0.20:
        return LikelihoodLevel.LOW

    if propagation_probability < 0.46:
        return LikelihoodLevel.MODERATE

    if propagation_probability < 0.80:
        return LikelihoodLevel.HIGH

    return LikelihoodLevel.VERY_HIGH

def determine_impact(
    downstream_edge_count: int,
) -> ImpactLevel:
    if downstream_edge_count < 0:
        raise ValueError(
            "Downstream edge count must be non-negative."
        )
    
    if downstream_edge_count <= 1:
        return ImpactLevel.LOCAL

    if downstream_edge_count <= 3:
        return ImpactLevel.MULTI_HOP

    return ImpactLevel.NETWORK

def determine_priority(
    likelihood: LikelihoodLevel,
    impact: ImpactLevel,
    urgency: UrgencyLevel,
) -> PriorityLevel:
    if (
        likelihood == LikelihoodLevel.VERY_HIGH
        and (
            impact == ImpactLevel.NETWORK
            or urgency == UrgencyLevel.IMMEDIATE
        )
    ):
        return PriorityLevel.P1_CRITICAL

    if (
        likelihood == LikelihoodLevel.VERY_HIGH
        or (
            likelihood == LikelihoodLevel.HIGH
            and (
                impact in {
                    ImpactLevel.MULTI_HOP,
                    ImpactLevel.NETWORK,
                }
                or urgency in {
                    UrgencyLevel.URGENT,
                    UrgencyLevel.IMMEDIATE,
                }
            )
        )
    ):
        return PriorityLevel.P2_HIGH

    if (
        likelihood != LikelihoodLevel.LOW
        or impact != ImpactLevel.LOCAL
        or urgency != UrgencyLevel.ROUTINE
    ):
        return PriorityLevel.P3_MONITOR

    return PriorityLevel.P4_NORMAL

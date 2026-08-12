"""Data contracts for operational decision support."""

from dataclasses import dataclass
from math import isfinite

from enum import Enum, IntEnum

class LikelihoodLevel(str, Enum):
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    VERY_HIGH = "VERY_HIGH"

class ImpactLevel(str, Enum):
    LOCAL = "LOCAL"
    MULTI_HOP = "MULTI_HOP"
    NETWORK = "NETWORK"

class UrgencyLevel(str, Enum):
    ROUTINE = "ROUTINE"
    WATCH = "WATCH"
    URGENT = "URGENT"
    IMMEDIATE = "IMMEDIATE"

class PriorityLevel(IntEnum):
    P1_CRITICAL = 1
    P2_HIGH = 2
    P3_MONITOR = 3
    P4_NORMAL = 4


#frozen=true değerlerin sonradan değişmesini engeller.

@dataclass(frozen=True)
class FlightDecisionInput:
    propagation_probability: float
    previous_arrival_delay: float
    turn_buffer: float
    previous_delay_ratio: float
    planned_turnaround: float
    downstream_edge_count: int

    def __post_init__(self) -> None:
        numeric_values = (
            self.propagation_probability,
            self.previous_arrival_delay,
            self.turn_buffer,
            self.previous_delay_ratio,
            self.planned_turnaround,
        )

        if not all(
            isfinite(value)
            for value in numeric_values
        ):
            raise ValueError(
                "Flight decision inputs must be finite."
            )


        if not 0.0 <= self.propagation_probability <= 1.0:
            raise ValueError(
                "Propagation probability must be between "
                "0 and 1."
            )


        if self.previous_delay_ratio < 0:
            raise ValueError(
                "Previous delay ratio cannot be negative."
            )

        if self.planned_turnaround <= 0:
            raise ValueError(
                "Planned turnaround must be positive."
            )

        if self.downstream_edge_count < 0:
            raise ValueError(
                "Downstream edge count cannot be negative."
            )
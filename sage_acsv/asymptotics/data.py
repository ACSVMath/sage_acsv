
from dataclasses import dataclass
from enum import Enum

class LocalPointStrategy(Enum):
    SMOOTH = "smooth"
    TRANSVERSE_SIMPLE = "transverse_simple"
    HYPERPLANE_COMPLETE = "hyperplane_complete"
    TRANSVERSE_HIGHER_ORDER = "transverse_higher_order"

@dataclass(frozen=True)
class ClassifiedPoint:
    strategy: LocalPointStrategy
    codimension: object
    direction: object
    variables: object
    point: object
    substitutions: object
    unit: object
    factors: object
    multiplicities: object
    extra_factors: object
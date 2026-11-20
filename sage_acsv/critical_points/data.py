from dataclasses import dataclass, field

@dataclass(frozen=True)
class CriticalPointSystem:
    generators: object
    parent_ideal: object
    parent_dimension: object
    parent_stratum: tuple[object, object]

@dataclass(frozen=True)
class CriticalPointProblem:
    is_combinatorial: bool
    require_minimal: bool
    require_contributing: bool
    H_polynomial: object
    variables: object
    lambda_variable: object
    direction: object
    t_variable: object = None
    r_polynomials: list[object] = field(default_factory=list)
    r_variable_values: dict[object, object] = field(default_factory=dict)
    duplicate_variables: object = None
    auxiliary_modulus_variables: object = None
    nu_variable: object = None

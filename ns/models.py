from enum import Enum

class EOSType(Enum):
    """Enumeration of different neutron star types and their EOS models."""
    ATOMIC_STAR = "atomic_star"
    STRANGE_STAR = "strange_star"
    MAGNETIC_ATOMIC_STAR = "magnetic_atomic_star"
    MAGNETIC_STRANGE_STAR = "magnetic_strange_star"
    LSV_ATOMIC_STAR = "lsv_atomic_star"
    LSV_STRANGE_STAR = "lsv_strange_star"
    NLEM_ATOMIC_STAR = "nlem_atomic_star"
    NLEM_STRANGE_STAR = "nlem_strange_star"


class LSVType(Enum):
    """Lorentz Symmetry Violation types."""
    CASE_A = "case_a"
    ISOLATED = "isolated"


class NLEMType(Enum):
    """Nonlinear Electrodynamics Model types."""
    BORN_INFELD = "born_infeld"
    LOGARITHM = "logarithm"

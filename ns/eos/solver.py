from .atomic import AtomicStarEOS, MagneticAtomicStarEOS, LSVAtomicStarEOS, NLEMAtomicStarEOS
from .strange import StrangeStarEOS, MagneticStrangeStarEOS, LSVStrangeStarEOS, NLEMStrangeStarEOS
from ..models import EOSType

class EOSSolver:
    """Solver class for creating EOS instances."""
    
    _eos_map = {
        EOSType.ATOMIC_STAR: AtomicStarEOS,
        EOSType.STRANGE_STAR: StrangeStarEOS,
        EOSType.MAGNETIC_ATOMIC_STAR: MagneticAtomicStarEOS,
        EOSType.MAGNETIC_STRANGE_STAR: MagneticStrangeStarEOS,
        EOSType.LSV_ATOMIC_STAR: LSVAtomicStarEOS,
        EOSType.LSV_STRANGE_STAR: LSVStrangeStarEOS,
        EOSType.NLEM_ATOMIC_STAR: NLEMAtomicStarEOS,
        EOSType.NLEM_STRANGE_STAR: NLEMStrangeStarEOS,
    }
    
    @classmethod
    def create_eos(cls, eos_type: EOSType, **kwargs):
        """Create an EOS instance of the specified type."""
        if eos_type not in cls._eos_map:
            raise ValueError(f"Unknown EOS type: {eos_type}")
        
        return cls._eos_map[eos_type](**kwargs)
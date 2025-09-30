from typing import Optional, Dict, Any, Union
import numpy as np
from enum import Enum
from .parametrizations import ParametrizationType, get_parametrization_parameters, get_parametrization_info
from .constants import CRITICAL_MAGNETIC_FIELD_GAUSS

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


class EOS:
    def __init__(self, name: str = "DefaultEOS", 
                 eos_type: EOSType = EOSType.ATOMIC_STAR,
                 parametrization: Optional[ParametrizationType] = None,
                 lsv_type: Optional[LSVType] = None,
                 nlem_type: Optional[NLEMType] = None,
                 parameters: Optional[Dict[str, Any]] = None, 
                 verbose: bool = False):
        self.name = name
        self.eos_type = eos_type
        self.parametrization = parametrization
        self.lsv_type = lsv_type
        self.nlem_type = nlem_type
        self.parameters = parameters or {}
        self.initialized = False
        self.verbose = verbose
        self._results = {}
        
        # Set default EOS parameters based on EOS type and parametrization
        if not self.parameters:
            self._set_default_eos_parameters()
    
    def _set_default_eos_parameters(self):
        """Set default parameters based on EOS type and parametrization."""
        base_params = {
            'density_min': 1e14,  # kg/m³ - minimum density
            'density_max': 1e18,  # kg/m³ - maximum density  
            'pressure_scale': 1e35,  # Pa - pressure scaling
            'mass_scale': 1.989e30,  # kg - solar mass scale
            'radius_scale': 1e3,  # m - km scale
            'tolerance': 1e-8,
            'interpolation_points': 1000
        }
        
        # Get parametrization-specific parameters first
        if self.parametrization:
            param_specific = get_parametrization_parameters(self.parametrization)
            base_params.update(param_specific)
        
        # Add EOS-specific default parameters (can override parametrization)
        if self.eos_type == EOSType.ATOMIC_STAR:
            eos_params = {
                'nuclear_density': 2.8e17,  # kg/m³
                'binding_energy': 16.0,     # MeV
            }
            # If no parametrization specified, use default polytropic
            if not self.parametrization:
                eos_params.update({
                    'gamma': 2.0,               # Polytropic index
                    'K': 1e5                    # Polytropic constant
                })
        elif self.eos_type == EOSType.STRANGE_STAR:
            eos_params = {
                'bag_constant': 60.0,       # MeV/fm³
                'strange_mass': 150.0,      # MeV
                'strong_coupling': 0.3,
                'quark_masses': {'u': 2.3, 'd': 4.8, 's': 95.0},  # MeV
                'gamma': 1.5,               # Stiffer for strange matter
                'K': 5e4
            }
        elif self.eos_type == EOSType.MAGNETIC_ATOMIC_STAR:
            eos_params = {
                'magnetic_field': 1e15,     # Gauss
                'nuclear_density': 2.8e17,
                'magnetic_permeability': 1.0,
                'anisotropy_factor': 0.1,
            }
            if not self.parametrization:
                eos_params.update({
                    'gamma': 2.2,               # Slightly modified by magnetic field
                    'K': 1.2e5
                })
        elif self.eos_type == EOSType.MAGNETIC_STRANGE_STAR:
            eos_params = {
                'magnetic_field': 1e15,     # Gauss
                'bag_constant': 60.0,
                'strange_mass': 150.0,
                'magnetic_susceptibility': 0.05,
                'gamma': 1.6,
                'K': 6e4
            }
        elif self.eos_type == EOSType.LSV_ATOMIC_STAR:
            eos_params = {
                'lsv_parameter': 1e-15,
                'nuclear_density': 2.8e17,
                'lsv_energy_scale': 1e19,  # eV
            }
            if not self.parametrization:
                eos_params.update({
                    'gamma': 2.1,               # Modified by LSV
                    'K': 1.1e5
                })
        elif self.eos_type == EOSType.LSV_STRANGE_STAR:
            eos_params = {
                'lsv_parameter': 1e-15,
                'bag_constant': 60.0,
                'lsv_energy_scale': 1e19,
                'gamma': 1.55,
                'K': 5.5e4
            }
        elif self.eos_type == EOSType.NLEM_ATOMIC_STAR:
            eos_params = {
                'nlem_parameter': 1e-13,
                'nuclear_density': 2.8e17,
                'electromagnetic_coupling': 1.0/137.0,
            }
            if not self.parametrization:
                eos_params.update({
                    'gamma': 2.05,
                    'K': 1.05e5
                })
        elif self.eos_type == EOSType.NLEM_STRANGE_STAR:
            eos_params = {
                'nlem_parameter': 1e-13,
                'bag_constant': 60.0,
                'electromagnetic_coupling': 1.0/137.0,
                'gamma': 1.52,
                'K': 5.2e4
            }
        else:
            # Default to atomic star parameters
            eos_params = {
                'gamma': 2.0,
                'K': 1e5
            }
        
        # Combine base and EOS-specific parameters
        base_params.update(eos_params)
        self.set_parameters(base_params)
    
    def get_parametrization_info(self) -> Dict[str, Any]:
        """Get information about the current parametrization."""
        if not self.parametrization:
            return {'parametrization': None, 'description': 'No specific parametrization set'}
        
        info = get_parametrization_info(self.parametrization)
        return {
            'parametrization': self.parametrization.value,
            'description': f"{info['name']} ({info['model_type']})",
            'reference': info['reference'],
            'parameters': get_parametrization_parameters(self.parametrization)
        }
    
    def set_parameter(self, key: str, value: Any) -> None:
        """Set a single parameter value."""
        self.parameters[key] = value
        self.initialized = False  # Require re-initialization
        
        if self.verbose:
            print(f"Set parameter {key} = {value}")
    
    def set_parameters(self, params: Dict[str, Any]) -> None:
        """Set multiple parameters at once."""
        self.parameters.update(params)
        self.initialized = False  # Require re-initialization
        
        if self.verbose:
            print(f"Updated {len(params)} parameters")
    
    def get_parameter(self, key: str, default: Any = None) -> Any:
        """Get a parameter value."""
        return self.parameters.get(key, default)
    
    def requires_lsv(self) -> bool:
        """Check if this EOS type requires LSV model."""
        return self.eos_type in [EOSType.LSV_ATOMIC_STAR, EOSType.LSV_STRANGE_STAR]
    
    def requires_nlem(self) -> bool:
        """Check if this EOS type requires NLEM model."""
        return self.eos_type in [EOSType.NLEM_ATOMIC_STAR, EOSType.NLEM_STRANGE_STAR]
    
    def is_strange_star(self) -> bool:
        """Check if this is a strange star EOS."""
        return self.eos_type in [
            EOSType.STRANGE_STAR,
            EOSType.MAGNETIC_STRANGE_STAR,
            EOSType.LSV_STRANGE_STAR,
            EOSType.NLEM_STRANGE_STAR
        ]
    
    def is_magnetic_star(self) -> bool:
        """Check if this is a magnetic star EOS."""
        return self.eos_type in [
            EOSType.MAGNETIC_ATOMIC_STAR,
            EOSType.MAGNETIC_STRANGE_STAR
        ]
    
    def save_results(self, results: Dict[str, Any]) -> None:
        """Save computation results."""
        self._results.update(results)
        
        if self.verbose:
            print(f"Saved {len(results)} result entries")
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about this EOS."""
        return {
            'name': self.name,
            'type': self.eos_type.value,
            'parametrization': self.parametrization.value if self.parametrization else None,
            'initialized': self.initialized,
            'parameters': self.parameters
        }


# Factory functions for parametrized EOS models
def create_gm1_eos(name: str = "GM1_EOS", 
                   eos_type: EOSType = EOSType.ATOMIC_STAR,
                   verbose: bool = False) -> EOS:
    """Create EOS with GM1 parametrization."""
    return EOS(name=name, eos_type=eos_type, 
               parametrization=ParametrizationType.GM1, verbose=verbose)

def create_gm3_eos(name: str = "GM3_EOS", 
                   eos_type: EOSType = EOSType.ATOMIC_STAR,
                   verbose: bool = False) -> EOS:
    """Create EOS with GM3 parametrization."""
    return EOS(name=name, eos_type=eos_type, 
               parametrization=ParametrizationType.GM3, verbose=verbose)

def create_nl3_eos(name: str = "NL3_EOS", 
                   eos_type: EOSType = EOSType.ATOMIC_STAR,
                   verbose: bool = False) -> EOS:
    """Create EOS with NL3 parametrization."""
    return EOS(name=name, eos_type=eos_type, 
               parametrization=ParametrizationType.NL3, verbose=verbose)

def create_apr_eos(name: str = "APR_EOS", 
                   eos_type: EOSType = EOSType.ATOMIC_STAR,
                   verbose: bool = False) -> EOS:
    """Create EOS with APR parametrization."""
    return EOS(name=name, eos_type=eos_type, 
               parametrization=ParametrizationType.APR, verbose=verbose)


if __name__ == "__main__":
    # Example usage
    print("EOS with Parametrizations Demonstration")
    print("=" * 50)
    
    # Create different parametrized EOS models
    gm1_eos = create_gm1_eos(verbose=True)
    nl3_eos = create_nl3_eos(verbose=True)
    apr_eos = create_apr_eos(verbose=True)
    
    # Show parametrization information
    for eos in [gm1_eos, nl3_eos, apr_eos]:
        info = eos.get_parametrization_info()
        print(f"\n{eos.name}:")
        print(f"  Description: {info['description']}")
        print(f"  Reference: {info['reference']}")
        print(f"  Incompressibility: {info['parameters'].get('incompressibility', 'N/A')} MeV")
import numpy as np
from abc import ABC, abstractmethod

from ns.constants import NEUTRON_MASS_MEV
from ..parametrizations import ParametrizationType
from typing import Optional, Dict, Any, Callable
from ..models import EOSType, LSVType, NLEMType
from ..particle import ParticlesData
from ..parametrizations import get_parametrization_parameters, ParametrizationType
from scipy.optimize import fsolve
from scipy.integrate import quad

class BaseEOS(ABC):
    """Base class for all Equation of State implementations."""
    
    def __init__(self, name: str = "DefaultEOS", 
                 eos_type: EOSType = EOSType.ATOMIC_STAR,
                 parametrization: Optional[ParametrizationType] = None,
                 lsv_type: Optional[LSVType] = None,
                 nlem_type: Optional[NLEMType] = None,
                 parameters: Optional[Dict[str, Any]] = None,
                 root_finder: Optional[Callable] = fsolve,
                 integrator: Optional[Callable] = quad,
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
        
        # Initialize particle data
        self.particles_data = ParticlesData()
        
        # Set up root finder and integrator
        self.root_finder = root_finder
        self.integrator = integrator
        
        # Physical constants and particle properties
        self._setup_particle_properties()
        
        # Set default EOS parameters
        if not self.parameters:
            self._set_default_eos_parameters()
    
    def _set_default_eos_parameters(self):
        """Set default EOS parameters for RMF calculations."""
        
        from ..constants import (
            DEFAULT_TOLERANCE, DEFAULT_MAX_ITERATIONS, 
            DEFAULT_BARYON_DENSITY_MIN, DEFAULT_BARYON_DENSITY_MAX, 
        )
        
        # Start with GM1 as default parametrization
        if not self.parametrization:
            self.parametrization = ParametrizationType.GM1
            
        # Get parametrization data
        param_data = get_parametrization_parameters(self.parametrization)
        
        # Default RMF parameters using parametrization data
        default_params = {
            # Reference mass (nucleon mass in MeV)
            'reference_mass': NEUTRON_MASS_MEV,
            
            # Meson-nucleon coupling constants from parametrization
            'coupling_sigma': param_data.get('coupling_constants', {}).get('g_sigma', 10.217),
            'coupling_omega': param_data.get('coupling_constants', {}).get('g_omega', 12.868),
            'coupling_rho': param_data.get('coupling_constants', {}).get('g_rho', 4.474),
            
            # Meson masses from parametrization (MeV)
            'meson_sigma_mass': param_data.get('meson_masses', {}).get('sigma', 508.194),
            'meson_omega_mass': param_data.get('meson_masses', {}).get('omega', 782.501),
            'meson_rho_mass': param_data.get('meson_masses', {}).get('rho', 763.0),
            
            # Nonlinear coupling parameters from parametrization
            'nonlinear_k': param_data.get('nonlinear_parameters', {}).get('g2', -10.431),
            'nonlinear_lambda': param_data.get('nonlinear_parameters', {}).get('g3', -28.885),
            'csi_parameter': 0.0,        # Quartic omega self-coupling
            
            # Physical properties from parametrization
            'incompressibility': param_data.get('incompressibility', 271.8),
            'symmetry_energy': param_data.get('symmetry_energy', 37.4),
            'effective_mass_ratio': param_data.get('effective_mass_ratio', 0.595),
            'saturation_density': param_data.get('saturation_density', 2.27e17),
            'binding_energy': param_data.get('binding_energy', 16.24),
            'gamma': param_data.get('gamma', 2.78),
            
            # Hyperon coupling ratios (for strange stars)
            'hyperon_sigma_coupling': 0.7,   # x_s = g_s^Y / g_s^N
            'hyperon_omega_coupling': 0.783, # x_v = g_v^Y / g_v^N
            'hyperon_rho_coupling': 1.0,     # x_r = g_r^Y / g_r^N
            
            # Numerical parameters
            'tolerance': DEFAULT_TOLERANCE,              # Convergence tolerance
            'max_iterations': DEFAULT_MAX_ITERATIONS,    # Maximum iterations for field equations
            
            # Baryon density range for calculations (fm⁻³)
            'baryon_density_min': DEFAULT_BARYON_DENSITY_MIN,  # Minimum baryon density
            'baryon_density_max': DEFAULT_BARYON_DENSITY_MAX,  # Maximum baryon density
            
            # Magnetic field parameters (for magnetic stars)
            'magnetic_field': 0.0,       # Magnetic field strength (Gauss)
            'anisotropy_factor': 0.0,    # Magnetic anisotropy parameter
            
            # LSV parameters (for Lorentz violation)
            'lsv_coefficient': 0.0,      # LSV coefficient
            'lsv_power': 1.0,           # LSV power law exponent
            
            # NLEM parameters (for nonlinear electrodynamics)
            'nlem_beta': 0.0,           # NLEM parameter β
            'nlem_scale': 1.0,          # NLEM scale parameter
        }
        
        # Set the parameters
        self.parameters.update(default_params)
        
        if self.verbose:
            param_name = param_data.get('name', 'Unknown')
            print(f"Initialized EOS with {param_name} parametrization")
            print(f"Reference: {param_data.get('reference', 'Unknown')}")
            print("Key parameters:")
            print(f"  Sigma coupling: {self.parameters['coupling_sigma']}")
            print(f"  Omega coupling: {self.parameters['coupling_omega']}")
            print(f"  Rho coupling: {self.parameters['coupling_rho']}")
            print(f"  Incompressibility: {self.parameters['incompressibility']} MeV")
            print(f"  Symmetry energy: {self.parameters['symmetry_energy']} MeV")
    
    
    def _setup_particle_properties(self):
        """Setup particle masses, charges, and other properties using particle.py definitions."""
        
        # Get all particles
        all_particles = self.particles_data.get_all_particles()
        
        # Define baryon order for octet: neutron, proton, lambda, sigma-, sigma0, sigma+, xi-, xi0
        if self.eos_type in [EOSType.ATOMIC_STAR, EOSType.MAGNETIC_ATOMIC_STAR, 
                             EOSType.LSV_ATOMIC_STAR, EOSType.NLEM_ATOMIC_STAR]:
            # Atomic stars: only nucleons
            self.baryon_list = [
                all_particles['neutron'],      # 0: neutron
                all_particles['proton']        # 1: proton  
            ]
            self.n_baryons = 2
        elif self.eos_type in [EOSType.STRANGE_STAR, EOSType.MAGNETIC_STRANGE_STAR,
                               EOSType.LSV_STRANGE_STAR, EOSType.NLEM_STRANGE_STAR]:
            # Strange stars: full baryon octet
            self.baryon_list = [
                all_particles['neutron'],      # 0: neutron
                all_particles['proton'],       # 1: proton
                all_particles['lambda'],       # 2: lambda
                all_particles['sigma_minus'],  # 3: sigma-
                all_particles['sigma_zero'],   # 4: sigma0 
                all_particles['sigma_plus'],   # 5: sigma+
                all_particles['xi_minus'],     # 6: xi- (cascade-)
                all_particles['xi_zero']       # 7: xi0 (cascade0)
            ]
            self.n_baryons = 8
        else:
            # Default to atomic
            self.baryon_list = [
                all_particles['neutron'],
                all_particles['proton']
            ]
            self.n_baryons = 2
        
        # Lepton list
        self.lepton_list = [
            all_particles['electron'],  # 0: electron
            all_particles['muon']       # 1: muon
        ]
        self.n_leptons = 2
        
        # Extract arrays for easy access and normalize by reference mass
        rm = NEUTRON_MASS_MEV  # Reference mass for normalization
        
        self.baryon_masses = np.array([p.mass / rm for p in self.baryon_list])
        self.baryon_charges = np.array([p.charge for p in self.baryon_list])
        self.baryon_isospin_3 = np.array([p.isospin_z for p in self.baryon_list])
        self.baryon_strangeness = np.array([p.strangeness for p in self.baryon_list])
        
        # Lepton masses also normalized
        self.lepton_masses = np.array([p.mass / rm for p in self.lepton_list])  # Normalized masses
        self.lepton_charges = np.array([p.charge for p in self.lepton_list])
        
        # Constants
        self.pi2 = 9.86960441009  # π²
        self.hbar_c = 197.32  # MeV⋅fm
        
        if self.verbose:
            print(f"Initialized {self.eos_type.value} with {self.n_baryons} baryons:")
            for i, baryon in enumerate(self.baryon_list):
                print(f"  {i}: {baryon.id} (m={self.baryon_masses[i]:.1f} MeV, "
                      f"q={self.baryon_charges[i]:+.1f}e, I_z={self.baryon_isospin_3[i]:+.1f}, "
                      f"S={self.baryon_strangeness[i]})")
    
    def get_parameter(self, name: str, default: Any = None) -> Any:
        """Get parameter value by name."""
        return self.parameters.get(name, default)
    
    @abstractmethod
    def compute_eos(self, solution: Dict[str, float]) -> float:
        """Calculate eos for given solution."""
        pass
    
    @abstractmethod
    def compute_energy_density(self, solution: Dict[str, float]) -> float:
        """Calculate energy density for given solution."""
        pass
    
    @abstractmethod
    def compute_pressure(self, solution: Dict[str, float]) -> float:
        """Calculate pressure for given density."""
        pass
    
    @abstractmethod
    def solve_field_equations(self, baryon_density: float) -> Dict[str, float]:
        """Calculate field equations for the eos"""
        pass
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
        
        # IMPORTANT: Set default EOS parameters FIRST (before particle setup)
        self._set_default_eos_parameters()
        
        # THEN: Physical constants and particle properties
        # This will call _parameters_to_variables() which sets self.parameters = None
        self._setup_particle_properties()
    
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
    
        # Baryon properties (normalized)
        self.baryon_masses = np.array([p.mass / rm for p in self.baryon_list])
        self.baryon_charges = np.array([p.charge for p in self.baryon_list])
        self.baryon_isospin_3 = np.array([p.isospin_z for p in self.baryon_list])
        self.baryon_strangeness = np.array([p.strangeness for p in self.baryon_list])
        
        # Lepton properties (normalized)
        self.lepton_masses = np.array([p.mass / rm for p in self.lepton_list])
        self.lepton_charges = np.array([p.charge for p in self.lepton_list])
        
        # Convert parameters to class variables for fast access
        self._parameters_to_variables()
        
        if self.verbose:
            print(f"\nInitialized {self.eos_type.value} with {self.n_baryons} baryons")
            print("\nBaryons:")
            for i, baryon in enumerate(self.baryon_list):
                print(f"  {i}: {baryon.id:12s} m={self.baryon_masses[i]*rm:8.3f} MeV, "
                      f"q={self.baryon_charges[i]:+.1f}e, I_z={self.baryon_isospin_3[i]:+.1f}, "
                      f"S={self.baryon_strangeness[i]:+d}")
            
            print("\nLeptons:")
            for i, lepton in enumerate(self.lepton_list):
                print(f"  {i}: {lepton.id:12s} m={self.lepton_masses[i]*rm:8.3f} MeV, "
                      f"q={self.lepton_charges[i]:+.1f}e")
    
    
    def _parameters_to_variables(self):
        """
        Convert parameters dictionary to class attributes for direct access.
        This improves performance by avoiding dictionary lookups in computationally intensive loops.
        
        Should be called after _set_default_eos_parameters() to ensure all parameters are set.
        """
        
        rm = NEUTRON_MASS_MEV  # Reference mass for normalization
        self.rm = rm  # Store reference mass
        
        # ============================================
        # Meson properties (normalized to rm)
        # ============================================
        self.ms = self.parameters.get('meson_sigma_mass', 508.194) / rm  # Sigma meson
        self.mw = self.parameters.get('meson_omega_mass', 782.501) / rm  # Omega meson
        self.mr = self.parameters.get('meson_rho_mass', 763.0) / rm      # Rho meson
        
        # ============================================
        # Coupling constants (dimensionless)
        # ============================================
        self.gs = self.parameters.get('coupling_sigma', 10.217)     # σ-N coupling
        self.gw = self.parameters.get('coupling_omega', 12.868)     # ω-N coupling
        self.gr = self.parameters.get('coupling_rho', 4.474)        # ρ-N coupling
        
        # ============================================
        # Nonlinear self-coupling parameters
        # ============================================
        self.k_nl = self.parameters.get('nonlinear_k', 0.0)         # κ (cubic σ)
        self.lambda_nl = self.parameters.get('nonlinear_lambda', 0.0)  # λ (quartic σ)
        self.csi = self.parameters.get('csi_parameter', 0.0)        # ξ (quartic ω)
        
        # ============================================
        # Hyperon coupling ratios (for strange stars)
        # ============================================
        self.xs = self.parameters.get('hyperon_sigma_coupling', 0.7)   # x_σ = g_σ^Y / g_σ^N
        self.xv = self.parameters.get('hyperon_omega_coupling', 0.783) # x_ω = g_ω^Y / g_ω^N
        self.xr = self.parameters.get('hyperon_rho_coupling', 1.0)     # x_ρ = g_ρ^Y / g_ρ^N
        
        # Hyperon coupling constants (derived)
        self.gs_hyperon = self.xs * self.gs
        self.gw_hyperon = self.xv * self.gw
        self.gr_hyperon = self.xr * self.gr
        
        # ============================================
        # Nuclear matter properties at saturation
        # ============================================
        self.rho_sat = self.parameters.get('saturation_density', 0.153)  # fm⁻³
        self.binding_energy = self.parameters.get('binding_energy', 16.24)  # MeV
        self.incompressibility = self.parameters.get('incompressibility', 271.8)  # MeV
        self.symmetry_energy = self.parameters.get('symmetry_energy', 37.4)  # MeV
        self.effective_mass_ratio = self.parameters.get('effective_mass_ratio', 0.595)  # m*/m
        self.gamma = self.parameters.get('gamma', 2.78)  # Polytropic index
        
        # ============================================
        # Magnetic field parameters (for magnetic stars)
        # ============================================
        self.B_field = self.parameters.get('magnetic_field', 0.0)  # Gauss
        self.anisotropy = self.parameters.get('anisotropy_factor', 0.0)
        
        # ============================================
        # LSV parameters (for Lorentz violation)
        # ============================================
        self.lsv_coeff = self.parameters.get('lsv_coefficient', 0.0)
        self.lsv_power = self.parameters.get('lsv_power', 1.0)
        
        # ============================================
        # NLEM parameters (for nonlinear electrodynamics)
        # ============================================
        self.nlem_beta = self.parameters.get('nlem_beta', 0.0)
        self.nlem_scale = self.parameters.get('nlem_scale', 1.0)
        
        # ============================================
        # Numerical parameters
        # ============================================
        self.tolerance = self.parameters.get('tolerance', 1e-6)
        self.max_iterations = self.parameters.get('max_iterations', 100)
        self.rho_min = self.parameters.get('baryon_density_min', 0.01)  # fm⁻³
        self.rho_max = self.parameters.get('baryon_density_max', 2.0)   # fm⁻³
        
        # ============================================
        # Store original parameters for reference/debugging
        # ============================================
        self._original_parameters = self.parameters.copy()
        
        # Optional: Clear parameters dict to save memory (can uncomment if needed)
        self.parameters = None  #Set to None to completely free memory
        
        if self.verbose:
            print("\n" + "="*60)
            print("CONVERTED PARAMETERS TO CLASS VARIABLES")
            print("="*60)
            
            print("\nMeson masses (normalized):")
            print(f"  σ: {self.ms:.6f} ({self.ms*rm:.3f} MeV)")
            print(f"  ω: {self.mw:.6f} ({self.mw*rm:.3f} MeV)")
            print(f"  ρ: {self.mr:.6f} ({self.mr*rm:.3f} MeV)")
            
            print("\nCoupling constants:")
            print(f"  g_σ: {self.gs:.6f}")
            print(f"  g_ω: {self.gw:.6f}")
            print(f"  g_ρ: {self.gr:.6f}")
            
            if self.n_baryons > 2:
                print("\nHyperon coupling ratios:")
                print(f"  x_σ: {self.xs:.6f} → g_σ^Y = {self.gs_hyperon:.6f}")
                print(f"  x_ω: {self.xv:.6f} → g_ω^Y = {self.gw_hyperon:.6f}")
                print(f"  x_ρ: {self.xr:.6f} → g_ρ^Y = {self.gr_hyperon:.6f}")
            
            if abs(self.k_nl) > 1e-10 or abs(self.lambda_nl) > 1e-10:
                print("\nNonlinear parameters:")
                print(f"  κ (g2): {self.k_nl:.6f}")
                print(f"  λ (g3): {self.lambda_nl:.6f}")
                if abs(self.csi) > 1e-10:
                    print(f"  ξ: {self.csi:.6f}")
            
            print("\nNuclear matter properties:")
            print(f"  ρ_sat: {self.rho_sat:.6f} fm⁻³")
            print(f"  Binding energy: {self.binding_energy:.3f} MeV")
            print(f"  Incompressibility: {self.incompressibility:.3f} MeV")
            print(f"  Symmetry energy: {self.symmetry_energy:.3f} MeV")
            print(f"  m*/m: {self.effective_mass_ratio:.3f}")
            print(f"  γ: {self.gamma:.3f}")
            
            if abs(self.B_field) > 1e-10:
                print(f"\nMagnetic field: {self.B_field:.4e} Gauss")
                print(f"Anisotropy: {self.anisotropy:.6f}")
            
            if abs(self.lsv_coeff) > 1e-10:
                print(f"\nLSV coefficient: {self.lsv_coeff:.6e}")
                print(f"LSV power: {self.lsv_power:.3f}")
            
            if abs(self.nlem_beta) > 1e-10:
                print(f"\nNLEM β: {self.nlem_beta:.6e}")
                print(f"NLEM scale: {self.nlem_scale:.6e}")
            
            print("\nNumerical parameters:")
            print(f"  Tolerance: {self.tolerance:.2e}")
            print(f"  Max iterations: {self.max_iterations}")
            print(f"  ρ range: [{self.rho_min:.3f}, {self.rho_max:.3f}] fm⁻³")
            print("="*60 + "\n")
    
    def get_parameter(self, name: str, default: Any = None) -> Any:
        """
        Get parameter value by name. 
        First checks class attributes, then falls back to original parameters dict.
        
        Args:
            name: Parameter name
            default: Default value if not found
            
        Returns:
            Parameter value
            
        Raises:
            KeyError: If parameter not found and no default provided
        """
        # First check if it exists as a class attribute (after _parameters_to_variables)
        if hasattr(self, name):
            return getattr(self, name)
        
        # Then check original parameters (stored before clearing)
        if hasattr(self, '_original_parameters') and name in self._original_parameters:
            return self._original_parameters[name]
        
        # Fall back to current parameters dict (if it still exists and is not None)
        if self.parameters is not None and name in self.parameters:
            return self.parameters[name]
        
        # Use default if provided
        if default is not None:
            return default
        
        # Raise error if nothing found
        raise KeyError(f"Parameter '{name}' not found and no default provided.")
    
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
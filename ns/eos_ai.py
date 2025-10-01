import numpy as np
from typing import Optional, Dict, Any, Union, List
from enum import Enum
from scipy.optimize import fsolve
from scipy.integrate import quad
from .parametrizations import ParametrizationType, get_parametrization_parameters, get_parametrization_info
from .constants import CRITICAL_MAGNETIC_FIELD_GAUSS, NEUTRON_MASS_MEV
from .particle import ParticlesData


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
    """
    Equation of State implementation based on relativistic mean field theory.
    
    This implementation follows the structure of the QHD-II model from the Fortran code,
    supporting baryon octet (neutron, proton, lambda, sigma, cascade) and leptons.
    """
    
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
        
        # Initialize particle data
        self.particles_data = ParticlesData()
        
        # Physical constants and particle properties
        self._setup_particle_properties()
        
        # Set default EOS parameters
        if not self.parameters:
            self._set_default_eos_parameters()
    
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
        
        # Extract arrays for easy access (convert from eV to MeV)
        self.baryon_masses = np.array([p.mass / 1e6 for p in self.baryon_list])  # Convert eV to MeV
        self.baryon_charges = np.array([p.charge for p in self.baryon_list])
        self.baryon_isospin_3 = np.array([p.isospin_z for p in self.baryon_list])
        self.baryon_strangeness = np.array([p.strangeness for p in self.baryon_list])
        
        self.lepton_masses = np.array([p.mass / 1e6 for p in self.lepton_list])  # Convert eV to MeV
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
    
    def get_baryon_properties(self, index: int) -> Dict[str, Any]:
        """Get properties of a specific baryon by index."""
        if 0 <= index < self.n_baryons:
            baryon = self.baryon_list[index]
            return {
                'name': baryon.id,
                'mass': self.baryon_masses[index],  # MeV
                'charge': self.baryon_charges[index],  # units of e
                'isospin': baryon.isospin,
                'isospin_z': self.baryon_isospin_3[index],
                'strangeness': self.baryon_strangeness[index],
                'spin': baryon.spin
            }
        else:
            raise IndexError(f"Baryon index {index} out of range [0, {self.n_baryons-1}]")
    
    def get_lepton_properties(self, index: int) -> Dict[str, Any]:
        """Get properties of a specific lepton by index."""
        if 0 <= index < self.n_leptons:
            lepton = self.lepton_list[index]
            return {
                'name': lepton.id,
                'mass': self.lepton_masses[index],  # MeV
                'charge': self.lepton_charges[index],  # units of e
                'spin': lepton.spin,
                'generation': lepton.generation
            }
        else:
            raise IndexError(f"Lepton index {index} out of range [0, {self.n_leptons-1}]")
    
    def print_particle_content(self):
        """Print the particle content for this EOS type."""
        print(f"\nParticle content for {self.eos_type.value}:")
        print("=" * 60)
        
        print("\nBaryons:")
        print("-" * 40)
        for i in range(self.n_baryons):
            props = self.get_baryon_properties(i)
            print(f"  {i}: {props['name']:10s} (m={props['mass']:7.1f} MeV, "
                  f"q={props['charge']:+.1f}e, I_z={props['isospin_z']:+.1f}, "
                  f"S={props['strangeness']})")
        
        print("\nLeptons:")
        print("-" * 40)
        for i in range(self.n_leptons):
            props = self.get_lepton_properties(i)
            print(f"  {i}: {props['name']:10s} (m={props['mass']:7.3f} MeV, "
                  f"q={props['charge']:+.1f}e, gen={props['generation']})")
    
    def _solve_atomic_field_equations(self, baryon_density: float) -> Dict[str, float]:
        """
        Solve field equations for atomic stars (only neutron, proton, electron, muon).
        
        Args:
            baryon_density: Total baryonic number density in fm⁻³
        """
        def field_equations(fields):
            """
            System of equations for atomic stars.
            fields = [kf_electron, kf_neutron, sigma_field, omega_field, rho_field]
            """
            kf_e, kf_n, sigma, omega, rho = fields
            
            # Apply mappings (like in Fortran)
            kf_e = abs(kf_e)
            kf_n = abs(kf_n)
            sigma = 0.999999 * np.sin(sigma)**2  # Map to (0, m_nucleon)
            omega = omega**2
            rho = rho  # No mapping needed
            
            # Get reference mass and couplings
            rm = self.get_parameter('reference_mass', NEUTRON_MASS_MEV)
            g_s = self.get_parameter('coupling_sigma', 10.0)
            g_v = self.get_parameter('coupling_omega', 12.0) 
            g_r = self.get_parameter('coupling_rho', 4.5)
            
            # Meson masses (in units of reference mass)
            m_s = self.get_parameter('meson_sigma_mass', 550.0) / rm
            m_v = self.get_parameter('meson_omega_mass', 783.0) / rm
            m_r = self.get_parameter('meson_rho_mass', 770.0) / rm
            
            # Nonlinear parameters
            k = self.get_parameter('nonlinear_k', 0.0)
            lam = self.get_parameter('nonlinear_lambda', 0.0)
            csi = self.get_parameter('csi_parameter', 0.0)
            
            # Initialize field source terms
            sigma_source = 0.0
            omega_source = 0.0
            rho_source = 0.0
            charge_density = 0.0
            total_baryon_density = 0.0
            
            # Only consider neutron (i=0) and proton (i=1) for atomic stars
            for i in range(2):  # Only nucleons for atomic stars
                # Effective mass (same for both nucleons in nuclear units)
                m_eff = 1.0 - sigma  # Nucleon effective mass in units of reference mass
                
                # Chemical potential constraint
                if i == 0:  # neutron
                    ef_n = np.sqrt(kf_n**2 + m_eff**2)
                    mu_n = omega + self.baryon_isospin_3[i] * rho + ef_n
                
                # Electron chemical potential
                m_e_nuclear = self.lepton_masses[0] / rm  # Electron mass in nuclear units
                mu_e = np.sqrt(kf_e**2 + m_e_nuclear**2)
                
                # Baryon chemical potential (β-equilibrium)
                mu_i = mu_n - self.baryon_charges[i] * mu_e
                
                # Fermi momentum for nucleon i
                kf_i_squared = (mu_i - omega - self.baryon_isospin_3[i] * rho)**2 - m_eff**2
                
                if kf_i_squared > 0:
                    kf_i = np.sqrt(kf_i_squared)
                    
                    # Baryon number density (fm⁻³)
                    n_i = kf_i**3 / (3.0 * self.pi2)
                    
                    # Field source terms
                    sigma_source += g_s**2 / m_s**2 * self._sigma_integral(kf_i, m_eff)
                    omega_source += g_v**2 / m_v**2 * n_i
                    rho_source += self.baryon_isospin_3[i] * g_r**2 / m_r**2 * n_i
                    
                    charge_density += self.baryon_charges[i] * n_i
                    total_baryon_density += n_i
            
            # Lepton contributions
            n_e = kf_e**3 / (3.0 * self.pi2)  # Electron number density (fm⁻³)
            charge_density -= self.lepton_charges[0] * n_e  # electron charge
            
            # Muon contribution if above threshold
            m_mu_nuclear = self.lepton_masses[1] / rm
            kf_mu_squared = kf_e**2 + m_e_nuclear**2 - m_mu_nuclear**2
            if kf_mu_squared > 0:
                kf_mu = np.sqrt(kf_mu_squared)
                n_mu = kf_mu**3 / (3.0 * self.pi2)
                charge_density -= self.lepton_charges[1] * n_mu  # muon charge
            
            # Nonlinear terms
            sigma_nl = -k/(2.0 * m_s**2) * sigma**2/g_s - lam/(6.0 * m_s**2) * sigma**3/g_s**2
            omega_nl = -csi * g_v**4/(6.0 * m_v**2) * omega**3/g_v**2
            
            # Field equations
            eq1 = sigma_source + sigma_nl - sigma  # Sigma field equation
            eq2 = omega_source + omega_nl - omega  # Omega field equation  
            eq3 = rho_source - rho                 # Rho field equation
            eq4 = charge_density                   # Charge neutrality
            eq5 = total_baryon_density - baryon_density  # Baryon number conservation
            
            return [eq1, eq2, eq3, eq4, eq5]
        
        # Initial guess based on baryon density
        initial_guess = [
            (3.0 * self.pi2 * 0.1 * baryon_density)**(1/3),  # kf_electron
            (3.0 * self.pi2 * 0.9 * baryon_density)**(1/3),  # kf_neutron  
            np.arcsin(np.sqrt(0.005/0.999999)),               # sigma field (mapped)
            np.sqrt(0.22731e-5),                              # omega field
            -0.57641e-6                                       # rho field
        ]
        
        try:
            solution = fsolve(field_equations, initial_guess, 
                             xtol=self.get_parameter('tolerance', 1e-8))
            
            # Extract fields
            kf_e, kf_n, sigma_mapped, omega, rho = solution
            
            # Apply mappings
            kf_e = abs(kf_e)
            kf_n = abs(kf_n)
            sigma = 0.999999 * np.sin(sigma_mapped)**2
            omega = omega**2
            
            return {
                'kf_electron': kf_e,
                'kf_neutron': kf_n,
                'sigma_field': sigma,
                'omega_field': omega,
                'rho_field': rho,
                'converged': True
            }
            
        except Exception as e:
            if self.verbose:
                print(f"Atomic field equation solution failed: {e}")
            return {
                'kf_electron': 0.0,
                'kf_neutron': 0.0,
                'sigma_field': 0.0,
                'omega_field': 0.0,
                'rho_field': 0.0,
                'converged': False
            }
    
    def _solve_strange_field_equations(self, baryon_density: float) -> Dict[str, float]:
        """
        Solve field equations for strange stars (full baryon octet).
        
        Args:
            baryon_density: Total baryonic number density in fm⁻³
        """
        def field_equations(fields):
            """
            System of equations for strange stars with full baryon octet.
            fields = [kf_electron, kf_neutron, sigma_field, omega_field, rho_field]
            """
            kf_e, kf_n, sigma, omega, rho = fields
            
            # Apply mappings
            kf_e = abs(kf_e)
            kf_n = abs(kf_n)
            sigma = 0.999999 * np.sin(sigma)**2
            omega = omega**2
            rho = rho
            
            # Get parameters
            rm = self.get_parameter('reference_mass', NEUTRON_MASS_MEV)
            g_s = self.get_parameter('coupling_sigma', 10.0)
            g_v = self.get_parameter('coupling_omega', 12.0) 
            g_r = self.get_parameter('coupling_rho', 4.5)
            
            # Meson masses
            m_s = self.get_parameter('meson_sigma_mass', 550.0) / rm
            m_v = self.get_parameter('meson_omega_mass', 783.0) / rm
            m_r = self.get_parameter('meson_rho_mass', 770.0) / rm
            
            # Nonlinear parameters
            k = self.get_parameter('nonlinear_k', 0.0)
            lam = self.get_parameter('nonlinear_lambda', 0.0)
            csi = self.get_parameter('csi_parameter', 0.0)
            
            # Hyperon coupling ratios
            x_s = self.get_parameter('hyperon_sigma_coupling', 0.7)
            x_v = self.get_parameter('hyperon_omega_coupling', 0.783)
            x_r = self.get_parameter('hyperon_rho_coupling', 1.0)
            
            # Initialize field source terms
            sigma_source = 0.0
            omega_source = 0.0
            rho_source = 0.0
            charge_density = 0.0
            total_baryon_density = 0.0
            
            # Calculate contributions from all 8 baryons
            for i in range(8):  # Full baryon octet for strange stars
                # Effective mass and couplings
                if i < 2:  # nucleons (n, p)
                    m_eff = 1.0 - sigma  # In units of reference mass
                    g_s_i = g_s
                    g_v_i = g_v
                    g_r_i = g_r
                else:  # hyperons (Λ, Σ, Ξ)
                    m_eff = self.baryon_masses[i]/rm - x_s * sigma
                    g_s_i = x_s * g_s
                    g_v_i = x_v * g_v
                    g_r_i = x_r * g_r
                
                # Chemical potential constraint
                if i == 0:  # neutron
                    ef_n = np.sqrt(kf_n**2 + (1.0 - sigma)**2)
                    mu_n = omega + self.baryon_isospin_3[0] * rho + ef_n
                
                # Electron chemical potential
                m_e_nuclear = self.lepton_masses[0] / rm
                mu_e = np.sqrt(kf_e**2 + m_e_nuclear**2)
                
                # Baryon chemical potential
                mu_i = mu_n - self.baryon_charges[i] * mu_e
                
                # Fermi momentum for baryon i
                kf_i_squared = (mu_i - g_v_i/g_v * omega - g_r_i/g_r * self.baryon_isospin_3[i] * rho)**2 - m_eff**2
            
                if kf_i_squared > 0:
                    kf_i = np.sqrt(kf_i_squared)
                    
                    # Baryon number density (fm⁻³)
                    n_i = kf_i**3 / (3.0 * self.pi2)
                    
                    # Field source terms
                    sigma_source += g_s_i * g_s / m_s**2 * self._sigma_integral(kf_i, m_eff)
                    omega_source += g_v_i * g_v / m_v**2 * n_i
                    rho_source += self.baryon_isospin_3[i] * g_r_i * g_r / m_r**2 * n_i
                    
                    charge_density += self.baryon_charges[i] * n_i
                    total_baryon_density += n_i
            
            # Lepton contributions (same as atomic)
            n_e = kf_e**3 / (3.0 * self.pi2)
            charge_density -= n_e
            
            # Muon contribution
            m_mu_nuclear = self.lepton_masses[1] / rm
            kf_mu_squared = kf_e**2 + m_e_nuclear**2 - m_mu_nuclear**2
            if kf_mu_squared > 0:
                kf_mu = np.sqrt(kf_mu_squared)
                n_mu = kf_mu**3 / (3.0 * self.pi2)
                charge_density -= n_mu
            
            # Nonlinear terms
            sigma_nl = -k/(2.0 * m_s**2) * sigma**2/g_s - lam/(6.0 * m_s**2) * sigma**3/g_s**2
            omega_nl = -csi * g_v**4/(6.0 * m_v**2) * omega**3/g_v**2
            
            # Field equations
            eq1 = sigma_source + sigma_nl - sigma
            eq2 = omega_source + omega_nl - omega
            eq3 = rho_source - rho
            eq4 = charge_density
            eq5 = total_baryon_density - baryon_density
            
            return [eq1, eq2, eq3, eq4, eq5]
        
        # Initial guess based on baryon density
        initial_guess = [
            (3.0 * self.pi2 * 0.1 * baryon_density)**(1/3),  # kf_electron
            (3.0 * self.pi2 * 0.8 * baryon_density)**(1/3),  # kf_neutron  
            np.arcsin(np.sqrt(0.005/0.999999)),               # sigma field (mapped)
            np.sqrt(0.22731e-5),                              # omega field
            -0.57641e-6                                       # rho field
        ]
        
        try:
            solution = fsolve(field_equations, initial_guess, 
                             xtol=self.get_parameter('tolerance', 1e-8))
            
            # Extract fields
            kf_e, kf_n, sigma_mapped, omega, rho = solution
            
            # Apply mappings
            kf_e = abs(kf_e)
            kf_n = abs(kf_n)
            sigma = 0.999999 * np.sin(sigma_mapped)**2
            omega = omega**2
            
            return {
                'kf_electron': kf_e,
                'kf_neutron': kf_n,
                'sigma_field': sigma,
                'omega_field': omega,
                'rho_field': rho,
                'converged': True
            }
            
        except Exception as e:
            if self.verbose:
                print(f"Strange field equation solution failed: {e}")
            return {
                'kf_electron': 0.0,
                'kf_neutron': 0.0,
                'sigma_field': 0.0,
                'omega_field': 0.0,
                'rho_field': 0.0,
                'converged': False
            }

    def compute_eos(self, baryon_density: float) -> Dict[str, float]:
        """
        Compute equation of state for given baryonic number density.
        
        Args:
            baryon_density: Baryonic number density in fm⁻³
            
        Returns:
            Dictionary with EOS results including pressure, energy density, etc.
        """
        if not self.initialized:
            self.initialize()
        
        # Route to appropriate computation method based on EOS type
        if self.eos_type in [EOSType.ATOMIC_STAR, EOSType.MAGNETIC_ATOMIC_STAR, 
                             EOSType.LSV_ATOMIC_STAR, EOSType.NLEM_ATOMIC_STAR]:
            base_result = self._compute_atomic_eos(baryon_density)
        elif self.eos_type in [EOSType.STRANGE_STAR, EOSType.MAGNETIC_STRANGE_STAR,
                               EOSType.LSV_STRANGE_STAR, EOSType.NLEM_STRANGE_STAR]:
            base_result = self._compute_strange_eos(baryon_density)
        else:
            raise ValueError(f"Unknown EOS type: {self.eos_type}")
        
        if not base_result['converged']:
            return base_result
        
        # Apply exotic physics modifications
        energy_density = base_result['energy_density']
        pressure = base_result['pressure']
        
        if self.is_magnetic_star():
            energy_density, pressure = self._apply_magnetic_corrections(energy_density, pressure, baryon_density)
        
        if self.requires_lsv():
            energy_density, pressure = self._apply_lsv_corrections(energy_density, pressure, baryon_density)
        
        if self.requires_nlem():
            energy_density, pressure = self._apply_nlem_corrections(energy_density, pressure, baryon_density)
        
        # Update result with modified values
        base_result.update({
            'energy_density': energy_density,
            'pressure': pressure
        })
        
        return base_result

    def _compute_atomic_eos(self, baryon_density: float) -> Dict[str, float]:
        """
        Compute EOS for atomic stars using only neutron, proton, electron, muon.
        
        Args:
            baryon_density: Baryonic number density in fm⁻³
        """
        if self.verbose:
            print(f"Computing atomic EOS (n, p, e, μ) for baryon density {baryon_density:.4f} fm⁻³")
        
        # Solve RMF field equations for atomic matter (only nucleons)
        fields = self._solve_atomic_field_equations(baryon_density)
        
        if not fields['converged']:
            return {
                'baryon_density': baryon_density,
                'pressure': 0.0,
                'energy_density': 0.0,
                'converged': False
            }
        
        # Extract field values
        sigma = fields['sigma_field']
        omega = fields['omega_field']
        rho = fields['rho_field']
        kf_e = fields['kf_electron']
        kf_n = fields['kf_neutron']
        
        # Calculate energy density and pressure for atomic matter
        energy_density, pressure = self._calculate_atomic_energy_pressure(
            sigma, omega, rho, kf_e, kf_n, baryon_density)
        
        return {
            'baryon_density': baryon_density,
            'pressure': pressure,
            'energy_density': energy_density,
            'sigma_field': sigma,
            'omega_field': omega,
            'rho_field': rho,
            'kf_electron': kf_e,
            'kf_neutron': kf_n,
            'converged': True
        }

    def _compute_strange_eos(self, baryon_density: float) -> Dict[str, float]:
        """
        Compute EOS for strange stars using full baryon octet.
        
        Args:
            baryon_density: Baryonic number density in fm⁻³
        """
        if self.verbose:
            print(f"Computing strange EOS (full octet) for baryon density {baryon_density:.4f} fm⁻³")
        
        # Solve RMF field equations for strange matter (full octet)
        fields = self._solve_strange_field_equations(baryon_density)
        
        if not fields['converged']:
            return {
                'baryon_density': baryon_density,
                'pressure': 0.0,
                'energy_density': 0.0,
                'converged': False
            }
        
        # Extract field values
        sigma = fields['sigma_field']
        omega = fields['omega_field']
        rho = fields['rho_field']
        kf_e = fields['kf_electron']
        kf_n = fields['kf_neutron']
        
        # Calculate energy density and pressure for strange matter
        energy_density, pressure = self._calculate_strange_energy_pressure_rmf(
            sigma, omega, rho, kf_e, kf_n, baryon_density)
        
        return {
            'baryon_density': baryon_density,
            'pressure': pressure,
            'energy_density': energy_density,
            'sigma_field': sigma,
            'omega_field': omega,
            'rho_field': rho,
            'kf_electron': kf_e,
            'kf_neutron': kf_n,
            'converged': True
        }

    def _calculate_atomic_energy_pressure(self, sigma: float, omega: float, rho: float,
                                        kf_e: float, kf_n: float, baryon_density: float) -> tuple:
        """
        Calculate energy density and pressure for atomic matter (n, p, e, μ only).
        
        Args:
            sigma, omega, rho: Meson field values
            kf_e, kf_n: Electron and neutron Fermi momenta
            baryon_density: Baryonic number density in fm⁻³
            
        Returns:
            (energy_density, pressure) in MeV/fm³
        """
        rm = self.get_parameter('reference_mass', NEUTRON_MASS_MEV)
        
        # Meson field contributions
        energy_meson, pressure_meson = self._calculate_meson_field_contributions(sigma, omega, rho)
        
        # Nucleon contributions (only neutron and proton)
        energy_nucleon, pressure_nucleon = self._calculate_nucleon_contributions(
            sigma, omega, rho, baryon_density)
        
        # Lepton contributions  
        energy_lepton, pressure_lepton = self._calculate_lepton_contributions(kf_e)
        
        # Total energy and pressure (in MeV/fm³)
        total_energy = energy_meson + energy_nucleon + energy_lepton
        total_pressure = pressure_meson + pressure_nucleon + pressure_lepton
        
        return total_energy, total_pressure

    def _calculate_strange_energy_pressure_rmf(self, sigma: float, omega: float, rho: float,
                                             kf_e: float, kf_n: float, baryon_density: float) -> tuple:
        """
        Calculate energy density and pressure for strange matter using RMF (full octet).
        
        Args:
            sigma, omega, rho: Meson field values
            kf_e, kf_n: Electron and neutron Fermi momenta
            baryon_density: Baryonic number density in fm⁻³
            
        Returns:
            (energy_density, pressure) in MeV/fm³
        """
        # Meson field contributions
        energy_meson, pressure_meson = self._calculate_meson_field_contributions(sigma, omega, rho)
        
        # Full baryon octet contributions
        energy_baryon, pressure_baryon = self._calculate_full_octet_contributions(
            sigma, omega, rho, baryon_density)
        
        # Lepton contributions
        energy_lepton, pressure_lepton = self._calculate_lepton_contributions(kf_e)
        
        # Total energy and pressure (in MeV/fm³)
        total_energy = energy_meson + energy_baryon + energy_lepton
        total_pressure = pressure_meson + pressure_baryon + pressure_lepton
        
        return total_energy, total_pressure

    def compute(self, baryon_density: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Main computation method for multiple baryonic densities.
        
        Args:
            baryon_density: Baryonic number density in fm⁻³
        """
        if not self.initialized:
            if not self.initialize():
                raise RuntimeError("Failed to initialize EOS")
        
        if baryon_density is None:
            # Create default baryon density grid (fm⁻³)
            # Nuclear saturation density is approximately 0.16 fm⁻³
            rho_min = 0.01   # 0.01 fm⁻³ (about 0.06 times nuclear density)
            rho_max = 1.0    # 1.0 fm⁻³ (about 6 times nuclear density)
            baryon_density = np.logspace(np.log10(rho_min), np.log10(rho_max), 50)
        
        if np.isscalar(baryon_density):
            baryon_density = np.array([baryon_density])
        
        results = {
            'baryon_density': baryon_density,
            'pressure': np.zeros_like(baryon_density),
            'energy_density': np.zeros_like(baryon_density),
            'converged': np.zeros_like(baryon_density, dtype=bool)
        }
        
        for i, rho_b in enumerate(baryon_density):
            eos_result = self.compute_eos(rho_b)
            results['pressure'][i] = eos_result['pressure']
            results['energy_density'][i] = eos_result['energy_density']
            results['converged'][i] = eos_result['converged']
        
        self.save_results(results)
        return results

    def _set_default_eos_parameters(self):
        """Set default EOS parameters for RMF calculations."""
        
        # Default RMF parameters (GM1-like values)
        default_params = {
            # Reference mass (nucleon mass in MeV)
            'reference_mass': NEUTRON_MASS_MEV,
            
            # Meson-nucleon coupling constants
            'coupling_sigma': 10.217,    # g_s - sigma meson coupling
            'coupling_omega': 12.868,    # g_v - omega meson coupling  
            'coupling_rho': 4.474,       # g_r - rho meson coupling
            
            # Meson masses (MeV)
            'meson_sigma_mass': 550.0,   # m_s - sigma meson mass
            'meson_omega_mass': 783.0,   # m_v - omega meson mass
            'meson_rho_mass': 770.0,     # m_r - rho meson mass
            
            # Nonlinear coupling parameters
            'nonlinear_k': 0.0,          # Cubic sigma self-coupling
            'nonlinear_lambda': 0.0,     # Quartic sigma self-coupling
            'csi_parameter': 0.0,        # Quartic omega self-coupling
            
            # Hyperon coupling ratios (for strange stars)
            'hyperon_sigma_coupling': 0.7,   # x_s = g_s^Y / g_s^N
            'hyperon_omega_coupling': 0.783, # x_v = g_v^Y / g_v^N
            'hyperon_rho_coupling': 1.0,     # x_r = g_r^Y / g_r^N
            
            # Numerical parameters
            'tolerance': 1e-8,           # Convergence tolerance
            'max_iterations': 200,       # Maximum iterations for field equations
            
            # Baryon density range for calculations (fm⁻³)
            'baryon_density_min': 0.01,  # Minimum baryon density
            'baryon_density_max': 1.0,   # Maximum baryon density
            
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
        
        # Update with parametrization-specific values if available
        if self.parametrization:
            try:
                param_values = get_parametrization_parameters(self.parametrization)
                default_params.update(param_values)
                
                if self.verbose:
                    print(f"Loaded {self.parametrization.value} parametrization")
                    
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Could not load parametrization {self.parametrization}: {e}")
                    print("Using default parameters")
        
        # Set the parameters
        self.parameters.update(default_params)
        
        if self.verbose:
            print(f"Initialized EOS with {len(default_params)} parameters")
            print("Key parameters:")
            print(f"  Reference mass: {self.parameters['reference_mass']} MeV")
            print(f"  Sigma coupling: {self.parameters['coupling_sigma']}")
            print(f"  Omega coupling: {self.parameters['coupling_omega']}")
            print(f"  Rho coupling: {self.parameters['coupling_rho']}")
            print(f"  Baryon density range: {self.parameters['baryon_density_min']:.3f} - {self.parameters['baryon_density_max']:.3f} fm⁻³")

    def initialize(self) -> bool:
        """
        Initialize the EOS for calculations.
        
        This method performs any necessary setup before EOS computations.
        """
        if self.initialized:
            return True
        
        try:
            # Validate parameters
            required_params = [
                'reference_mass', 'coupling_sigma', 'coupling_omega', 'coupling_rho',
                'meson_sigma_mass', 'meson_omega_mass', 'meson_rho_mass'
            ]
            
            for param in required_params:
                if param not in self.parameters:
                    raise ValueError(f"Missing required parameter: {param}")
                if self.parameters[param] <= 0:
                    raise ValueError(f"Parameter {param} must be positive")
            
            # Additional validation for specific EOS types
            if self.is_strange_star():
                hyperon_params = ['hyperon_sigma_coupling', 'hyperon_omega_coupling', 'hyperon_rho_coupling']
                for param in hyperon_params:
                    if param not in self.parameters:
                        raise ValueError(f"Missing hyperon parameter for strange star: {param}")
            
            if self.is_magnetic_star():
                if 'magnetic_field' not in self.parameters:
                    self.parameters['magnetic_field'] = 0.0
                    
            # Validate particle data
            if not hasattr(self, 'baryon_list') or len(self.baryon_list) == 0:
                raise RuntimeError("Particle data not properly initialized")
            
            # Set initialization flag
            self.initialized = True
            
            if self.verbose:
                print(f"✓ EOS '{self.name}' initialized successfully")
                print(f"  Type: {self.eos_type.value}")
                print(f"  Particles: {self.n_baryons} baryons, {self.n_leptons} leptons")
                if self.parametrization:
                    print(f"  Parametrization: {self.parametrization.value}")
            
            return True
            
        except Exception as e:
            if self.verbose:
                print(f"✗ EOS initialization failed: {e}")
            self.initialized = False
            return False

    def get_parameter(self, name: str, default: Any = None) -> Any:
        """Get parameter value by name."""
        return self.parameters.get(name, default)

    def set_parameters(self, new_params: Dict[str, Any]):
        """Set multiple parameters."""
        self.parameters.update(new_params)

    def is_strange_star(self) -> bool:
        """Check if this is a strange star EOS."""
        return self.eos_type in [EOSType.STRANGE_STAR, EOSType.MAGNETIC_STRANGE_STAR,
                                EOSType.LSV_STRANGE_STAR, EOSType.NLEM_STRANGE_STAR]

    def is_magnetic_star(self) -> bool:
        """Check if this EOS includes magnetic effects."""
        return self.eos_type in [EOSType.MAGNETIC_ATOMIC_STAR, EOSType.MAGNETIC_STRANGE_STAR]

    def requires_lsv(self) -> bool:
        """Check if this EOS requires LSV corrections."""
        return self.eos_type in [EOSType.LSV_ATOMIC_STAR, EOSType.LSV_STRANGE_STAR]

    def requires_nlem(self) -> bool:
        """Check if this EOS requires NLEM corrections."""
        return self.eos_type in [EOSType.NLEM_ATOMIC_STAR, EOSType.NLEM_STRANGE_STAR]

    def get_info(self) -> Dict[str, Any]:
        """Get general information about this EOS."""
        return {
            'name': self.name,
            'eos_type': self.eos_type.value,
            'parametrization': self.parametrization.value if self.parametrization else None,
            'n_baryons': self.n_baryons,
            'n_leptons': self.n_leptons,
            'initialized': self.initialized,
            'lsv_type': self.lsv_type.value if self.lsv_type else None,
            'nlem_type': self.nlem_type.value if self.nlem_type else None
        }

    def get_parametrization_info(self) -> Dict[str, Any]:
        """Get information about the current parametrization."""
        if self.parametrization:
            try:
                return get_parametrization_info(self.parametrization)
            except:
                return {
                    'description': f'{self.parametrization.value} parametrization',
                    'reference': 'No reference available',
                    'type': 'RMF'
                }
        else:
            return {
                'description': 'Default parameters',
                'reference': 'Built-in defaults',
                'type': 'RMF'
            }

    def save_results(self, results: Dict[str, Any], filename: str = None):
        """Save EOS results to file."""
        if filename is None:
            filename = f"{self.name}_eos_results.npz"
        
        try:
            np.savez(filename, **results)
            if self.verbose:
                print(f"Results saved to {filename}")
        except Exception as e:
            if self.verbose:
                print(f"Failed to save results: {e}")

    def _sigma_integral(self, kf: float, m_eff: float) -> float:
        """
        Calculate the sigma field source integral.
        
        This is the scalar density: ∫ d³k/(2π)³ m_eff/E_k = m_eff * density_s
        """
        if kf <= 0 or m_eff <= 0:
            return 0.0
        
        # Scalar density integral: m_eff ∫₀^kf k²dk/(2π)³ 1/√(k²+m_eff²)
        def integrand(k):
            if k**2 + m_eff**2 <= 0:
                return 0.0
            return k**2 / np.sqrt(k**2 + m_eff**2)
        
        try:
            integral_result, _ = quad(integrand, 0, kf)
            return m_eff * integral_result / self.pi2
        except:
            # Fallback approximation for numerical issues
            if kf >> m_eff:  # Ultra-relativistic limit
                return m_eff * kf**3 / (3.0 * self.pi2) * (1.0 - 1.5 * m_eff**2 / kf**2)
            else:  # Non-relativistic limit  
                return m_eff**2 * kf / (2.0 * self.pi2)

    def _calculate_lepton_contributions(self, kf_e: float) -> tuple:
        """Calculate energy and pressure contributions from leptons (electrons and muons)."""
        rm = self.get_parameter('reference_mass', NEUTRON_MASS_MEV)
        
        # Electron contribution
        m_e = self.lepton_masses[0] / rm  # Electron mass in units of reference mass
        
        if kf_e > 0:
            # Electron energy and pressure integrals
            def electron_energy_integrand(k):
                return k**2 * np.sqrt(k**2 + m_e**2) / self.pi2
            
            def electron_pressure_integrand(k):
                return k**4 / (3.0 * np.sqrt(k**2 + m_e**2)) / self.pi2
            
            try:
                energy_electron, _ = quad(electron_energy_integrand, 0, kf_e)
                pressure_electron, _ = quad(electron_pressure_integrand, 0, kf_e)
            except:
                # Fallback to approximation
                if kf_e >> m_e:  # Ultra-relativistic limit
                    energy_electron = kf_e**4 / (4.0 * self.pi2)
                    pressure_electron = kf_e**4 / (12.0 * self.pi2)
                else:  # Non-relativistic limit
                    energy_electron = m_e * kf_e**3 / (3.0 * self.pi2)
                    pressure_electron = kf_e**5 / (15.0 * self.pi2 * m_e)
        else:
            energy_electron = 0.0
            pressure_electron = 0.0
        
        # Muon contribution (if above threshold)
        m_mu = self.lepton_masses[1] / rm  # Muon mass in units of reference mass
        kf_mu_squared = kf_e**2 + m_e**2 - m_mu**2
        
        if kf_mu_squared > 0:
            kf_mu = np.sqrt(kf_mu_squared)
            
            def muon_energy_integrand(k):
                return k**2 * np.sqrt(k**2 + m_mu**2) / self.pi2
            
            def muon_pressure_integrand(k):
                return k**4 / (3.0 * np.sqrt(k**2 + m_mu**2)) / self.pi2
            
            try:
                energy_muon, _ = quad(muon_energy_integrand, 0, kf_mu)
                pressure_muon, _ = quad(muon_pressure_integrand, 0, kf_mu)
            except:
                # Fallback to approximation
                if kf_mu >> m_mu:  # Ultra-relativistic limit
                    energy_muon = kf_mu**4 / (4.0 * self.pi2)
                    pressure_muon = kf_mu**4 / (12.0 * self.pi2)
                else:  # Non-relativistic limit
                    energy_muon = m_mu * kf_mu**3 / (3.0 * self.pi2)
                    pressure_muon = kf_mu**5 / (15.0 * self.pi2 * m_mu)
        else:
            energy_muon = 0.0
            pressure_muon = 0.0
        
        total_energy = energy_electron + energy_muon
        total_pressure = pressure_electron + pressure_muon
        
        return total_energy, total_pressure

    def _calculate_meson_field_contributions(self, sigma: float, omega: float, rho: float) -> tuple:
        """Calculate energy density and pressure contributions from meson fields."""
        # Get parameters
        rm = self.get_parameter('reference_mass', NEUTRON_MASS_MEV)
        g_s = self.get_parameter('coupling_sigma', 10.0)
        g_v = self.get_parameter('coupling_omega', 12.0)
        g_r = self.get_parameter('coupling_rho', 4.5)
        
        m_s = self.get_parameter('meson_sigma_mass', 550.0) / rm
        m_v = self.get_parameter('meson_omega_mass', 783.0) / rm
        m_r = self.get_parameter('meson_rho_mass', 770.0) / rm
        
        # Nonlinear parameters
        k = self.get_parameter('nonlinear_k', 0.0)
        lam = self.get_parameter('nonlinear_lambda', 0.0)
        csi = self.get_parameter('csi_parameter', 0.0)
        
        # Meson field energy contributions
        energy_sigma = 0.5 * m_s**2 * sigma**2
        energy_omega = 0.5 * m_v**2 * omega**2
        energy_rho = 0.5 * m_r**2 * rho**2
        
        # Nonlinear contributions
        energy_sigma_nl = k/(6.0) * sigma**3 + lam/(24.0) * sigma**4
        energy_omega_nl = csi * g_v**4/(24.0) * omega**4/g_v**2
        
        total_energy = energy_sigma + energy_omega + energy_rho + energy_sigma_nl + energy_omega_nl
        
        # Pressure contributions (with opposite sign for fields)
        pressure_meson = -total_energy
        
        return total_energy, pressure_meson

    def _calculate_nucleon_contributions(self, sigma: float, omega: float, rho: float, 
                                       baryon_density: float) -> tuple:
        """Calculate nucleon energy and pressure contributions."""
        rm = self.get_parameter('reference_mass', NEUTRON_MASS_MEV)
        
        total_energy = 0.0
        total_pressure = 0.0
        
        # Only nucleons (neutron and proton) for atomic matter
        for i in range(2):  # nucleons only
            # Effective mass
            m_eff = 1.0 - sigma  # In units of reference mass
            
            # This is a simplified calculation - would need full field solution
            # For now, use approximate contribution
            kf_approx = (3.0 * self.pi2 * baryon_density * 0.5)**(1/3)  # Approximate
            
            if kf_approx > 0:
                # Energy density integral
                def energy_integrand(k):
                    return k**2 * np.sqrt(k**2 + m_eff**2)
                
                def pressure_integrand(k):
                    return k**4 / (3.0 * np.sqrt(k**2 + m_eff**2))
                
                try:
                    energy_i, _ = quad(energy_integrand, 0, kf_approx)
                    pressure_i, _ = quad(pressure_integrand, 0, kf_approx)
                    
                    total_energy += energy_i / self.pi2
                    total_pressure += pressure_i / self.pi2
                except:
                    # Fallback
                    if kf_approx >> m_eff:
                        total_energy += kf_approx**4 / (4.0 * self.pi2)
                        total_pressure += kf_approx**4 / (12.0 * self.pi2)
                    else:
                        total_energy += m_eff * kf_approx**3 / (3.0 * self.pi2)
                        total_pressure += kf_approx**5 / (15.0 * self.pi2 * m_eff)
        
        return total_energy, total_pressure

    def _calculate_full_octet_contributions(self, sigma: float, omega: float, rho: float,
                                          baryon_density: float) -> tuple:
        """Calculate contributions from full baryon octet."""
        # This is similar to nucleon contributions but includes all 8 baryons
        # For now, use simplified version
        return self._calculate_nucleon_contributions(sigma, omega, rho, baryon_density)
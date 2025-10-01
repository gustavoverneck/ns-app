from .base import BaseEOS

class StrangeStarEOS(BaseEOS):
    """Strange quark star EOS."""
    
    def solve_field_equations(self, baryon_density: float) -> Dict[str, float]:
        """Simple atomic star without RMF fields - just beta equilibrium."""
        
        # Reference mass for normalization
        rm = NEUTRON_MASS_MEV  # MeV
        hc = 197.327  # MeV⋅fm
        
        # Normalized masses (dimensionless)
        mn = NEUTRON_MASS_MEV / rm  # = 1.0
        mp = PROTON_MASS_MEV / rm   # ≈ 0.999
        me = ELECTRON_MASS_MEV / rm # ≈ 0.0005
        
        # Beta equilibrium: μ_n = μ_p + μ_e
        # For charge neutrality: n_p = n_e
        # For baryon conservation: n_n + n_p = baryon_density
        
        # Normalize baryon density: ρ* = ρ / (rm/ℏc)³
        # This makes density dimensionless in natural units
        rho_normalized = baryon_density * (hc / rm)**3
        
        # Beta equilibrium relations (all in normalized units)
        kf_n = (3 * np.pi**2 * rho_normalized)**(1/3)  # Normalized neutron Fermi momentum
        
        # Proton Fermi momentum from beta equilibrium (normalized)
        kf_p = (kf_n**2 + mn**2 - mp**2) / (2 * np.sqrt(kf_n**2 + mn**2))
        
        # Electron Fermi momentum (charge neutrality, normalized)
        kf_e = kf_p
        
        return {
            'kf_electron': kf_e,
            'kf_neutron': kf_n,
            'kf_proton': kf_p,
            'sigma_field': 0.0,  # No RMF fields
            'omega_field': 0.0,
            'rho_field': 0.0,
            'converged': True
        }
    
    def compute_eos(self, baryon_density: float) -> Dict[str, float]:
        """Calculate eos for given solution."""
        if self.verbose:
            print(f"Computing atomic EOS (n, p, e, μ) for baryon density {baryon_density:.4f} fm⁻³")
        
        # Solve RMF field equations for atomic matter (only nucleons)
        solution = self.solve_field_equations(baryon_density)
        
        if not solution['converged']:
            return {
                'baryon_density': baryon_density,
                'pressure': 0.0,
                'energy_density': 0.0,
                'converged': False
            }
        
        # Extract field values
        sigma = solution['sigma_field']
        omega = solution['omega_field']
        rho = solution['rho_field']
        kf_e = solution['kf_electron']
        kf_n = solution['kf_neutron']
        
        # Calculate energy density and pressure for atomic matter
        energy_density = self.compute_energy_density(
            sigma, omega, rho, kf_e, kf_n, baryon_density)
        pressure = self.compute_pressure(
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
    
    def compute_energy_density(self, sigma, omega, rho, kf_e, kf_n, baryon_density) -> float:
        """Calculate energy density using normalized units."""
        
        # Reference mass and constants
        rm = NEUTRON_MASS_MEV  # MeV
        hc = 197.327  # MeV⋅fm
        pi2 = np.pi**2
        
        # Normalized masses (dimensionless)
        mn = NEUTRON_MASS_MEV / rm  # = 1.0
        mp = PROTON_MASS_MEV / rm   # ≈ 0.999
        me = ELECTRON_MASS_MEV / rm # ≈ 0.0005
        
        # Get normalized proton Fermi momentum (consistent with solve_field_equations)
        kf_p = (kf_n**2 + mn**2 - mp**2) / (2 * np.sqrt(kf_n**2 + mn**2))
        
        # Fermi momenta and masses arrays (all normalized)
        kf = [kf_n, kf_p, kf_e]
        m = [mn, mp, me]
        
        # Calculate energy density in normalized units
        ener = 0.0
        for i in range(3):  # neutron, proton, electron
            if kf[i] > 0:
                sqrt_term = np.sqrt(kf[i]**2 + m[i]**2)
                ener += (1/(8*pi2)) * (
                    kf[i] * sqrt_term * (2*kf[i]**2 + m[i]**2) - 
                    m[i]**4 * np.log((kf[i] + sqrt_term) / m[i])
                )
        
        # Convert from normalized units to MeV/fm³
        # ε* (dimensionless) → ε (MeV/fm³) = ε* × rm⁴/ℏc³
        return ener * rm**4 / hc**3
    
    def compute_pressure(self, sigma, omega, rho, kf_e, kf_n, baryon_density) -> float:
        """Calculate pressure using normalized units."""
        
        # Reference mass and constants
        rm = NEUTRON_MASS_MEV  # MeV
        hc = 197.327  # MeV⋅fm
        pi2 = np.pi**2
        
        # Normalized masses (dimensionless)
        mn = NEUTRON_MASS_MEV / rm  # = 1.0
        mp = PROTON_MASS_MEV / rm   # ≈ 0.999
        me = ELECTRON_MASS_MEV / rm # ≈ 0.0005
        
        # Get normalized proton Fermi momentum (consistent with solve_field_equations)
        kf_p = (kf_n**2 + mn**2 - mp**2) / (2 * np.sqrt(kf_n**2 + mn**2))
        
        # Fermi momenta and masses arrays (all normalized)
        kf = [kf_n, kf_p, kf_e]
        m = [mn, mp, me]
        
        # Calculate pressure in normalized units
        pres = 0.0
        for i in range(3):  # neutron, proton, electron
            if kf[i] > 0:
                sqrt_term = np.sqrt(m[i]**2 + kf[i]**2)
                pres += (1/(24*pi2)) * (
                    (2*kf[i]**5 - kf[i]**3*m[i]**2 - 3*kf[i]*m[i]**4) / sqrt_term + 
                    3*m[i]**4 * np.log(abs((kf[i] + sqrt_term) / m[i]))
                )
        
        # Convert from normalized units to MeV/fm³
        # P* (dimensionless) → P (MeV/fm³) = P* × rm⁴/ℏc³
        return pres * rm**4 / hc**3

class MagneticStrangeStarEOS(StrangeStarEOS):
    """Magnetic strange quark star EOS."""
    pass

class LSVStrangeStarEOS(StrangeStarEOS):
    """LSV strange quark star EOS."""
    pass

class NLEMStrangeStarEOS(StrangeStarEOS):
    """NLEM strange quark star EOS."""
    pass
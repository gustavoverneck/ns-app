from typing import Dict, Tuple
from .base import BaseEOS
from ..constants import HBARC_MEV_FM, PI
import numpy as np

class StrangeStarEOS(BaseEOS):
    """
    Strange star EOS using Relativistic Mean Field theory with baryon octet.
    
    Based on the Walecka model with nonlinear meson self-interactions.
    Includes full baryon octet: n, p, Λ, Σ⁻, Σ⁰, Σ⁺, Ξ⁻, Ξ⁰
    Plus leptons: e⁻, μ⁻
    
    Uses the parametrization from Glendenning (GM1, GM3, etc.)
    """
    
    def solve_field_equations(self, baryon_density: float) -> Dict[str, float]:
        """
        Solve the coupled RMF field equations for strange matter at T=0.
        
        Initial conditions based on mean-field theory expectations and
        typical convergence behavior from RMF literature.
        """
        
        if self.verbose:
            print(f"\nSolving RMF equations for ρ_B = {baryon_density:.6f} fm⁻³")
        
        # Normalize density to saturation density
        rho_norm = baryon_density / self.rho_sat
        
        # ============================================
        # IMPROVED INITIAL CONDITIONS
        # ============================================
        
        x0 = np.zeros(5)
        
        # 1. Electron angle: Use charge neutrality estimate
        # For nuclear matter, typical proton fraction y_p ≈ 0.03-0.10 at high density
        # μ_e ≈ 100-150 MeV at nuclear saturation
        y_p_estimate = 0.05 * (1.0 + 0.1 * (rho_norm - 1.0))  # Increases slightly with density
        y_p_estimate = np.clip(y_p_estimate, 0.01, 0.15)
        
        kf_e_estimate = (3.0 * PI**2 * baryon_density * y_p_estimate)**(1.0/3.0)
        me_norm = self.lepton_masses[0]  # Already normalized
        
        # Electron angle: sin(θ) = kf / sqrt(kf² + m²)
        sin_theta = kf_e_estimate / np.sqrt(kf_e_estimate**2 + me_norm**2 + 1e-10)
        x0[0] = np.arcsin(np.clip(sin_theta, 0.0, 0.999))
        
        # 2. Omega field: From baryon number conservation
        # mω²·ω ≈ gω·ρ_B  =>  ω ≈ gω·ρ_B / mω²
        # In normalized units: ω_norm = gω·ρ_B / (mω²·rm²)
        omega_estimate = self.gw * baryon_density / (self.mw**2 * self.rm**2)
        x0[1] = np.sqrt(max(omega_estimate, 1e-10))
        
        # 3. Rho field: From isospin asymmetry
        # For neutron-rich matter: I₃ ≈ (n-p)/2 ≈ ρ_B/2 initially
        # mρ²·ρ ≈ gρ·I₃  =>  ρ ≈ gρ·ρ_B·0.5 / mρ²
        rho_estimate = self.gr * baryon_density * 0.5 / (self.mr**2 * self.rm**2)
        x0[2] = np.sqrt(max(rho_estimate, 1e-10))
        
        # 4. Sigma field: From effective mass reduction
        # At saturation: m*/m ≈ 0.6-0.8
        # σ ≈ (1 - m*/m) × m_N × ρ_s/ρ_B
        # Typical scalar density: ρ_s ≈ 0.6-0.8 × ρ_B
        # For high density, expect σ ≈ 200-400 MeV at ρ_sat
        sigma_sat = (1.0 - self.effective_mass_ratio) * self.rm * 0.7  # MeV
        sigma_estimate = sigma_sat * np.sqrt(rho_norm)  # Scales with sqrt(density)
        x0[3] = sigma_estimate / (self.ms * self.rm)  # Normalize
        
        # 5. Proton Fermi momentum
        # Use charge neutrality: n_p ≈ n_e + n_μ
        # At high density with muons: y_p ≈ 0.1-0.15
        kf_p_estimate = (3.0 * PI**2 * baryon_density * y_p_estimate)**(1.0/3.0)
        x0[4] = kf_p_estimate / self.rm  # Normalize
        
        # ============================================
        # ALTERNATIVE: Density-dependent scaling
        # ============================================
        # Adjust initial guess based on density regime
        
        if rho_norm < 0.5:
            # Low density: reduce field strengths
            x0[1] *= 0.5  # Smaller omega
            x0[2] *= 0.3  # Smaller rho
            x0[3] *= 0.7  # Smaller sigma
        elif rho_norm > 3.0:
            # High density: expect hyperon onset, adjust fields
            x0[1] *= 1.2  # Stronger omega (more baryons)
            x0[3] *= 1.1  # Slightly stronger sigma
            # Increase proton fraction estimate for hyperons
            x0[4] *= 1.3
        
        if self.verbose:
            print(f"Initial guess for ρ_B = {baryon_density:.4f} fm⁻³ (ρ/ρ₀ = {rho_norm:.2f}):")
            print(f"  x[0] (θ_e):     {x0[0]:.6f} (y_p ≈ {y_p_estimate:.3f})")
            print(f"  x[1] (ω_norm):  {x0[1]:.6e}")
            print(f"  x[2] (ρ_norm):  {x0[2]:.6e}")
            print(f"  x[3] (σ/m_σ):   {x0[3]:.6f}")
            print(f"  x[4] (k_fp/m):  {x0[4]:.6f}")
        
        # ============================================
        # Solve using Broyden's method
        # ============================================
        
        def equations(x):
            return self._field_equations_residual(x, baryon_density)
        
        try:
            # Broyden1 with improved settings
            solution = self.root_finder(
                equations, x0,
                f_tol=self.tolerance,       # Function tolerance
                maxiter=self.max_iterations,
                line_search='wolfe',         # Use line search for robustness
                verbose=False
            )
            
            converged = True
            
            # Check if solution is physical
            residual = equations(solution)
            max_residual = np.max(np.abs(residual))
            
            if max_residual > self.tolerance * 10:
                converged = False
                if self.verbose:
                    print(f"  Warning: Large residual {max_residual:.2e} > tolerance")
                
        except Exception as e:
            if self.verbose:
                print(f"  Error in root finder: {e}")
            converged = False
            solution = x0
        
        # Unpack solution and compute all quantities
        result = self._unpack_solution(solution, baryon_density, converged)
        
        if self.verbose:
            if converged:
                print(f"  ✓ Converged!")
                print(f"    σ = {result['sigma_field']:.3f} MeV")
                print(f"    ω = {result['omega_field']:.3f} MeV")
                print(f"    ρ = {result['rho_field']:.6f} MeV")
                print(f"    y_p = {result['proton_fraction']:.4f}")
                print(f"    y_Y = {result['hyperon_fraction']:.4f}")
            else:
                print(f"  ✗ Did not converge")
        
        return result
    
    def _field_equations_residual(self, x: np.ndarray, rho_baryon: float) -> np.ndarray:
        """
        Compute residuals for the RMF field equations.
        
        System of 5 equations:
        F1: Beta equilibrium for muon
        F2: Beta equilibrium for electron  
        F3: Charge neutrality
        F4: Sigma field equation (scalar density)
        F5: Baryon number conservation
        
        Args:
            x: Solution vector [theta_e, omega, rho, sigma_norm, kf_p]
            rho_baryon: Total baryon density (fm⁻³)
            
        Returns:
            Residual vector F(x) = 0
        """
        
        # Unpack variables
        theta_e = x[0]  # Electron angle parameter
        omega = x[1]    # Omega field (normalized)
        rho = x[2]      # Rho field (normalized)
        sigma_norm = x[3]  # Sigma field / m_sigma
        kf_p = x[4]     # Proton Fermi momentum (normalized)
        
        # Convert normalized fields to physical units
        sigma = sigma_norm * self.ms * self.rm  # MeV
        omega_field = omega * self.mw * self.rm  # MeV
        rho_field = rho * self.mr * self.rm      # MeV
        
        # Compute Fermi momenta for leptons from theta
        me_eff = self.lepton_masses[0] * self.rm  # Electron mass in MeV
        mmu_eff = self.lepton_masses[1] * self.rm # Muon mass in MeV
        
        kf_e = me_eff * np.tan(theta_e) if abs(np.cos(theta_e)) > 1e-10 else 0.0
        
        # Chemical potentials for leptons
        mu_e = np.sqrt(kf_e**2 + me_eff**2)
        mu_mu = np.sqrt(max(mu_e**2 - mmu_eff**2, 0.0))  # Muon threshold
        
        if mu_mu > mmu_eff:
            kf_mu = np.sqrt(mu_mu**2 - mmu_eff**2)
        else:
            kf_mu = 0.0
        
        # Compute effective baryon masses (nucleons)
        m_eff_n = self.baryon_masses[0] * self.rm - self.gs * sigma  # Neutron
        m_eff_p = self.baryon_masses[1] * self.rm - self.gs * sigma  # Proton
        
        # Compute effective baryon masses (hyperons with coupling ratios)
        m_eff_hyperons = []
        for i in range(2, self.n_baryons):
            m_eff_hyperons.append(
                self.baryon_masses[i] * self.rm - self.gs_hyperon * sigma
            )
        
        # Chemical potentials for baryons
        # μᵢ = √(kfᵢ² + m*ᵢ²) + gω·ω + gρ·ρ·I₃ᵢ
        
        # Neutron chemical potential (reference)
        kf_n_guess = (3 * PI**2 * rho_baryon * 0.9)**(1/3)  # Guess ~90% neutrons
        mu_n = np.sqrt(kf_n_guess**2 + m_eff_n**2) + self.gw * omega_field - 0.5 * self.gr * rho_field
        
        # Proton chemical potential
        mu_p = np.sqrt(kf_p**2 + m_eff_p**2) + self.gw * omega_field + 0.5 * self.gr * rho_field
        
        # Compute Fermi momenta for all baryons using beta equilibrium
        # μₙ = μₚ + μₑ (beta equilibrium)
        # μᵢ = μₙ - Qᵢ·μₑ (for hyperon i with charge Qᵢ)
        
        kf_baryons = np.zeros(self.n_baryons)
        kf_baryons[1] = kf_p  # Proton (given)
        
        # Neutron from beta equilibrium
        mu_n_target = mu_p + mu_e
        kf_n = np.sqrt(max((mu_n_target - self.gw * omega_field + 0.5 * self.gr * rho_field)**2 - m_eff_n**2, 0.0))
        kf_baryons[0] = kf_n
        
        # Hyperons
        for i in range(2, self.n_baryons):
            charge = self.baryon_charges[i]
            isospin = self.baryon_isospin_3[i]
            
            mu_hyperon = mu_n - charge * mu_e
            
            # Remove fields to get kinetic part
            mu_kin = mu_hyperon - self.gw_hyperon * omega_field - isospin * self.gr_hyperon * rho_field
            
            kf_i = np.sqrt(max(mu_kin**2 - m_eff_hyperons[i-2]**2, 0.0))
            kf_baryons[i] = kf_i
        
        # Compute densities
        rho_baryons = np.array([kf**3 / (3 * PI**2) for kf in kf_baryons])
        rho_e = kf_e**3 / (3 * PI**2)
        rho_mu = kf_mu**3 / (3 * PI**2) if kf_mu > 0 else 0.0
        
        # Compute scalar densities for sigma field equation
        scalar_density_nucleons = 0.0
        for i in range(2):  # Neutron and proton
            if kf_baryons[i] > 0:
                m_eff = self.baryon_masses[i] * self.rm - self.gs * sigma
                scalar_density_nucleons += (m_eff / (2 * PI**2)) * (
                    kf_baryons[i] * np.sqrt(kf_baryons[i]**2 + m_eff**2) - 
                    m_eff**2 * np.log((kf_baryons[i] + np.sqrt(kf_baryons[i]**2 + m_eff**2)) / m_eff)
                )
        
        scalar_density_hyperons = 0.0
        for i in range(2, self.n_baryons):
            if kf_baryons[i] > 0:
                m_eff = m_eff_hyperons[i-2]
                scalar_density_hyperons += (m_eff / (2 * PI**2)) * (
                    kf_baryons[i] * np.sqrt(kf_baryons[i]**2 + m_eff**2) - 
                    m_eff**2 * np.log((kf_baryons[i] + np.sqrt(kf_baryons[i]**2 + m_eff**2)) / m_eff)
                )
        
        # Residuals (5 equations)
        F = np.zeros(5)
        
        # F[0]: Beta equilibrium (μₙ = μₚ + μₑ)
        F[0] = mu_n - mu_p - mu_e
        
        # F[1]: Sigma field equation
        # mσ²·σ = gσ·ρₛ + nonlinear terms
        sigma_source = self.gs * scalar_density_nucleons + self.gs_hyperon * scalar_density_hyperons
        sigma_source += self.k_nl * sigma**2 + self.lambda_nl * sigma**3  # Nonlinear
        F[1] = self.ms**2 * self.rm**2 * sigma - sigma_source
        
        # F[2]: Omega field equation  
        # mω²·ω = gω·ρᴮ
        omega_source = self.gw * np.sum(rho_baryons[:2]) + self.gw_hyperon * np.sum(rho_baryons[2:])
        F[2] = self.mw**2 * self.rm**2 * omega_field - omega_source
        
        # F[3]: Rho field equation
        # mρ²·ρ = gρ·Σ I₃·ρᵢ
        rho_source = 0.0
        for i in range(2):
            rho_source += self.gr * self.baryon_isospin_3[i] * rho_baryons[i]
        for i in range(2, self.n_baryons):
            rho_source += self.gr_hyperon * self.baryon_isospin_3[i] * rho_baryons[i]
        F[3] = self.mr**2 * self.rm**2 * rho_field - rho_source
        
        # F[4]: Charge neutrality
        # Σ Qᵢ·ρᵢ = 0
        charge_density = np.sum(self.baryon_charges * rho_baryons) - rho_e - rho_mu
        F[4] = charge_density
        
        return F
    
    def _unpack_solution(self, x: np.ndarray, rho_baryon: float, converged: bool) -> Dict[str, float]:
        """
        Unpack solution vector and compute all physical quantities.
        
        Args:
            x: Solution vector
            rho_baryon: Baryon density
            converged: Whether solver converged
            
        Returns:
            Dictionary with all fields, Fermi momenta, fractions, etc.
        """
        
        # Unpack solution (same as in residual function)
        theta_e = x[0]
        omega = x[1]
        rho = x[2]
        sigma_norm = x[3]
        kf_p = x[4]
        
        # Physical fields
        sigma = sigma_norm * self.ms * self.rm
        omega_field = omega * self.mw * self.rm
        rho_field = rho * self.mr * self.rm
        
        # Lepton Fermi momenta
        me_eff = self.lepton_masses[0] * self.rm
        mmu_eff = self.lepton_masses[1] * self.rm
        kf_e = me_eff * np.tan(theta_e) if abs(np.cos(theta_e)) > 1e-10 else 0.0
        mu_e = np.sqrt(kf_e**2 + me_eff**2)
        mu_mu = np.sqrt(max(mu_e**2 - mmu_eff**2, 0.0))
        kf_mu = np.sqrt(mu_mu**2 - mmu_eff**2) if mu_mu > mmu_eff else 0.0
        
        # Reconstruct all baryon Fermi momenta (same logic as residual)
        m_eff_n = self.baryon_masses[0] * self.rm - self.gs * sigma
        m_eff_p = self.baryon_masses[1] * self.rm - self.gs * sigma
        
        mu_p = np.sqrt(kf_p**2 + m_eff_p**2) + self.gw * omega_field + 0.5 * self.gr * rho_field
        mu_n_target = mu_p + mu_e
        kf_n = np.sqrt(max((mu_n_target - self.gw * omega_field + 0.5 * self.gr * rho_field)**2 - m_eff_n**2, 0.0))
        
        kf_baryons = np.zeros(self.n_baryons)
        kf_baryons[0] = kf_n
        kf_baryons[1] = kf_p
        
        # Hyperons
        for i in range(2, self.n_baryons):
            charge = self.baryon_charges[i]
            isospin = self.baryon_isospin_3[i]
            m_eff_hyperon = self.baryon_masses[i] * self.rm - self.gs_hyperon * sigma
            
            mu_hyperon = mu_n_target - charge * mu_e
            mu_kin = mu_hyperon - self.gw_hyperon * omega_field - isospin * self.gr_hyperon * rho_field
            
            kf_i = np.sqrt(max(mu_kin**2 - m_eff_hyperon**2, 0.0))
            kf_baryons[i] = kf_i
        
        # Compute densities
        rho_baryons = kf_baryons**3 / (3 * PI**2)
        rho_total = np.sum(rho_baryons)
        
        # Fractions
        proton_fraction = rho_baryons[1] / rho_total if rho_total > 0 else 0.0
        hyperon_fraction = np.sum(rho_baryons[2:]) / rho_total if rho_total > 0 else 0.0
        
        return {
            'sigma_field': sigma,
            'omega_field': omega_field,
            'rho_field': rho_field,
            'kf_electron': kf_e,
            'kf_muon': kf_mu,
            'kf_baryons': kf_baryons,
            'rho_baryons': rho_baryons,
            'proton_fraction': proton_fraction,
            'hyperon_fraction': hyperon_fraction,
            'converged': converged
        }
    
    def compute_eos(self, baryon_density: float) -> Dict[str, float]:
        """
        Compute complete equation of state at given baryon density.
        
        Args:
            baryon_density: Baryon number density in fm⁻³
            
        Returns:
            Dictionary with pressure, energy density, and all other quantities
        """
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Computing Strange Star EOS at ρ_B = {baryon_density:.6f} fm⁻³")
            print(f"{'='*60}")
        
        # Solve field equations
        solution = self.solve_field_equations(baryon_density)
        
        if not solution['converged']:
            return {
                'baryon_density': baryon_density,
                'pressure': 0.0,
                'energy_density': 0.0,
                'converged': False
            }
        
        # Extract solution
        sigma = solution['sigma_field']
        omega = solution['omega_field']
        rho = solution['rho_field']
        kf_e = solution['kf_electron']
        kf_mu = solution['kf_muon']
        kf_baryons = solution['kf_baryons']
        
        # Compute thermodynamic quantities
        energy_density = self.compute_energy_density(
            sigma, omega, rho, kf_e, kf_mu, kf_baryons, baryon_density
        )
        pressure = self.compute_pressure(
            sigma, omega, rho, kf_e, kf_mu, kf_baryons, baryon_density
        )
        
        # Chemical potentials
        mu_n = self._compute_chemical_potential(0, kf_baryons[0], sigma, omega, rho)
        mu_e = np.sqrt(kf_e**2 + (self.lepton_masses[0] * self.rm)**2)
        
        if self.verbose:
            print(f"\nResults:")
            print(f"  Pressure: {pressure:.6e} MeV/fm³")
            print(f"  Energy density: {energy_density:.6e} MeV/fm³")
            print(f"  μₙ: {mu_n:.3f} MeV")
            print(f"  μₑ: {mu_e:.3f} MeV")
            print(f"{'='*60}\n")
        
        return {
            'baryon_density': baryon_density,
            'pressure': pressure,
            'energy_density': energy_density,
            'chemical_potential_neutron': mu_n,
            'chemical_potential_electron': mu_e,
            **solution  # Include all solution details
        }
    
    def _compute_chemical_potential(self, baryon_index: int, kf: float, 
                                   sigma: float, omega: float, rho: float) -> float:
        """Compute chemical potential for baryon species."""
        
        if baryon_index < 2:  # Nucleon
            m_eff = self.baryon_masses[baryon_index] * self.rm - self.gs * sigma
            gw_eff = self.gw
            gr_eff = self.gr
        else:  # Hyperon
            m_eff = self.baryon_masses[baryon_index] * self.rm - self.gs_hyperon * sigma
            gw_eff = self.gw_hyperon
            gr_eff = self.gr_hyperon
        
        mu = np.sqrt(kf**2 + m_eff**2) + gw_eff * omega + self.baryon_isospin_3[baryon_index] * gr_eff * rho
        return mu
    
    def compute_energy_density(self, sigma: float, omega: float, rho: float,
                              kf_e: float, kf_mu: float, kf_baryons: np.ndarray,
                              baryon_density: float) -> float:
        """
        Calculate total energy density including all contributions.
        
        ε = ε_kinetic + ε_mesons
        
        Args:
            sigma, omega, rho: Meson fields in MeV
            kf_e, kf_mu: Lepton Fermi momenta
            kf_baryons: Array of baryon Fermi momenta
            baryon_density: Total baryon density
            
        Returns:
            Energy density in MeV/fm³
        """
        
        energy = 0.0
        
        # Baryon kinetic energy (nucleons)
        for i in range(2):
            if kf_baryons[i] > 0:
                m_eff = self.baryon_masses[i] * self.rm - self.gs * sigma
                energy += self._fermi_integral_energy(kf_baryons[i], m_eff)
        
        # Baryon kinetic energy (hyperons)
        for i in range(2, self.n_baryons):
            if kf_baryons[i] > 0:
                m_eff = self.baryon_masses[i] * self.rm - self.gs_hyperon * sigma
                energy += self._fermi_integral_energy(kf_baryons[i], m_eff)
        
        # Lepton kinetic energy
        if kf_e > 0:
            me_eff = self.lepton_masses[0] * self.rm
            energy += self._fermi_integral_energy(kf_e, me_eff)
        if kf_mu > 0:
            mmu_eff = self.lepton_masses[1] * self.rm
            energy += self._fermi_integral_energy(kf_mu, mmu_eff)
        
        # Meson field energy
        energy += 0.5 * (self.ms * self.rm)**2 * sigma**2
        energy += 0.5 * (self.mw * self.rm)**2 * omega**2
        energy += 0.5 * (self.mr * self.rm)**2 * rho**2
        
        # Nonlinear meson self-interactions
        energy += (self.k_nl / 3.0) * sigma**3
        energy += (self.lambda_nl / 4.0) * sigma**4
        
        return energy
    
    def compute_pressure(self, sigma: float, omega: float, rho: float,
                        kf_e: float, kf_mu: float, kf_baryons: np.ndarray,
                        baryon_density: float) -> float:
        """
        Calculate total pressure.
        
        P = P_kinetic - U_mesons
        
        Args:
            sigma, omega, rho: Meson fields in MeV
            kf_e, kf_mu: Lepton Fermi momenta
            kf_baryons: Array of baryon Fermi momenta
            baryon_density: Total baryon density
            
        Returns:
            Pressure in MeV/fm³
        """
        
        pressure = 0.0
        
        # Baryon kinetic pressure (nucleons)
        for i in range(2):
            if kf_baryons[i] > 0:
                m_eff = self.baryon_masses[i] * self.rm - self.gs * sigma
                pressure += self._fermi_integral_pressure(kf_baryons[i], m_eff)
        
        # Baryon kinetic pressure (hyperons)
        for i in range(2, self.n_baryons):
            if kf_baryons[i] > 0:
                m_eff = self.baryon_masses[i] * self.rm - self.gs_hyperon * sigma
                pressure += self._fermi_integral_pressure(kf_baryons[i], m_eff)
        
        # Lepton kinetic pressure
        if kf_e > 0:
            me_eff = self.lepton_masses[0] * self.rm
            pressure += self._fermi_integral_pressure(kf_e, me_eff)
        if kf_mu > 0:
            mmu_eff = self.lepton_masses[1] * self.rm
            pressure += self._fermi_integral_pressure(kf_mu, mmu_eff)
        
        # Meson field contributions (negative)
        pressure -= 0.5 * (self.ms * self.rm)**2 * sigma**2
        pressure += 0.5 * (self.mw * self.rm)**2 * omega**2
        pressure += 0.5 * (self.mr * self.rm)**2 * rho**2
        
        # Nonlinear meson contributions
        pressure -= (self.k_nl / 3.0) * sigma**3
        pressure -= (self.lambda_nl / 4.0) * sigma**4
        
        return pressure
    
    def _fermi_integral_energy(self, kf: float, m_eff: float) -> float:
        """
        Compute kinetic energy density integral for fermions.
        
        ε_kin = (1/8π²)[kf·Ef·(2kf² + m*²) - m*⁴·ln((kf + Ef)/m*)]
        where Ef = √(kf² + m*²)
        """
        if kf <= 0:
            return 0.0
        
        ef = np.sqrt(kf**2 + m_eff**2)
        
        energy = (1.0 / (8.0 * PI**2)) * (
            kf * ef * (2.0 * kf**2 + m_eff**2) - 
            m_eff**4 * np.log((kf + ef) / (m_eff + 1e-10))
        )
        
        return energy
    
    def _fermi_integral_pressure(self, kf: float, m_eff: float) -> float:
        """
        Compute kinetic pressure integral for fermions.
        
        P_kin = (1/24π²)[(2kf⁵ - kf³·m*² - 3kf·m*⁴)/Ef + 3m*⁴·ln((kf + Ef)/m*)]
        where Ef = √(kf² + m*²)
        """
        if kf <= 0:
            return 0.0
        
        ef = np.sqrt(kf**2 + m_eff**2)
        
        pressure = (1.0 / (24.0 * PI**2)) * (
            (2.0 * kf**5 - kf**3 * m_eff**2 - 3.0 * kf * m_eff**4) / ef + 
            3.0 * m_eff**4 * np.log((kf + ef) / (m_eff + 1e-10))
        )
        
        return pressure


class MagneticStrangeStarEOS(StrangeStarEOS):
    """Magnetic strange quark star EOS with quantized Landau levels."""
    pass

class LSVStrangeStarEOS(StrangeStarEOS):
    """LSV strange quark star EOS with Lorentz symmetry violation."""
    pass

class NLEMStrangeStarEOS(StrangeStarEOS):
    """NLEM strange quark star EOS with nonlinear electrodynamics."""
    pass
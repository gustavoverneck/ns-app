"""
Tolman-Oppenheimer-Volkoff (TOV) equation solver for neutron star structure.

This module solves the TOV equations using the EOS from eos.py to calculate
neutron star mass-radius relationships and internal structure.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# Import constants from our constants module
from .constants import (
    # Physical constants
    G_SI, C_SI, M_SUN_KG, 
    # Unit conversions
    MEV_FM3_TO_PA, PA_TO_MEV_FM3,
    KM_TO_M, M_TO_KM,
    # Conversion factors
    HBARC_MEV_FM, PI
)

class TOVSolver:
    """
    Solver for the Tolman-Oppenheimer-Volkoff equations.
    
    The TOV equations describe the structure of spherically symmetric,
    static relativistic stars in hydrostatic equilibrium.
    """
    
    def __init__(self, eos, verbose: bool = False):
        """
        Initialize TOV solver with an equation of state.
        
        Args:
            eos: EOS object from eos.py with compute_eos(baryon_density) method
            verbose: Print detailed information during computation
        """
        self.eos = eos
        self.verbose = verbose
        
        # Use constants from constants.py
        self.G = G_SI  # m³/kg/s²
        self.c = C_SI  # m/s
        self.M_sun = M_SUN_KG  # kg
        self.km_to_m = KM_TO_M
        
        # Unit conversions from constants.py
        self.MeV_fm3_to_Pa = MEV_FM3_TO_PA
        self.Pa_to_MeV_fm3 = PA_TO_MEV_FM3
        
        # Geometric units conversion
        self.length_scale = self.G / self.c**2  # m/kg
        self.pressure_scale = self.c**4 / self.G  # Pa
        
        # Pre-computed EOS data
        self.eos_data = None
        self.pressure_interp = None
        self.energy_density_interp = None
        
    def prepare_eos_data(self, rho_min: float = 0.08, rho_max: float = 8.0, n_points: int = 200):
        """
        Pre-compute EOS data for interpolation during TOV integration.
        
        Args:
            rho_min: Minimum baryon density in fm⁻³
            rho_max: Maximum baryon density in fm⁻³  
            n_points: Number of grid points
        """
        if self.verbose:
            print(f"Preparing EOS data for TOV solver...")
            print(f"  Density range: {rho_min:.3f} - {rho_max:.3f} fm⁻³")
            print(f"  Grid points: {n_points}")
        
        # Create density grid - use better spacing for neutron star conditions
        # Nuclear density is ~0.16 fm⁻³, so we need to go much higher
        rho_low = np.linspace(rho_min, 0.5, n_points//4)
        rho_mid = np.linspace(0.5, 2.0, n_points//2)  
        rho_high = np.logspace(np.log10(2.0), np.log10(rho_max), n_points//4)
        
        baryon_densities = np.concatenate([rho_low[:-1], rho_mid[:-1], rho_high])
        baryon_densities = np.unique(baryon_densities)  # Remove duplicates
        
        # Compute EOS for all densities
        pressures = []
        energy_densities = []
        valid_densities = []
        failed_count = 0
        
        for i, rho_b in enumerate(baryon_densities):
            try:
                # Use the EOS's compute_eos method directly
                result = self.eos.compute_eos(rho_b)
                
                if result and result.get('converged', True):  # Default to True if not specified
                    P = result.get('pressure', 0.0)
                    eps = result.get('energy_density', 0.0)
                    
                    # Basic validity checks
                    if P > 0 and eps > 0:  # Both must be positive
                        pressures.append(P)
                        energy_densities.append(eps)
                        valid_densities.append(rho_b)
                    else:
                        failed_count += 1
                        if self.verbose and i < 5:
                            print(f"    Invalid EOS values at ρ={rho_b:.3f}: P={P:.2e}, ε={eps:.2e}")
                else:
                    failed_count += 1
                    if self.verbose and i < 5:
                        print(f"    EOS convergence failed at ρ={rho_b:.3f}")
                        
                if i % 20 == 0 and self.verbose:
                    print(f"    Progress: {i+1}/{len(baryon_densities)} ({(i+1)/len(baryon_densities)*100:.1f}%)")
                    if len(valid_densities) > 0:
                        print(f"      Last valid: ρ={rho_b:.3f} fm⁻³, P={pressures[-1]:.2f} MeV/fm³")
                    
            except Exception as e:
                failed_count += 1
                if self.verbose and i % 50 == 0:
                    print(f"    Warning: EOS failed at ρ = {rho_b:.4f} fm⁻³: {e}")
        
        if len(valid_densities) < 10:
            raise RuntimeError(f"Too few valid EOS points: {len(valid_densities)} (failed: {failed_count})")
        
        # Convert to arrays and sort by density
        valid_densities = np.array(valid_densities)
        valid_pressures = np.array(pressures)
        valid_energies = np.array(energy_densities)
        
        # Sort by density
        sort_idx = np.argsort(valid_densities)
        valid_densities = valid_densities[sort_idx]
        valid_pressures = valid_pressures[sort_idx]
        valid_energies = valid_energies[sort_idx]
        
        # Remove any duplicates or non-monotonic behavior
        # Keep only points where pressure increases with energy density
        monotonic_mask = np.ones(len(valid_energies), dtype=bool)
        for i in range(1, len(valid_energies)):
            if valid_pressures[i] <= valid_pressures[i-1] or valid_energies[i] <= valid_energies[i-1]:
                monotonic_mask[i] = False
        
        valid_densities = valid_densities[monotonic_mask]
        valid_pressures = valid_pressures[monotonic_mask]
        valid_energies = valid_energies[monotonic_mask]
        
        # Store EOS data
        self.eos_data = {
            'baryon_density': valid_densities,
            'pressure': valid_pressures,
            'energy_density': valid_energies
        }
        
        # Create interpolation functions - use log space for better behavior
        if len(valid_pressures) > 3:  # Need at least 4 points for interpolation
            log_pressure = np.log(valid_pressures)
            log_energy = np.log(valid_energies)
            
            # P(ε) relationship
            self.pressure_interp = interp1d(log_energy, log_pressure, 
                                           kind='linear', bounds_error=False, 
                                           fill_value=(log_pressure[0], log_pressure[-1]))
            
            # ε(P) relationship  
            self.energy_density_interp = interp1d(log_pressure, log_energy,
                                                 kind='linear', bounds_error=False,
                                                 fill_value=(log_energy[0], log_energy[-1]))
        else:
            raise RuntimeError("Not enough points for interpolation")
        
        if self.verbose:
            print(f"  ✓ EOS data prepared: {len(valid_densities)} valid points")
            print(f"  Density range: {valid_densities.min():.3f} - {valid_densities.max():.3f} fm⁻³")
            print(f"  Pressure range: {valid_pressures.min():.1f} - {valid_pressures.max():.1f} MeV/fm³")
            print(f"  Energy range: {valid_energies.min():.1f} - {valid_energies.max():.1f} MeV/fm³")
            print(f"  Failed points: {failed_count}")

    def get_pressure_from_energy_density(self, energy_density: float) -> float:
        """Get pressure from energy density using interpolation."""
        if self.pressure_interp is None:
            raise RuntimeError("EOS data not prepared. Call prepare_eos_data() first.")
        
        # Clamp to valid range
        min_energy = np.min(self.eos_data['energy_density'])
        max_energy = np.max(self.eos_data['energy_density'])
        energy_density = np.clip(energy_density, min_energy, max_energy)
        
        log_energy = np.log(max(energy_density, 1e-10))
        log_pressure = self.pressure_interp(log_energy)
        return np.exp(log_pressure)
    
    def get_energy_density_from_pressure(self, pressure: float) -> float:
        """Get energy density from pressure using interpolation."""
        if self.energy_density_interp is None:
            raise RuntimeError("EOS data not prepared. Call prepare_eos_data() first.")
        
        # Clamp to valid range
        min_pressure = np.min(self.eos_data['pressure'])
        max_pressure = np.max(self.eos_data['pressure'])
        pressure = np.clip(pressure, min_pressure, max_pressure)
        
        log_pressure = np.log(max(pressure, 1e-10))
        log_energy = self.energy_density_interp(log_pressure)
        return np.exp(log_energy)
    
    def tov_equations(self, r: float, y: np.ndarray) -> np.ndarray:
        """
        TOV differential equations in SI units.
        
        Args:
            r: Radial coordinate (m)
            y: [m(r), P(r)] where m is mass and P is pressure
            
        Returns:
            [dm/dr, dP/dr]
        """
        m, P = y
        
        if P <= 0 or r <= 0:
            return np.array([0.0, 0.0])
        
        try:
            # Get energy density from pressure (convert units)
            P_MeV_fm3 = P * self.Pa_to_MeV_fm3
            
            # Check if pressure is within EOS range
            if self.eos_data is not None:
                P_min = np.min(self.eos_data['pressure'])
                P_max = np.max(self.eos_data['pressure'])
                
                if P_MeV_fm3 < P_min or P_MeV_fm3 > P_max:
                    # Outside EOS range - integration should stop
                    return np.array([0.0, -1e20])
            
            epsilon_MeV_fm3 = self.get_energy_density_from_pressure(P_MeV_fm3)
            epsilon = epsilon_MeV_fm3 * self.MeV_fm3_to_Pa  # Convert back to Pa
            
            # TOV equations in SI units
            # dm/dr = 4πr²ε/c²
            dmdr = 4.0 * PI * r**2 * epsilon / self.c**2
            
            # dP/dr = -G(ε + P/c²)(m + 4πr³P/c²)/(c²r(r - 2Gm/c²))
            numerator = (epsilon + P/self.c**2) * (m + 4.0*PI*r**3*P/self.c**2)
            denominator = r * (r - 2.0*self.G*m/self.c**2)
            
            # Check for approaching Schwarzschild radius
            if denominator <= r * 1e-6:  # Very close to 2GM/c²
                return np.array([dmdr, -1e20])  # Force integration to stop
            
            dPdr = -self.G * numerator / (self.c**2 * denominator)
            
            # Sanity checks
            if not np.isfinite(dmdr) or not np.isfinite(dPdr):
                return np.array([0.0, -1e20])
            
            if abs(dPdr) > 1e15:  # Pressure gradient too steep
                return np.array([dmdr, -1e15 * np.sign(dPdr)])
            
            return np.array([dmdr, dPdr])
            
        except Exception as e:
            # Any error in EOS lookup or calculation
            return np.array([0.0, -1e20])  # Force integration to stop
    
    def solve_star(self, central_pressure: float, r_max: float = 30000.0, 
                   rtol: float = 1e-8, atol: float = 1e-10, max_step: float = 50.0) -> Dict[str, Any]:
        """
        Solve TOV equations for a given central pressure.
        
        Args:
            central_pressure: Central pressure in MeV/fm³
            r_max: Maximum integration radius in meters
            rtol: Relative tolerance for integration
            atol: Absolute tolerance for integration  
            max_step: Maximum step size in meters
            
        Returns:
            Dictionary with stellar structure data
        """
        if self.eos_data is None:
            self.prepare_eos_data()
        
        # Check if central pressure is within EOS range
        P_min = np.min(self.eos_data['pressure'])
        P_max = np.max(self.eos_data['pressure'])
        
        if central_pressure < P_min or central_pressure > P_max:
            if self.verbose:
                print(f"  Central pressure {central_pressure:.1f} MeV/fm³ outside EOS range "
                      f"[{P_min:.1f}, {P_max:.1f}] MeV/fm³")
            return {'converged': False, 'error': 'pressure_out_of_range'}
        
        # Convert central pressure to SI units
        P_central_Pa = central_pressure * self.MeV_fm3_to_Pa
        
        # Initial conditions at small radius (avoid r=0 singularity)
        r0 = 1e-4  # m (0.1 mm)
        
        # Get central energy density
        central_energy_density = self.get_energy_density_from_pressure(central_pressure)  # MeV/fm³
        epsilon_central_Pa = central_energy_density * self.MeV_fm3_to_Pa  # Convert to Pa
        
        # Initial mass: m(r0) ≈ (4π/3) * r0³ * ε_central / c²
        m0 = 4.0/3.0 * PI * r0**3 * epsilon_central_Pa / self.c**2
        
        initial_conditions = [m0, P_central_Pa]
        
        if self.verbose:
            print(f"  Starting integration:")
            print(f"    r0 = {r0*1000:.3f} mm")
            print(f"    P_central = {P_central_Pa:.2e} Pa ({central_pressure:.1f} MeV/fm³)")
            print(f"    ε_central = {epsilon_central_Pa:.2e} Pa ({central_energy_density:.1f} MeV/fm³)")
            print(f"    m0 = {m0:.2e} kg")
        
        # Integration stopping condition (pressure drops to near zero)
        def pressure_zero(r, y):
            return y[1] - P_central_Pa * 1e-6  # Stop when P < 10⁻⁶ P_central
        pressure_zero.terminal = True
        pressure_zero.direction = -1
        
        # Additional stopping condition for numerical issues
        def numerical_limit(r, y):
            m, P = y
            if P <= 0 or m <= 0:
                return 0
            # Check if approaching Schwarzschild radius
            if r > 0 and (2.0 * self.G * m / self.c**2) >= 0.9 * r:
                return 0
            return 1
        numerical_limit.terminal = True
        
        try:
            # Solve TOV equations with both stopping conditions
            solution = solve_ivp(
                self.tov_equations,
                [r0, r_max],
                initial_conditions,
                events=[pressure_zero, numerical_limit],
                rtol=rtol,
                atol=atol,
                max_step=max_step,
                dense_output=True,
                method='DOP853'  # High-order method for better accuracy
            )
            
            if not solution.success:
                if self.verbose:
                    print(f"    Integration failed: {solution.message}")
                return {'converged': False, 'error': 'integration_failed', 'message': solution.message}
            
            # Check which stopping condition was triggered
            surface_found = False
            R_star = None
            M_star = None
            
            # Check pressure_zero event (natural surface)
            if len(solution.t_events[0]) > 0:
                R_star = solution.t_events[0][0]  # m
                M_star = solution.y_events[0][0][0]  # kg
                surface_found = True
                stop_reason = "pressure_zero"
            # Check numerical_limit event
            elif len(solution.t_events[1]) > 0:
                R_star = solution.t_events[1][0]  # m
                M_star = solution.y_events[1][0][0]  # kg
                surface_found = True
                stop_reason = "numerical_limit"
            # Integration reached r_max without finding surface
            else:
                R_star = solution.t[-1]
                M_star = solution.y[0, -1]
                stop_reason = "max_radius"
                
                # Check if pressure is still significant
                final_pressure = solution.y[1, -1] * self.Pa_to_MeV_fm3
                if final_pressure > central_pressure * 1e-3:  # More than 0.1% of central
                    if self.verbose:
                        print(f"    Warning: Integration stopped at r_max with significant pressure: {final_pressure:.2f} MeV/fm³")
                    return {'converged': False, 'error': 'no_surface_found'}
            
            # Validate results
            if R_star <= r0 or M_star <= 0:
                if self.verbose:
                    print(f"    Invalid solution: R = {R_star:.1e} m, M = {M_star:.1e} kg")
                return {'converged': False, 'error': 'invalid_solution'}
            
            # Convert to convenient units
            R_km = R_star * M_TO_KM  # Convert m to km using constants.py
            M_solar = M_star / self.M_sun  # Solar masses
            
            # Calculate compactness
            compactness = self.G * M_star / (self.c**2 * R_star)
            
            # Physical validity checks
            if compactness >= 0.5:  # Too close to black hole limit
                if self.verbose:
                    print(f"    Unphysical compactness: {compactness:.3f} >= 0.5")
                return {'converged': False, 'error': 'unphysical_compactness'}
            
            if R_km < 5.0 or R_km > 25.0:  # Unrealistic radius
                if self.verbose:
                    print(f"    Unrealistic radius: {R_km:.1f} km")
                return {'converged': False, 'error': 'unrealistic_radius'}
            
            if M_solar < 0.1 or M_solar > 5.0:  # Unrealistic mass
                if self.verbose:
                    print(f"    Unrealistic mass: {M_solar:.2f} M☉")
                return {'converged': False, 'error': 'unrealistic_mass'}
            
            if self.verbose:
                print(f"    ✓ Success ({stop_reason}): M = {M_solar:.3f} M☉, R = {R_km:.2f} km, ξ = {compactness:.3f}")
            
            return {
                'converged': True,
                'central_pressure': central_pressure,  # MeV/fm³
                'central_energy_density': central_energy_density,  # MeV/fm³
                'radius': R_km,  # km
                'mass': M_solar,  # M_sun
                'radius_m': R_star,  # m
                'mass_kg': M_star,  # kg
                'compactness': compactness,  # GM/Rc²
                'solution': solution,
                'r0': r0,
                'stop_reason': stop_reason
            }
            
        except Exception as e:
            if self.verbose:
                print(f"    TOV integration error: {e}")
            return {'converged': False, 'error': 'exception', 'message': str(e)}
    
    def compute_mass_radius_sequence(self, pressure_range: Tuple[float, float] = None,
                                   n_stars: int = 30, max_attempts: int = 3) -> Dict[str, np.ndarray]:
        """
        Compute mass-radius sequence for a range of central pressures.
        
        Args:
            pressure_range: (P_min, P_max) in MeV/fm³. If None, uses EOS data range.
            n_stars: Number of stellar models to compute
            max_attempts: Maximum attempts per pressure value
            
        Returns:
            Dictionary with arrays of masses, radii, and other properties
        """
        if self.verbose:
            print(f"\nComputing M-R sequence for {self.eos.name}")
        
        # Prepare EOS data if not already done
        if self.eos_data is None:
            if self.verbose:
                print("Preparing EOS data for TOV integration...")
            try:
                self.prepare_eos_data(rho_min=0.08, rho_max=8.0, n_points=150)
            except Exception as e:
                if self.verbose:
                    print(f"✗ EOS data preparation failed: {e}")
                return None
        
        # Check if EOS data is valid
        if (self.eos_data is None or 
            len(self.eos_data.get('pressure', [])) == 0):
            if self.verbose:
                print(f"✗ No valid EOS data available")
            return None
        
        # Use EOS data pressure range if no range specified
        if pressure_range is None:
            eos_P_min = np.min(self.eos_data['pressure'])
            eos_P_max = np.max(self.eos_data['pressure'])
            
            # Use a safe margin from the actual EOS range
            # Start well above minimum to avoid numerical issues
            P_min = eos_P_min * 3.0  # Start at 3x minimum pressure
            P_max = eos_P_max * 0.9  # Stay within 90% of maximum
            
            if self.verbose:
                print(f"Using EOS pressure range:")
                print(f"  EOS range: {eos_P_min:.1f} - {eos_P_max:.1f} MeV/fm³")
                print(f"  Safe range: {P_min:.1f} - {P_max:.1f} MeV/fm³")
        else:
            P_min, P_max = pressure_range
            eos_P_min = np.min(self.eos_data['pressure'])
            eos_P_max = np.max(self.eos_data['pressure'])
            
            # Check if requested range is within EOS capabilities
            if P_min < eos_P_min or P_max > eos_P_max:
                if self.verbose:
                    print(f"Warning: Requested pressure range ({P_min:.1f}-{P_max:.1f}) "
                          f"exceeds EOS range ({eos_P_min:.1f}-{eos_P_max:.1f})")
                    print(f"Adjusting to EOS limits...")
                
                P_min = max(P_min, eos_P_min * 2.0)  # At least 2x minimum
                P_max = min(P_max, eos_P_max * 0.9)  # Stay within 90% of max
        
        if P_min >= P_max:
            if self.verbose:
                print(f"Error: Invalid pressure range: {P_min:.1f} >= {P_max:.1f}")
            return None
        
        if self.verbose:
            print(f"Final pressure range: {P_min:.2f} - {P_max:.2f} MeV/fm³")
            print(f"Number of models: {n_stars}")
        
        # Create pressure grid
        central_pressures = np.logspace(np.log10(P_min), np.log10(P_max), n_stars)
        
        # Storage for results
        valid_pressures = []
        central_energy_densities = []
        masses = []
        radii = []
        compactnesses = []
        successful_models = 0
        
        for i, P_central in enumerate(central_pressures):
            if self.verbose and i % 5 == 0:
                print(f"  Model {i+1}/{len(central_pressures)}: P_c = {P_central:.1f} MeV/fm³")
            
            success = False
            for attempt in range(max_attempts):
                try:
                    # Solve TOV equations
                    result = self.solve_star(P_central, 
                                           rtol=1e-8 if attempt == 0 else 1e-6,
                                           max_step=50.0 if attempt == 0 else 100.0)
                    
                    if result and result.get('converged', False):
                        # Basic validity checks
                        mass = result['mass']
                        radius = result['radius']
                        
                        if (0.1 <= mass <= 4.0 and  # Reasonable mass range
                            5.0 <= radius <= 25.0 and  # Reasonable radius range
                            result['compactness'] < 0.5):  # Not a black hole
                            
                            valid_pressures.append(P_central)
                            central_energy_densities.append(result['central_energy_density'])
                            masses.append(mass)
                            radii.append(radius)
                            compactnesses.append(result['compactness'])
                            successful_models += 1
                            success = True
                            break
                            
                except Exception as e:
                    if self.verbose and attempt == max_attempts - 1 and i < 3:
                        print(f"    Failed at P = {P_central:.1f} MeV/fm³: {e}")
                    continue
            
            if not success and self.verbose and i < 3:
                print(f"    All attempts failed for P = {P_central:.1f} MeV/fm³")
        
        if self.verbose:
            if successful_models > 0:
                print(f"\n✓ Successfully computed {successful_models}/{len(central_pressures)} models")
                if len(masses) > 0:
                    print(f"  Mass range: {np.min(masses):.3f} - {np.max(masses):.3f} M☉")
                    print(f"  Radius range: {np.min(radii):.2f} - {np.max(radii):.2f} km")
            else:
                print(f"\n✗ No successful stellar models computed!")
        
        # Convert to arrays and return
        return {
            'central_pressures': np.array(valid_pressures),
            'central_energy_densities': np.array(central_energy_densities),
            'masses': np.array(masses),
            'radii': np.array(radii),
            'compactnesses': np.array(compactnesses),
            'converged_fraction': successful_models / len(central_pressures) if len(central_pressures) > 0 else 0.0,
            'eos_name': self.eos.name
        }
    
    def get_maximum_mass_star(self, mr_sequence: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Find the maximum mass configuration from M-R sequence."""
        if len(mr_sequence['masses']) == 0:
            return None
        
        max_idx = np.argmax(mr_sequence['masses'])
        
        return {
            'mass': mr_sequence['masses'][max_idx],
            'radius': mr_sequence['radii'][max_idx],
            'central_pressure': mr_sequence['central_pressures'][max_idx],
            'central_energy_density': mr_sequence['central_energy_densities'][max_idx],
            'compactness': mr_sequence['compactnesses'][max_idx]
        }
    
    def plot_mass_radius_diagram(self, mr_sequences: List[Dict], title: str = "Mass-Radius Diagram",
                               save_filename: str = None):
        """
        Plot mass-radius diagram for multiple EOS models.
        
        Args:
            mr_sequences: List of M-R sequence dictionaries
            title: Plot title
            save_filename: Optional filename to save plot
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(mr_sequences)))
        
        for i, mr_data in enumerate(mr_sequences):
            if len(mr_data['masses']) == 0:
                continue
                
            label = mr_data.get('eos_name', f'EOS {i+1}')
            color = colors[i]
            
            # Plot M-R curve
            ax.plot(mr_data['radii'], mr_data['masses'], 
                   color=color, linewidth=2.5, label=label, alpha=0.8)
            
            # Mark maximum mass
            max_mass_data = self.get_maximum_mass_star(mr_data)
            if max_mass_data:
                ax.plot(max_mass_data['radius'], max_mass_data['mass'],
                       'o', color=color, markersize=8, markeredgecolor='black', markeredgewidth=1)
                
                # Annotate maximum mass
                ax.annotate(f"M_max = {max_mass_data['mass']:.2f} M☉",
                          xy=(max_mass_data['radius'], max_mass_data['mass']),
                          xytext=(5, 5), textcoords='offset points',
                          fontsize=9, color=color, fontweight='bold')
        
        # Observational constraints
        self._add_observational_constraints(ax)
        
        # Formatting
        ax.set_xlabel('Radius (km)', fontsize=14)
        ax.set_ylabel('Mass (M☉)', fontsize=14)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12)
        
        # Set reasonable limits
        ax.set_xlim(8, 18)
        ax.set_ylim(0.5, 3.0)
        
        plt.tight_layout()
        
        if save_filename:
            plt.savefig(save_filename, dpi=300, bbox_inches='tight')
            if self.verbose:
                print(f"M-R diagram saved to {save_filename}")
        
        plt.show()
    
    def _add_observational_constraints(self, ax):
        """Add observational constraints to M-R plot."""
        # PSR J0740+6620 (2.08 ± 0.07 M☉, R = 12.39 ± 1.3 km)
        ax.errorbar(12.39, 2.08, xerr=1.3, yerr=0.07, 
                   fmt='s', color='red', markersize=8, capsize=5,
                   label='PSR J0740+6620', markeredgecolor='black')
        
        # PSR J0348+0432 (2.01 ± 0.04 M☉)
        ax.axhline(y=2.01, xmin=0.1, xmax=0.9, color='orange', linestyle='--', alpha=0.7)
        ax.fill_between([8, 18], [1.97, 1.97], [2.05, 2.05], 
                       color='orange', alpha=0.2, label='PSR J0348+0432')
        
        # Causality limit (c_s ≤ c)
        ax.axhline(y=2.16, color='black', linestyle=':', alpha=0.5, label='Causality limit')
        
        # GW170817 constraint (approximate)
        ax.fill_between([10, 13.5], [1.16, 1.16], [1.36, 1.36],
                       color='lightblue', alpha=0.4, label='GW170817 (approx.)')

def create_mass_radius_comparison(eos_list: List, pressure_range: Tuple[float, float] = None,
                                n_stars: int = 30, title: str = "Mass-Radius Comparison") -> Dict[str, Any]:
    """
    Create mass-radius comparison for multiple EOS models.
    
    Args:
        eos_list: List of EOS objects
        pressure_range: Range of central pressures in MeV/fm³. If None, uses each EOS's optimal range.
        n_stars: Number of stellar models per EOS
        title: Plot title
        
    Returns:
        Dictionary with all M-R sequences and summary statistics
    """
    print(f"\nComputing Mass-Radius sequences for {len(eos_list)} EOS models")
    print("=" * 60)
    
    mr_sequences = []
    summary_stats = []
    
    for i, eos in enumerate(eos_list):
        print(f"\nProcessing EOS {i+1}/{len(eos_list)}: {eos.name}")
        
        # Create TOV solver
        tov_solver = TOVSolver(eos, verbose=True)
        
        try:
            # Compute M-R sequence - let each EOS use its optimal pressure range
            mr_data = tov_solver.compute_mass_radius_sequence(
                pressure_range=pressure_range,  # None means auto-detect
                n_stars=n_stars
            )
            
            if mr_data and len(mr_data['masses']) > 0:
                mr_sequences.append(mr_data)
                
                # Get maximum mass star
                max_mass_star = tov_solver.get_maximum_mass_star(mr_data)
                
                stats = {
                    'eos_name': eos.name,
                    'max_mass': max_mass_star['mass'] if max_mass_star else 0.0,
                    'radius_at_max_mass': max_mass_star['radius'] if max_mass_star else 0.0,
                    'converged_models': len(mr_data['masses']),  # Use consistent key name
                    'num_models': len(mr_data['masses']),  # Also provide alternative key
                    'success_rate': mr_data['converged_fraction'],
                    'pressure_range_used': (np.min(mr_data['central_pressures']), 
                                          np.max(mr_data['central_pressures'])) if len(mr_data['central_pressures']) > 0 else (0, 0)
                }
                summary_stats.append(stats)
                
                print(f"  ✓ Success: M_max = {stats['max_mass']:.3f} M☉ at R = {stats['radius_at_max_mass']:.1f} km")
                print(f"    Models: {stats['converged_models']}, Success rate: {stats['success_rate']*100:.1f}%")
                print(f"    Pressure range: {stats['pressure_range_used'][0]:.1f} - {stats['pressure_range_used'][1]:.1f} MeV/fm³")
            else:
                print(f"  ✗ Failed: No valid stellar models")
                
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    # Plot comparison
    if mr_sequences:
        tov_solver = TOVSolver(eos_list[0])  # Use first EOS for plotting
        tov_solver.plot_mass_radius_diagram(mr_sequences, title=title)
        
        # Print summary
        print(f"\nFinal Summary:")
        print("-" * 70)
        for stats in summary_stats:
            print(f"{stats['eos_name']:15s}: M_max = {stats['max_mass']:5.2f} M☉, "
                  f"R = {stats['radius_at_max_mass']:5.1f} km, "
                  f"Models = {stats['converged_models']:2d}, "
                  f"Success = {stats['success_rate']*100:4.1f}%")
    
    return {
        'mr_sequences': mr_sequences,
        'summary_stats': summary_stats,
        'pressure_range': pressure_range,
        'n_stars': n_stars
    }
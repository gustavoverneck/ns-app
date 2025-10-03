import numpy as np
import matplotlib.pyplot as plt
from ns.eos import EOSSolver
from ns.models import EOSType
from ns.parametrizations import ParametrizationType
from ns.tov_solver import TOVSolver
import argparse
from scipy.optimize import broyden1

def test_atomic_eos():
    """Test the atomic neutron star EOS."""
    
    print("=" * 60)
    print("Testing Atomic Neutron Star EOS")
    print("=" * 60)
    
    # Create atomic EOS with GM1 parametrization
    atomic_eos = EOSSolver.create_eos(
        eos_type=EOSType.ATOMIC_STAR,
        parametrization=ParametrizationType.GM1,
        root_finder=broyden1,
        verbose=True
    )
    
    print(f"\nEOS Type: {atomic_eos.eos_type}")
    print(f"Parametrization: {atomic_eos.parametrization}")
    print(f"Number of baryons: {atomic_eos.n_baryons}")
    print(f"Number of leptons: {atomic_eos.n_leptons}")
    
    # Test density range (nuclear saturation density e acima)
    rho_sat = 0.153  # Nuclear saturation density (fm^-3)
    # Inclui densidades mais altas para obter P>0 de forma robusta (ajuda o TOV)
    densities = np.linspace(0.1 * rho_sat, 12.0 * rho_sat, 140)
    
    print(f"\nTesting {len(densities)} density points from {densities[0]:.3f} to {densities[-1]:.3f} fm^-3")
    
    # Test field equation solver and EOS calculation
    print(f"\n{'Density':>8} {'Energy':>12} {'Pressure':>12} {'Conv':>6}")
    print(f"{'(fm⁻³)':>8} {'(MeV/fm³)':>12} {'(MeV/fm³)':>12} {'':>6}")
    print("-" * 50)
    
    eos_data = []
    failed_count = 0
    
    for rho in densities:
        solution = atomic_eos.solve_field_equations(rho)
        
        if solution['converged']:
            try:
                # Calculate EOS
                eos_result = atomic_eos.compute_eos(rho)
                
                energy = eos_result['energy_density']
                pressure = eos_result['pressure']
                
                if len(eos_data) < 5:  # Only print first few for brevity
                    print(f"{rho:8.3f} {energy:12.2f} {pressure:12.2f} {'Yes':>6}")
                eos_data.append((rho, energy, pressure))
                
            except Exception as e:
                if failed_count < 5:
                    print(f"{rho:8.3f} {'ERROR':>12} {'ERROR':>12} {'No':>6}")
                failed_count += 1
        else:
            if failed_count < 5:
                print(f"{rho:8.3f} {'NO CONV':>12} {'NO CONV':>12} {'No':>6}")
            failed_count += 1
    
    if len(eos_data) > 5:
        print(f"... (showing first 5 of {len(eos_data)} successful calculations)")
    
    print(f"\nSuccessful calculations: {len(eos_data)}/{len(densities)}")
    print(f"Failed calculations: {failed_count}/{len(densities)}")
    
    return atomic_eos, eos_data

def _estimate_pressure_range_for_tov(eos, verbose=True):
    """
    Varre o EOS para estimar um intervalo de pressão positivo (MeV/fm³)
    apropriado para usar como pressão central no TOV.
    """
    # Varredura em densidade (fm^-3), cobrindo de ~0.3 a ~3.5 fm^-3
    dens_scan = np.geomspace(0.3, 3.5, 120)
    ps, es = [], []
    for i, rho in enumerate(dens_scan):
        try:
            res = eos.compute_eos(rho)
            p, e = res.get('pressure', 0.0), res.get('energy_density', 0.0)
            if np.isfinite(p) and np.isfinite(e) and p > 0 and e > 0:
                ps.append(p); es.append(e)
        except Exception:
            continue
    if len(ps) == 0:
        # Fallback razoável
        if verbose:
            print("  Could not find positive-pressure band; using default [100, 600] MeV/fm³")
        return 100.0, 600.0
    p_min, p_max = float(np.min(ps)), float(np.max(ps))
    if verbose:
        print(f"  Estimated EOS pressure band: {p_min:.1f} - {p_max:.1f} MeV/fm³ (from scan)")
    # Afrouxa levemente as bordas para segurança
    return max(1.05 * p_min, 1.0), 0.95 * p_max

def test_tov_integration(eos, eos_data, star_type=""):
    """Test TOV integration and generate M-R diagram."""
    
    print(f"\n{'='*60}")
    print(f"TOV Integration and Mass-Radius Calculation ({star_type})")
    print(f"{'='*60}")
    
    if len(eos_data) < 10:
        print("Not enough EOS data points for TOV integration")
        return None, None
    
    # Create TOV solver
    tov_solver = TOVSolver(eos=eos, verbose=True)
    
    # Set up pressure range for central pressures
    if len(eos_data[0]) == 3:  # atomic: (rho, energy, pressure)
        densities, energies, pressures = zip(*eos_data)
    else:  # strange: (rho, energy, pressure, sigma, omega)
        densities, energies, pressures = zip(*[(d[0], d[1], d[2]) for d in eos_data])
    
    pressures = np.array(pressures)
    energies = np.array(energies)
    
    # Filter out negative or zero pressures
    valid_mask = (pressures > 0) & (energies > 0)
    if not np.any(valid_mask):
        print("No valid positive pressures found")
        return None, None
    
    pressures_valid = pressures[valid_mask]
    energies_valid = energies[valid_mask]
    
    # Set pressure range for central pressures (from min to max valid pressure)
    # p_min = pressures_valid.min()
    # p_max = pressures_valid.max()
    
    # Intervalo baseado no scan do EOS (garante P dentro da faixa válida do solver)
    print("Estimating valid central-pressure range from EOS scan...")
    p_min_scan, p_max_scan = _estimate_pressure_range_for_tov(eos, verbose=True)
    print(f"Using central-pressure range: {p_min_scan:.2f} - {p_max_scan:.2f} MeV/fm³")
    # Gera pressões centrais (log)
    n_stars = 50
    central_pressures = np.logspace(np.log10(p_min_scan), np.log10(p_max_scan), n_stars)
    
    print(f"Computing {n_stars} {star_type} neutron star models...")
    
    # Solve TOV equations for each central pressure
    masses = []
    radii = []
    valid_central_pressures = []
    
    for i, pc in enumerate(central_pressures):
        try:
            result = tov_solver.solve_star(central_pressure=pc)
            
            if result['converged'] and result['mass'] > 0 and result['radius'] > 0:
                masses.append(result['mass'])
                radii.append(result['radius'])
                valid_central_pressures.append(pc)
                
                if i < 5 or i % 10 == 0:  # Print progress
                    print(f"Star {i+1:2d}: Pc = {pc:.2e} MeV/fm³ → "
                          f"M = {result['mass']:.3f} M☉, R = {result['radius']:.2f} km")
            else:
                if i < 5:
                    print(f"Star {i+1:2d}: Failed to converge or invalid result")
                    
        except Exception as e:
            if i < 5:
                print(f"Star {i+1:2d}: Error - {e}")
            continue
    
    masses = np.array(masses)
    radii = np.array(radii)
    valid_central_pressures = np.array(valid_central_pressures)
    
    print(f"\nSuccessfully computed {len(masses)} {star_type} neutron star models")
    
    if len(masses) > 0:
        max_mass_idx = np.argmax(masses)
        print(f"Maximum mass: {masses[max_mass_idx]:.3f} M☉ at R = {radii[max_mass_idx]:.2f} km")
        print(f"Mass range: {masses.min():.3f} - {masses.max():.3f} M☉")
        print(f"Radius range: {radii.min():.2f} - {radii.max():.2f} km")
    
    return (masses, radii, valid_central_pressures), eos_data

def test_strange_eos():
    """Test the strange star EOS with full baryon octet."""
    
    print("\n" + "=" * 60)
    print("Testing Strange Star EOS (Full Baryon Octet)")
    print("=" * 60)
    
    try:
        # Create strange EOS with GM1 parametrization
        print("Creating strange star EOS...")
        strange_eos = EOSSolver.create_eos(
            eos_type=EOSType.STRANGE_STAR,
            parametrization=ParametrizationType.GM1,
            verbose=True
        )
        print("Strange star EOS created successfully!")
        
        print(f"\nEOS Type: {strange_eos.eos_type}")
        print(f"Parametrization: {strange_eos.parametrization}")
        print(f"Number of baryons: {getattr(strange_eos, 'n_baryons', 'UNKNOWN')}")
        print(f"Number of leptons: {getattr(strange_eos, 'n_leptons', 'UNKNOWN')}")
        
        # Check if EOS has required methods
        print(f"\nEOS methods check:")
        print(f"  - has solve_field_equations: {hasattr(strange_eos, 'solve_field_equations')}")
        print(f"  - has compute_eos: {hasattr(strange_eos, 'compute_eos')}")
        print(f"  - has get_parameter: {hasattr(strange_eos, 'get_parameter')}")
        
        # Print particle content if available
        if hasattr(strange_eos, 'print_particle_content'):
            strange_eos.print_particle_content()
        elif hasattr(strange_eos, 'baryon_masses'):
            print(f"\nBaryon masses: {strange_eos.baryon_masses}")
            print(f"Baryon charges: {getattr(strange_eos, 'baryon_charges', 'UNKNOWN')}")
        
    except Exception as e:
        print(f"ERROR creating strange star EOS: {e}")
        import traceback
        traceback.print_exc()
        return None, []
    
    # Test density range (higher densities for strange matter)
    rho_sat = 0.153  # Nuclear saturation density (fm^-3)
    # Start with fewer points for debugging
    densities = np.linspace(1.0 * rho_sat, 3.0 * rho_sat, 10)
    
    print(f"\nTesting {len(densities)} density points from {densities[0]:.3f} to {densities[-1]:.3f} fm^-3")
    print("(Higher densities for hyperon onset)")
    
    # Test field equation solver and EOS calculation
    print(f"\n{'Density':>8} {'Energy':>12} {'Pressure':>12} {'σ-field':>10} {'ω-field':>10} {'Conv':>6}")
    print(f"{'(fm⁻³)':>8} {'(MeV/fm³)':>12} {'(MeV/fm³)':>12} {'(MeV)':>10} {'(MeV)':>10} {'':>6}")
    print("-" * 70)
    
    eos_data = []
    failed_count = 0
    
    for i, rho in enumerate(densities):
        print(f"\nTesting density {i+1}/{len(densities)}: {rho:.3f} fm^-3")
        
        try:
            # First test field equation solving
            print(f"  Solving field equations...")
            solution = strange_eos.solve_field_equations(rho)
            print(f"  Field equations result: converged = {solution.get('converged', False)}")
            
            if solution.get('converged', False):
                print(f"  Field values: sigma={solution.get('sigma_field', 'N/A'):.3f}, omega={solution.get('omega_field', 'N/A'):.3f}")
                
                try:
                    # Then test EOS calculation
                    print(f"  Computing EOS...")
                    eos_result = strange_eos.compute_eos(rho)
                    print(f"  EOS result: converged = {eos_result.get('converged', False)}")
                    
                    if eos_result.get('converged', False):
                        energy = eos_result['energy_density']
                        pressure = eos_result['pressure']
                        sigma = eos_result.get('sigma_field', 0.0)
                        omega = eos_result.get('omega_field', 0.0)
                        
                        print(f"{rho:8.3f} {energy:12.2f} {pressure:12.2f} {sigma:10.2f} {omega:10.2f} {'Yes':>6}")
                        eos_data.append((rho, energy, pressure, sigma, omega))
                    else:
                        print(f"{rho:8.3f} {'EOS FAIL':>12} {'EOS FAIL':>12} {'EOS FAIL':>10} {'EOS FAIL':>10} {'No':>6}")
                        failed_count += 1
                        
                except Exception as e:
                    print(f"  ERROR in compute_eos: {e}")
                    print(f"{rho:8.3f} {'ERROR':>12} {'ERROR':>12} {'ERROR':>10} {'ERROR':>10} {'No':>6}")
                    failed_count += 1
            else:
                print(f"{rho:8.3f} {'NO CONV':>12} {'NO CONV':>12} {'NO CONV':>10} {'NO CONV':>10} {'No':>6}")
                failed_count += 1
                
        except Exception as e:
            print(f"  ERROR in solve_field_equations: {e}")
            print(f"{rho:8.3f} {'ERROR':>12} {'ERROR':>12} {'ERROR':>10} {'ERROR':>10} {'No':>6}")
            failed_count += 1
            
        # Stop after 3 failures for debugging
        if failed_count >= 3:
            print(f"\nStopping after {failed_count} consecutive failures for debugging...")
            break
    
    print(f"\nSuccessful calculations: {len(eos_data)}/{len(densities)}")
    print(f"Failed calculations: {failed_count}/{len(densities)}")
    
    # Only analyze hyperon onset if we have some data
    if len(eos_data) > 2:
        try:
            analyze_hyperon_onset(strange_eos, eos_data)
        except Exception as e:
            print(f"Error in hyperon onset analysis: {e}")
    
    return strange_eos, eos_data

def analyze_hyperon_onset(strange_eos, eos_data):
    """Analyze when hyperons start appearing in the EOS."""
    
    print(f"\n{'='*40}")
    print("Hyperon Onset Analysis")
    print(f"{'='*40}")
    
    # Check a few sample densities to see which particles are present
    sample_densities = [eos_data[i][0] for i in [len(eos_data)//4, len(eos_data)//2, 3*len(eos_data)//4]]
    
    for rho in sample_densities:
        print(f"\nAt density ρ = {rho:.3f} fm⁻³ ({rho/0.153:.1f}ρ₀):")
        
        # Solve field equations
        solution = strange_eos.solve_field_equations(rho)
        
        if solution['converged']:
            # Check which particles have non-zero Fermi momenta
            # This would require access to the internal calculation
            # For now, just show the field values
            sigma = solution.get('sigma_field', 0.0)
            omega = solution.get('omega_field', 0.0)
            rho_field = solution.get('rho_field', 0.0)
            
            print(f"  σ-field: {sigma:.3f} MeV")
            print(f"  ω-field: {omega:.3f} MeV")
            print(f"  ρ-field: {rho_field:.6f} MeV")
            
            # Estimate effective masses (simplified)
            x_s = strange_eos.get_parameter('hyperon_sigma_coupling', 0.7)
            
            if hasattr(strange_eos, 'baryon_masses') and len(strange_eos.baryon_masses) >= 8:
                print(f"  Effective masses:")
                print(f"    Neutron: {strange_eos.baryon_masses[0] - sigma:.1f} MeV")
                print(f"    Proton:  {strange_eos.baryon_masses[1] - sigma:.1f} MeV")
                if len(strange_eos.baryon_masses) > 2:
                    print(f"    Lambda:  {strange_eos.baryon_masses[2] - x_s * sigma:.1f} MeV")

def plot_comparison_results(atomic_data, strange_data, atomic_mr, strange_mr):
    """Plot comparison between atomic and strange star EOS."""
    
    # Create figure with subplots
    fig = plt.figure(figsize=(24, 16))
    
    # EOS comparison plots (top row)
    ax1 = plt.subplot(3, 3, 1)
    ax2 = plt.subplot(3, 3, 2)
    ax3 = plt.subplot(3, 3, 3)
    
    # M-R comparison (middle row)
    ax4 = plt.subplot(3, 3, 4)
    ax5 = plt.subplot(3, 3, 5)
    ax6 = plt.subplot(3, 3, 6)
    
    # Field evolution for strange star (bottom row)
    ax7 = plt.subplot(3, 3, 7)
    ax8 = plt.subplot(3, 3, 8)
    ax9 = plt.subplot(3, 3, 9)
    
    # Extract atomic EOS data
    if atomic_data:
        atomic_densities, atomic_energies, atomic_pressures = zip(*atomic_data)
        atomic_densities = np.array(atomic_densities)
        atomic_energies = np.array(atomic_energies)
        atomic_pressures = np.array(atomic_pressures)
    
    # Extract strange EOS data
    if strange_data:
        if len(strange_data[0]) == 5:  # (rho, energy, pressure, sigma, omega)
            strange_densities, strange_energies, strange_pressures, sigma_fields, omega_fields = zip(*strange_data)
        else:
            strange_densities, strange_energies, strange_pressures = zip(*strange_data)
            sigma_fields = omega_fields = None
        
        strange_densities = np.array(strange_densities)
        strange_energies = np.array(strange_energies)
        strange_pressures = np.array(strange_pressures)
    
    # 1. Energy density comparison
    if atomic_data:
        ax1.plot(atomic_densities, atomic_energies, 'b-', linewidth=2, label='Atomic Star', marker='o', markersize=3)
    if strange_data:
        ax1.plot(strange_densities, strange_energies, 'r-', linewidth=2, label='Strange Star', marker='s', markersize=3)
    ax1.set_xlabel('Baryon Density (fm⁻³)')
    ax1.set_ylabel('Energy Density (MeV/fm³)')
    ax1.set_title('Energy Density Comparison')
    ax1.grid(True, alpha=0.3)
    ax1.axvline(0.153, color='gray', linestyle='--', alpha=0.7, label='ρ₀')
    ax1.legend()
    
    # 2. Pressure comparison
    if atomic_data:
        ax2.plot(atomic_densities, atomic_pressures, 'b-', linewidth=2, label='Atomic Star', marker='o', markersize=3)
    if strange_data:
        ax2.plot(strange_densities, strange_pressures, 'r-', linewidth=2, label='Strange Star', marker='s', markersize=3)
    ax2.set_xlabel('Baryon Density (fm⁻³)')
    ax2.set_ylabel('Pressure (MeV/fm³)')
    ax2.set_title('Pressure Comparison')
    ax2.grid(True, alpha=0.3)
    ax2.axvline(0.153, color='gray', linestyle='--', alpha=0.7, label='ρ₀')
    ax2.legend()
    
    # 3. P-ε diagram (log-log)
    if atomic_data:
        pos_mask_a = (atomic_energies > 0) & (atomic_pressures > 0)
        if np.any(pos_mask_a):
            ax3.loglog(atomic_energies[pos_mask_a], atomic_pressures[pos_mask_a], 
                      'b-', linewidth=2, label='Atomic Star', marker='o', markersize=3)
    
    if strange_data:
        pos_mask_s = (strange_energies > 0) & (strange_pressures > 0)
        if np.any(pos_mask_s):
            ax3.loglog(strange_energies[pos_mask_s], strange_pressures[pos_mask_s], 
                      'r-', linewidth=2, label='Strange Star', marker='s', markersize=3)
    
    ax3.set_xlabel('Energy Density (MeV/fm³)')
    ax3.set_ylabel('Pressure (MeV/fm³)')
    ax3.set_title('EOS: P vs ε (Log-Log)')
    ax3.grid(True, alpha=0.3)
    
    # Add ultra-relativistic reference line
    if atomic_data or strange_data:
        all_energies = []
        if atomic_data:
            all_energies.extend(atomic_energies[atomic_energies > 0])
        if strange_data:
            all_energies.extend(strange_energies[strange_energies > 0])
        
        if all_energies:
            e_min, e_max = min(all_energies), max(all_energies)
            e_line = np.linspace(e_min, e_max, 100)
            p_ultrarel = e_line / 3
            ax3.plot(e_line, p_ultrarel, 'k--', alpha=0.5, label='P = ε/3')
    
    ax3.legend()
    
    # 4. Mass-Radius comparison
    if atomic_mr:
        atomic_masses, atomic_radii, _ = atomic_mr
        ax4.plot(atomic_radii, atomic_masses, 'b-', linewidth=3, label='Atomic Star', marker='o', markersize=4)
    
    if strange_mr:
        strange_masses, strange_radii, _ = strange_mr
        ax4.plot(strange_radii, strange_masses, 'r-', linewidth=3, label='Strange Star', marker='s', markersize=4)
    
    ax4.set_xlabel('Radius (km)')
    ax4.set_ylabel('Mass (M☉)')
    ax4.set_title('Mass-Radius Comparison')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(1.4, color='gray', linestyle=':', alpha=0.7, label='Typical NS mass')
    ax4.axhline(2.0, color='red', linestyle=':', alpha=0.7, label='~2 M☉ constraint')
    ax4.legend()
    
    # 5. Maximum mass comparison
    if atomic_mr and strange_mr:
        atomic_masses, _, atomic_pc = atomic_mr
        strange_masses, _, strange_pc = strange_mr
        
        max_masses = []
        labels = []
        colors = []
        
        if len(atomic_masses) > 0:
            max_masses.append(atomic_masses.max())
            labels.append('Atomic')
            colors.append('blue')
        
        if len(strange_masses) > 0:
            max_masses.append(strange_masses.max())
            labels.append('Strange')
            colors.append('red')
        
        ax5.bar(labels, max_masses, color=colors, alpha=0.7)
        ax5.set_ylabel('Maximum Mass (M☉)')
        ax5.set_title('Maximum Mass Comparison')
        ax5.grid(True, alpha=0.3)
        
        # Add numerical values on bars
        for i, (label, mass) in enumerate(zip(labels, max_masses)):
            ax5.text(i, mass + 0.02, f'{mass:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 6. Radius at 1.4 solar masses comparison
    if atomic_mr and strange_mr:
        target_mass = 1.4
        radii_at_14 = []
        
        if len(atomic_masses) > 0:
            # Interpolate to find radius at 1.4 solar masses
            if atomic_masses.min() <= target_mass <= atomic_masses.max():
                r_14_atomic = np.interp(target_mass, atomic_masses, atomic_radii)
                radii_at_14.append(r_14_atomic)
            else:
                radii_at_14.append(0)
        
        if len(strange_masses) > 0:
            if strange_masses.min() <= target_mass <= strange_masses.max():
                r_14_strange = np.interp(target_mass, strange_masses, strange_radii)
                radii_at_14.append(r_14_strange)
            else:
                radii_at_14.append(0)
        
        if len(radii_at_14) == 2:
            ax6.bar(['Atomic', 'Strange'], radii_at_14, color=['blue', 'red'], alpha=0.7)
            ax6.set_ylabel('Radius at 1.4 M☉ (km)')
            ax6.set_title('Radius at 1.4 M☉')
            ax6.grid(True, alpha=0.3)
            
            for i, r in enumerate(radii_at_14):
                if r > 0:
                    ax6.text(i, r + 0.1, f'{r:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # 7-9. Strange star field evolution
    if strange_data and sigma_fields and omega_fields:
        sigma_fields = np.array(sigma_fields)
        omega_fields = np.array(omega_fields)
        
        # 7. Sigma field evolution
        ax7.plot(strange_densities, sigma_fields, 'g-', linewidth=2, marker='o', markersize=3)
        ax7.set_xlabel('Baryon Density (fm⁻³)')
        ax7.set_ylabel('σ-field (MeV)')
        ax7.set_title('Sigma Field Evolution')
        ax7.grid(True, alpha=0.3)
        ax7.axvline(0.153, color='gray', linestyle='--', alpha=0.7, label='ρ₀')
        
        # 8. Omega field evolution
        ax8.plot(strange_densities, omega_fields, 'm-', linewidth=2, marker='s', markersize=3)
        ax8.set_xlabel('Baryon Density (fm⁻³)')
        ax8.set_ylabel('ω-field (MeV)')
        ax8.set_title('Omega Field Evolution')
        ax8.grid(True, alpha=0.3)
        ax8.axvline(0.153, color='gray', linestyle='--', alpha=0.7, label='ρ₀')
        
        # 9. Field ratio
        field_ratio = omega_fields / (sigma_fields + 1e-10)  # Avoid division by zero
        ax9.plot(strange_densities, field_ratio, 'c-', linewidth=2, marker='^', markersize=3)
        ax9.set_xlabel('Baryon Density (fm⁻³)')
        ax9.set_ylabel('ω/σ Field Ratio')
        ax9.set_title('Field Ratio Evolution')
        ax9.grid(True, alpha=0.3)
        ax9.axvline(0.153, color='gray', linestyle='--', alpha=0.7, label='ρ₀')
    
    plt.tight_layout()
    plt.savefig('atomic_vs_strange_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\nComparison plots saved as 'atomic_vs_strange_comparison.png'")

def plot_complete_results(eos_data, mr_data):
    """Plot both EOS and M-R diagram."""
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    
    # EOS plots (top row)
    ax1 = plt.subplot(2, 3, 1)
    ax2 = plt.subplot(2, 3, 2)
    ax3 = plt.subplot(2, 3, 3)
    
    # M-R plots (bottom row)
    ax4 = plt.subplot(2, 3, 4)
    ax5 = plt.subplot(2, 3, 5)
    ax6 = plt.subplot(2, 3, 6)
    
    # Extract EOS data
    if eos_data:
        if len(eos_data[0]) == 3:  # atomic
            densities, energies, pressures = zip(*eos_data)
        else:  # strange
            densities, energies, pressures = zip(*[(d[0], d[1], d[2]) for d in eos_data])
        
        densities = np.array(densities)
        energies = np.array(energies)
        pressures = np.array(pressures)
        
        # Filter positive values for log plots
        pos_mask = (energies > 0) & (pressures > 0)
        energies_pos = energies[pos_mask]
        pressures_pos = pressures[pos_mask]
        
        # 1. Energy density vs density
        ax1.plot(densities, energies, 'bo-', linewidth=2, markersize=3)
        ax1.set_xlabel('Baryon Density (fm⁻³)')
        ax1.set_ylabel('Energy Density (MeV/fm³)')
        ax1.set_title('Energy Density vs Density')
        ax1.grid(True, alpha=0.3)
        ax1.axvline(0.153, color='r', linestyle='--', alpha=0.7, label='ρ₀')
        ax1.legend()
        
        # 2. Pressure vs density
        ax2.plot(densities, pressures, 'ro-', linewidth=2, markersize=3)
        ax2.set_xlabel('Baryon Density (fm⁻³)')
        ax2.set_ylabel('Pressure (MeV/fm³)')
        ax2.set_title('Pressure vs Density')
        ax2.grid(True, alpha=0.3)
        ax2.axvline(0.153, color='r', linestyle='--', alpha=0.7, label='ρ₀')
        ax2.legend()
        
        # 3. Energy vs Pressure (log-log)
        if len(energies_pos) > 0:
            ax3.loglog(energies_pos, pressures_pos, 'go-', linewidth=2, markersize=3, label='GM1')
            ax3.set_xlabel('Energy Density (MeV/fm³)')
            ax3.set_ylabel('Pressure (MeV/fm³)')
            ax3.set_title('EOS: P vs ε (Log-Log)')
            ax3.grid(True, alpha=0.3)
            
            # Add ultra-relativistic reference line
            e_line = np.linspace(energies_pos.min(), energies_pos.max(), 100)
            p_ultrarel = e_line / 3
            ax3.plot(e_line, p_ultrarel, 'k--', alpha=0.5, label='P = ε/3')
            ax3.legend()
    
    # Extract M-R data
    if mr_data:
        masses, radii, central_pressures = mr_data
        
        if len(masses) > 0:
            # 4. Mass-Radius diagram
            ax4.plot(radii, masses, 'bo-', linewidth=3, markersize=5, label='GM1')
            ax4.set_xlabel('Radius (km)')
            ax4.set_ylabel('Mass (M☉)')
            ax4.set_title('Mass-Radius Diagram')
            ax4.grid(True, alpha=0.3)
            ax4.legend()
            
            # Add observational constraints (approximate)
            ax4.axhline(1.4, color='gray', linestyle=':', alpha=0.7, label='Typical NS mass')
            ax4.axhline(2.0, color='red', linestyle=':', alpha=0.7, label='~2 M☉ constraint')
            ax4.legend()
            
            # 5. Mass vs Central Pressure
            ax5.semilogx(central_pressures, masses, 'ro-', linewidth=2, markersize=4)
            ax5.set_xlabel('Central Pressure (MeV/fm³)')
            ax5.set_ylabel('Mass (M☉)')
            ax5.set_title('Mass vs Central Pressure')
            ax5.grid(True, alpha=0.3)
            
            # 6. Radius vs Central Pressure
            ax6.semilogx(central_pressures, radii, 'go-', linewidth=2, markersize=4)
            ax6.set_xlabel('Central Pressure (MeV/fm³)')
            ax6.set_ylabel('Radius (km)')
            ax6.set_title('Radius vs Central Pressure')
            ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('complete_ns_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\nComplete analysis plots saved as 'complete_ns_analysis.png'")

def plot_ep_mr(atomic_data, atomic_mr, strange_data=None, strange_mr=None, filename='ep_mr.png'):
    """Plot only P vs ε and Mass-Radius."""
    fig = plt.figure(figsize=(12, 5))
    ax1 = plt.subplot(1, 2, 1)  # P vs ε
    ax2 = plt.subplot(1, 2, 2)  # M-R

    # Atomic EOS: (rho, energy, pressure)
    if atomic_data:
        a_rho, a_eps, a_p = zip(*atomic_data)
        a_eps = np.array(a_eps)
        a_p = np.array(a_p)
        mask = (a_eps > 0) & (a_p > 0)
        if np.any(mask):
            ax1.loglog(a_eps[mask], a_p[mask], 'b-', lw=2, label='Atomic')

    # Strange EOS: (rho, energy, pressure, ...)
    if strange_data:
        s_eps = np.array([d[1] for d in strange_data])
        s_p = np.array([d[2] for d in strange_data])
        mask = (s_eps > 0) & (s_p > 0)
        if np.any(mask):
            ax1.loglog(s_eps[mask], s_p[mask], 'r-', lw=2, label='Strange')

    # Reference line P = ε/3
    all_eps = []
    if atomic_data:
        all_eps.extend(list(a_eps[a_eps > 0]))
    if strange_data is not None:
        all_eps.extend(list(s_eps[s_eps > 0]))
    if len(all_eps) > 1:
        e_line = np.linspace(min(all_eps), max(all_eps), 100)
        ax1.loglog(e_line, e_line/3.0, 'k--', alpha=0.5, label='P = ε/3')

    ax1.set_xlabel('Energy density ε (MeV/fm³)')
    ax1.set_ylabel('Pressure P (MeV/fm³)')
    ax1.set_title('P vs ε')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # M-R (Atomic)
    if atomic_mr and len(atomic_mr) == 3:
        a_m, a_r, _ = atomic_mr
        a_m = np.array(a_m); a_r = np.array(a_r)
        if len(a_m) > 0:
            ax2.plot(a_r, a_m, 'b-', lw=3, label='Atomic')

    # M-R (Strange)
    if strange_mr and len(strange_mr) == 3:
        s_m, s_r, _ = strange_mr
        s_m = np.array(s_m); s_r = np.array(s_r)
        if len(s_m) > 0:
            ax2.plot(s_r, s_m, 'r-', lw=3, label='Strange')

    ax2.set_xlabel('Radius (km)')
    ax2.set_ylabel('Mass (M☉)')
    ax2.set_title('Mass-Radius')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(1.4, color='gray', ls=':', alpha=0.7)
    ax2.axhline(2.0, color='red', ls=':', alpha=0.7)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\nSaved '{filename}'")

# --- main ---
if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="Neutron star EOS tests")
        parser.add_argument("--atomic-only", action="store_true", help="Run only Atomic Star tests")
        args = parser.parse_args()

        print("STARTING ATOMIC STAR ANALYSIS")
        atomic_eos, atomic_eos_data = test_atomic_eos()

        print("\nSTARTING ATOMIC STAR TOV INTEGRATION")
        atomic_mr_data, _ = test_tov_integration(atomic_eos, atomic_eos_data, "Atomic")

        if not args.atomic_only:
            print("\nSTARTING STRANGE STAR ANALYSIS")
            strange_eos, strange_eos_data = test_strange_eos()

            print("\nSTARTING STRANGE STAR TOV INTEGRATION")
            strange_mr_data, _ = test_tov_integration(strange_eos, strange_eos_data, "Strange")

            plot_ep_mr(atomic_eos_data, atomic_mr_data, strange_eos_data, strange_mr_data, filename='ep_mr_atomic_strange.png')
        else:
            plot_ep_mr(atomic_eos_data, atomic_mr_data, filename='ep_mr_atomic.png')

        print("\nDone.")
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()
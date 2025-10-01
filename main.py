import numpy as np
import matplotlib.pyplot as plt
from ns.eos import EOSSolver
from ns.models import EOSType
from ns.parametrizations import ParametrizationType
from ns.tov_solver import TOVSolver

def test_atomic_eos():
    """Test the atomic neutron star EOS."""
    
    print("=" * 60)
    print("Testing Atomic Neutron Star EOS")
    print("=" * 60)
    
    # Create atomic EOS with GM1 parametrization
    atomic_eos = EOSSolver.create_eos(
        eos_type=EOSType.ATOMIC_STAR,
        parametrization=ParametrizationType.GM1,
        verbose=True
    )
    
    print(f"\nEOS Type: {atomic_eos.eos_type}")
    print(f"Parametrization: {atomic_eos.parametrization}")
    print(f"Number of baryons: {atomic_eos.n_baryons}")
    print(f"Number of leptons: {atomic_eos.n_leptons}")
    
    # Test density range (nuclear saturation density and above)
    rho_sat = 0.153  # Nuclear saturation density (fm^-3)
    densities = np.linspace(0.1 * rho_sat, 5.0 * rho_sat, 100)
    
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

def test_tov_integration(atomic_eos, eos_data):
    """Test TOV integration and generate M-R diagram."""
    
    print("\n" + "=" * 60)
    print("TOV Integration and Mass-Radius Calculation")
    print("=" * 60)
    
    if len(eos_data) < 10:
        print("Not enough EOS data points for TOV integration")
        return None, None
    
    # Create TOV solver
    tov_solver = TOVSolver(eos=atomic_eos, verbose=True)
    
    # Set up pressure range for central pressures
    densities, energies, pressures = zip(*eos_data)
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
    p_min = pressures_valid.min()
    p_max = pressures_valid.max()
    
    print(f"Valid pressure range: {p_min:.2e} - {p_max:.2e} MeV/fm³")
    
    # Generate central pressures (logarithmic spacing)
    n_stars = 50
    central_pressures = np.logspace(np.log10(p_min), np.log10(p_max * 0.9), n_stars)
    
    print(f"Computing {n_stars} neutron star models...")
    
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
    
    print(f"\nSuccessfully computed {len(masses)} neutron star models")
    
    if len(masses) > 0:
        max_mass_idx = np.argmax(masses)
        print(f"Maximum mass: {masses[max_mass_idx]:.3f} M☉ at R = {radii[max_mass_idx]:.2f} km")
        print(f"Mass range: {masses.min():.3f} - {masses.max():.3f} M☉")
        print(f"Radius range: {radii.min():.2f} - {radii.max():.2f} km")
    
    return (masses, radii, valid_central_pressures), eos_data

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
        densities, energies, pressures = zip(*eos_data)
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
            ax3.loglog(energies_pos, pressures_pos, 'go-', linewidth=2, markersize=3, label='GM1 Atomic')
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
            ax4.plot(radii, masses, 'bo-', linewidth=3, markersize=5, label='GM1 Atomic')
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

if __name__ == "__main__":
    try:
        # Test atomic EOS
        atomic_eos, eos_data = test_atomic_eos()
        
        # Test TOV integration and M-R diagram
        mr_data, _ = test_tov_integration(atomic_eos, eos_data)
        
        # Plot complete results
        plot_complete_results(eos_data, mr_data)
        
        print(f"\n{'='*60}")
        print("Complete neutron star analysis completed successfully!")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()
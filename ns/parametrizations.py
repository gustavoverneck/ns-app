"""
Nuclear matter parametrizations for neutron star equation of state models.

This module contains parametrization data for various nuclear physics models
including relativistic mean field (RMF), Skyrme-Hartree-Fock, and variational
Monte Carlo approaches.

All densities are in natural units (fm^-3) and energies in MeV.
"""

from typing import Dict, Any, List
from enum import Enum


class ParametrizationType(Enum):
    """Nuclear matter parametrizations for relativistic mean field models."""
    GM1 = "gm1"
    GM3 = "gm3"
    NL3 = "nl3"
    TM1 = "tm1"
    TM2 = "tm2"
    FSUGold = "fsugold"
    IU_FSU = "iu_fsu"
    SLy4 = "sly4"
    APR = "apr"
    MPA1 = "mpa1"
    MS1 = "ms1"
    WFF1 = "wff1"
    WFF2 = "wff2"
    ENGVIK = "engvik"
    G300 = "g300"
    BSK20 = "bsk20"
    BSK21 = "bsk21"
    CUSTOM = "custom"


# Nuclear matter parametrization database
PARAMETRIZATIONS = {
    ParametrizationType.GM1: {
        # GM1 parametrization (Glendenning & Moszkowski 1991)
        'name': 'GM1',
        'reference': 'Glendenning & Moszkowski (1991)',
        'model_type': 'relativistic_mean_field',
        'incompressibility': 300.0,    # MeV
        'symmetry_energy': 32.5,       # MeV
        'effective_mass_ratio': 0.70,  # m*/m
        'saturation_density': 0.141,   # fm^-3 (converted from 2.36e17 kg/m³)
        'binding_energy': 16.3,        # MeV
        'surface_tension': 1.1,        # MeV/fm²
        'gamma': 2.37,
        'K': 1.2e5,
        # RMF parameters
        'meson_masses': {
            'sigma': 550.0,    # MeV
            'omega': 783.0,    # MeV  
            'rho': 770.0       # MeV
        },
        'coupling_constants': {
            'g_sigma': 10.62,
            'g_omega': 12.62,
            'g_rho': 4.78
        },
        'nonlinear_parameters': {
            'g2': -10.431,     # fm⁻¹
            'g3': -28.885      # dimensionless
        }
    },
    
    ParametrizationType.GM3: {
        # GM3 parametrization (Glendenning & Moszkowski 1991)
        'name': 'GM3',
        'reference': 'Glendenning & Moszkowski (1991)',
        'model_type': 'relativistic_mean_field',
        'incompressibility': 240.0,    # MeV
        'symmetry_energy': 32.5,       # MeV
        'effective_mass_ratio': 0.78,  # m*/m
        'saturation_density': 0.137,   # fm^-3 (converted from 2.30e17 kg/m³)
        'binding_energy': 16.3,        # MeV
        'surface_tension': 1.0,        # MeV/fm²
        'gamma': 2.24,
        'K': 1.0e5,
        'meson_masses': {
            'sigma': 550.0,
            'omega': 783.0,
            'rho': 770.0
        },
        'coupling_constants': {
            'g_sigma': 9.57,
            'g_omega': 11.67,
            'g_rho': 4.63
        },
        'nonlinear_parameters': {
            'g2': -7.233,
            'g3': 0.618
        }
    },
    
    ParametrizationType.NL3: {
        # NL3 parametrization (Lalazissis et al. 1997)
        'name': 'NL3',
        'reference': 'Lalazissis et al. (1997)',
        'model_type': 'relativistic_mean_field',
        'incompressibility': 271.8,    # MeV
        'symmetry_energy': 37.4,       # MeV
        'effective_mass_ratio': 0.595, # m*/m
        'saturation_density': 0.148,   # fm^-3 (converted from 2.27e17 kg/m³)
        'binding_energy': 16.24,       # MeV
        'gamma': 2.78,
        'K': 1.8e5,
        'meson_masses': {
            'sigma': 508.194,
            'omega': 782.501,
            'rho': 763.0
        },
        'coupling_constants': {
            'g_sigma': 10.217,
            'g_omega': 12.868,
            'g_rho': 4.474
        },
        'nonlinear_parameters': {
            'g2': -10.431,
            'g3': -28.885
        }
    },
    
    ParametrizationType.TM1: {
        # TM1 parametrization (Sugahara & Toki 1994)
        'name': 'TM1',
        'reference': 'Sugahara & Toki (1994)',
        'model_type': 'relativistic_mean_field',
        'incompressibility': 281.2,    # MeV
        'symmetry_energy': 36.9,       # MeV
        'effective_mass_ratio': 0.634, # m*/m
        'saturation_density': 0.138,   # fm^-3 (converted from 2.31e17 kg/m³)
        'binding_energy': 16.26,       # MeV
        'gamma': 2.72,
        'K': 1.7e5,
        'meson_masses': {
            'sigma': 511.198,
            'omega': 783.0,
            'rho': 770.0
        },
        'coupling_constants': {
            'g_sigma': 10.028,
            'g_omega': 12.614,
            'g_rho': 4.632
        }
    },
    
    ParametrizationType.TM2: {
        # TM2 parametrization (Sugahara & Toki 1994)
        'name': 'TM2',
        'reference': 'Sugahara & Toki (1994)',
        'model_type': 'relativistic_mean_field',
        'incompressibility': 281.5,    # MeV
        'symmetry_energy': 32.0,       # MeV
        'effective_mass_ratio': 0.64,  # m*/m
        'saturation_density': 0.134,   # fm^-3 (converted from 2.24e17 kg/m³)
        'binding_energy': 16.26,       # MeV
        'gamma': 2.73,
        'K': 1.65e5,
        'meson_masses': {
            'sigma': 526.059,
            'omega': 783.0,
            'rho': 770.0
        },
        'coupling_constants': {
            'g_sigma': 9.569,
            'g_omega': 11.908,
            'g_rho': 4.256
        }
    },
    
    ParametrizationType.FSUGold: {
        # FSUGold parametrization (Todd-Rutel & Piekarewicz 2005)
        'name': 'FSUGold',
        'reference': 'Todd-Rutel & Piekarewicz (2005)',
        'model_type': 'relativistic_mean_field',
        'incompressibility': 230.0,    # MeV
        'symmetry_energy': 32.6,       # MeV
        'effective_mass_ratio': 0.61,  # m*/m
        'saturation_density': 0.148,   # fm^-3 (converted from 2.30e17 kg/m³)
        'binding_energy': 16.3,        # MeV
        'gamma': 2.15,
        'K': 9.5e4,
        'meson_masses': {
            'sigma': 491.5,
            'omega': 783.0,
            'rho': 763.0
        },
        'coupling_constants': {
            'g_sigma': 10.5975,
            'g_omega': 13.0319,
            'g_rho': 4.3838
        },
        'nonlinear_parameters': {
            'g2': -10.7,
            'g3': -39.3
        }
    },
    
    ParametrizationType.IU_FSU: {
        # IU-FSU parametrization (Fattoyev et al. 2010)
        'name': 'IU-FSU',
        'reference': 'Fattoyev et al. (2010)',
        'model_type': 'relativistic_mean_field',
        'incompressibility': 231.0,    # MeV
        'symmetry_energy': 31.3,       # MeV
        'effective_mass_ratio': 0.61,  # m*/m
        'saturation_density': 0.155,   # fm^-3 (converted from 2.30e17 kg/m³)
        'binding_energy': 16.4,        # MeV
        'gamma': 2.18,
        'K': 9.8e4,
        'slope_parameter': 60.5,       # MeV (L parameter)
        'meson_masses': {
            'sigma': 497.479,
            'omega': 782.5,
            'rho': 763.0
        },
        'coupling_constants': {
            'g_sigma': 10.444,
            'g_omega': 13.030,
            'g_rho': 4.474
        }
    },
    
    ParametrizationType.SLy4: {
        # SLy4 parametrization (Chabanat et al. 1998)
        'name': 'SLy4',
        'reference': 'Chabanat et al. (1998)',
        'model_type': 'skyrme_hartree_fock',
        'incompressibility': 229.9,    # MeV
        'symmetry_energy': 32.0,       # MeV
        'effective_mass_ratio': 0.69,  # m*/m
        'saturation_density': 0.160,   # fm^-3 (converted from 2.32e17 kg/m³)
        'binding_energy': 15.97,       # MeV
        'gamma': 2.05,
        'K': 8.8e4,
        # Skyrme parameters
        'skyrme_parameters': {
            't0': -2488.91,    # MeV⋅fm³
            't1': 486.82,      # MeV⋅fm⁵
            't2': -546.39,     # MeV⋅fm⁵
            't3': 13777.0,     # MeV⋅fm^(3+3α)
            'x0': 0.834,
            'x1': -0.344,
            'x2': -1.0,
            'x3': 1.354,
            'alpha': 1.0/6.0,
            'W0': 123.0        # MeV⋅fm⁵
        }
    },
    
    ParametrizationType.APR: {
        # APR parametrization (Akmal, Pandharipande & Ravenhall 1998)
        'name': 'APR',
        'reference': 'Akmal, Pandharipande & Ravenhall (1998)',
        'model_type': 'variational_monte_carlo',
        'incompressibility': 269.0,    # MeV
        'symmetry_energy': 34.5,       # MeV
        'effective_mass_ratio': 0.65,  # m*/m
        'saturation_density': 0.16,    # fm^-3 (standard nuclear saturation)
        'binding_energy': 16.0,        # MeV
        'gamma': 2.8,
        'K': 1.9e5,
        'potential_model': 'argonne_v18_urbana_ix'
    },
    
    ParametrizationType.MPA1: {
        # MPA1 parametrization (Müther, Prakash & Ainsworth 1987)
        'name': 'MPA1',
        'reference': 'Müther, Prakash & Ainsworth (1987)',
        'model_type': 'brueckner_hartree_fock',
        'incompressibility': 231.0,    # MeV
        'symmetry_energy': 36.8,       # MeV
        'effective_mass_ratio': 0.70,  # m*/m
        'saturation_density': 0.153,   # fm^-3 (nuclear saturation density)
        'binding_energy': 15.85,       # MeV
        'gamma': 2.12,
        'K': 9.2e4,
        'potential_model': 'paris_potential'
    },
    
    ParametrizationType.MS1: {
        # MS1 parametrization (Müller & Serot 1996)
        'name': 'MS1',
        'reference': 'Müller & Serot (1996)',
        'model_type': 'relativistic_mean_field',
        'incompressibility': 200.0,    # MeV
        'symmetry_energy': 30.0,       # MeV
        'effective_mass_ratio': 0.78,  # m*/m
        'saturation_density': 0.153,   # fm^-3 (nuclear saturation density)
        'binding_energy': 16.3,        # MeV
        'gamma': 1.95,
        'K': 7.5e4
    },
    
    ParametrizationType.WFF1: {
        # WFF1 parametrization (Wiringa, Fiks & Fabrocini 1988)
        'name': 'WFF1',
        'reference': 'Wiringa, Fiks & Fabrocini (1988)',
        'model_type': 'variational',
        'incompressibility': 250.0,    # MeV
        'symmetry_energy': 32.0,       # MeV
        'effective_mass_ratio': 0.65,  # m*/m
        'saturation_density': 0.165,   # fm^-3 (converted from 2.40e17 kg/m³)
        'binding_energy': 16.0,        # MeV
        'gamma': 2.3,
        'K': 1.1e5,
        'potential_model': 'uix_potential'
    },
    
    ParametrizationType.WFF2: {
        # WFF2 parametrization (Wiringa, Fiks & Fabrocini 1988)
        'name': 'WFF2',
        'reference': 'Wiringa, Fiks & Fabrocini (1988)',
        'model_type': 'variational',
        'incompressibility': 235.0,    # MeV
        'symmetry_energy': 32.7,       # MeV
        'effective_mass_ratio': 0.67,  # m*/m
        'saturation_density': 0.153,   # fm^-3 (nuclear saturation density)
        'binding_energy': 16.0,        # MeV
        'gamma': 2.25,
        'K': 1.0e5
    },
    
    ParametrizationType.BSK20: {
        # BSK20 parametrization (Goriely et al. 2013)
        'name': 'BSK20',
        'reference': 'Goriely et al. (2013)',
        'model_type': 'brussels_skyrme',
        'incompressibility': 241.3,    # MeV
        'symmetry_energy': 30.0,       # MeV
        'effective_mass_ratio': 0.70,  # m*/m
        'saturation_density': 0.160,   # fm^-3 (converted from 2.32e17 kg/m³)
        'binding_energy': 16.05,       # MeV
        'gamma': 2.28,
        'K': 1.05e5
    },
    
    ParametrizationType.BSK21: {
        # BSK21 parametrization (Goriely et al. 2013)
        'name': 'BSK21',
        'reference': 'Goriely et al. (2013)',
        'model_type': 'brussels_skyrme',
        'incompressibility': 220.4,    # MeV
        'symmetry_energy': 30.0,       # MeV
        'effective_mass_ratio': 0.69,  # m*/m
        'saturation_density': 0.153,   # fm^-3 (nuclear saturation density)
        'binding_energy': 16.05,       # MeV
        'gamma': 2.15,
        'K': 9.8e4
    }
}


def get_parametrization_parameters(param_type: ParametrizationType) -> Dict[str, Any]:
    """
    Get parameters for a specific nuclear matter parametrization.
    
    Args:
        param_type: Type of parametrization from ParametrizationType enum
        
    Returns:
        Dictionary containing all parameters for the specified parametrization
    """
    return PARAMETRIZATIONS.get(param_type, {}).copy()


def get_parametrization_info(param_type: ParametrizationType) -> Dict[str, str]:
    """
    Get basic information about a parametrization.
    
    Args:
        param_type: Type of parametrization from ParametrizationType enum
        
    Returns:
        Dictionary with name, reference, and model type
    """
    param_data = PARAMETRIZATIONS.get(param_type, {})
    return {
        'name': param_data.get('name', 'Unknown'),
        'reference': param_data.get('reference', 'Unknown'),
        'model_type': param_data.get('model_type', 'Unknown')
    }


def list_available_parametrizations() -> List[str]:
    """Get list of all available parametrization names."""
    return [param.value for param in ParametrizationType if param != ParametrizationType.CUSTOM]


def get_parametrizations_by_model_type(model_type: str) -> List[ParametrizationType]:
    """
    Get parametrizations that use a specific model type.
    
    Args:
        model_type: Model type (e.g., 'relativistic_mean_field', 'skyrme_hartree_fock')
        
    Returns:
        List of ParametrizationType enums matching the model type
    """
    matching_params = []
    for param_type, data in PARAMETRIZATIONS.items():
        if data.get('model_type') == model_type:
            matching_params.append(param_type)
    return matching_params


def compare_parametrizations(param_types: List[ParametrizationType]) -> Dict[str, Dict[str, float]]:
    """
    Compare key parameters across multiple parametrizations.
    
    Args:
        param_types: List of parametrization types to compare
        
    Returns:
        Dictionary with parameter comparison data
    """
    comparison = {}
    key_params = ['incompressibility', 'symmetry_energy', 'effective_mass_ratio', 
                  'saturation_density', 'binding_energy', 'gamma']
    
    for param_type in param_types:
        if param_type in PARAMETRIZATIONS:
            data = PARAMETRIZATIONS[param_type]
            comparison[data.get('name', param_type.value)] = {
                param: data.get(param, 'N/A') for param in key_params
            }
    
    return comparison


def get_rmf_parametrizations() -> List[ParametrizationType]:
    """Get all relativistic mean field parametrizations."""
    return get_parametrizations_by_model_type('relativistic_mean_field')


def get_skyrme_parametrizations() -> List[ParametrizationType]:
    """Get all Skyrme-Hartree-Fock parametrizations."""
    return get_parametrizations_by_model_type('skyrme_hartree_fock')


def get_variational_parametrizations() -> List[ParametrizationType]:
    """Get all variational parametrizations."""
    variational_types = ['variational_monte_carlo', 'variational', 'brueckner_hartree_fock']
    variational_params = []
    for model_type in variational_types:
        variational_params.extend(get_parametrizations_by_model_type(model_type))
    return variational_params


if __name__ == "__main__":
    # Demonstration of parametrizations
    print("Nuclear Matter Parametrizations Database")
    print("=" * 50)
    print("Note: All densities in fm^-3, energies in MeV")
    print()
    
    # Show all available parametrizations
    print("Available parametrizations:")
    for param in list_available_parametrizations():
        print(f"  - {param}")
    
    # Show RMF parametrizations
    print(f"\nRelativistic Mean Field parametrizations:")
    rmf_params = get_rmf_parametrizations()
    for param in rmf_params:
        info = get_parametrization_info(param)
        print(f"  - {info['name']}: {info['reference']}")
    
    # Compare some parametrizations
    print(f"\nComparison of key parametrizations:")
    comparison_params = [ParametrizationType.GM1, ParametrizationType.GM3, 
                        ParametrizationType.NL3, ParametrizationType.APR]
    comparison = compare_parametrizations(comparison_params)
    
    # Print comparison table
    headers = ['Name', 'K₀ (MeV)', 'S₀ (MeV)', 'm*/m', 'ρ₀ (fm⁻³)', 'γ']
    print(f"{headers[0]:<10} {headers[1]:<10} {headers[2]:<10} {headers[3]:<8} {headers[4]:<12} {headers[5]:<6}")
    print("-" * 65)
    
    for name, data in comparison.items():
        k0 = data.get('incompressibility', 'N/A')
        s0 = data.get('symmetry_energy', 'N/A')
        eff_mass = data.get('effective_mass_ratio', 'N/A')
        rho0 = data.get('saturation_density', 'N/A')
        gamma = data.get('gamma', 'N/A')
        print(f"{name:<10} {k0:<10} {s0:<10} {eff_mass:<8} {rho0:<12.3f} {gamma:<6}")
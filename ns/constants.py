# Conversion constants

# Math
PI = 3.141592653589793
PI2 = PI * PI

# Fundamental physical constants (CODATA 2018)
SPEED_OF_LIGHT = 2.99792458e8  # m/s
C_SI = SPEED_OF_LIGHT  # Alias for consistency
PLANCK_CONSTANT = 6.62607015e-34  # J*s
HBAR = PLANCK_CONSTANT / (2 * PI)  # J*s
ELEMENTARY_CHARGE = 1.602176634e-19  # C
AVOGADRO_NUMBER = 6.02214076e23  # 1/mol

# Gravitational constant (CODATA 2018)
G_SI = 6.67430e-11  # m³/kg/s²
GRAVITATIONAL_CONSTANT = G_SI  # Alias

# Particle physics constants
ELECTRON_MASS_KG = 9.1093837015e-31  # kg
ELECTRON_MASS_MEV = 0.51099895000  # MeV/c²
PROTON_MASS_KG = 1.67262192369e-27  # kg
NEUTRON_MASS_KG = 1.67492749804e-27  # kg

# Nucleon masses in different units
PROTON_MASS_MEV = 938.27208816  # MeV/c²
NEUTRON_MASS_MEV = 939.56542052  # MeV/c²
PROTON_MASS_AMU = 1.00727647  # atomic mass units
NEUTRON_MASS_AMU = 1.00866492  # atomic mass units

# Mass differences
NEUTRON_PROTON_MASS_DIFF_MEV = NEUTRON_MASS_MEV - PROTON_MASS_MEV  # ~1.29 MeV
NEUTRON_PROTON_MASS_DIFF_KG = NEUTRON_MASS_KG - PROTON_MASS_KG

# Average nucleon mass
AVERAGE_NUCLEON_MASS_KG = (PROTON_MASS_KG + NEUTRON_MASS_KG) / 2
AVERAGE_NUCLEON_MASS_MEV = (PROTON_MASS_MEV + NEUTRON_MASS_MEV) / 2

# Atomic mass unit
ATOMIC_MASS_UNIT_KG = 1.66053906660e-27  # kg
AMU_TO_KG = ATOMIC_MASS_UNIT_KG
KG_TO_AMU = 1 / ATOMIC_MASS_UNIT_KG

# Lepton masses (PDG 2022)
MUON_MASS_KG = 1.883531627e-28  # kg
MUON_MASS_MEV = 105.6583755     # MeV/c²
MUON_MASS_AMU = MUON_MASS_KG / ATOMIC_MASS_UNIT_KG

TAU_MASS_KG = 3.16752e-27       # kg
TAU_MASS_MEV = 1776.86          # MeV/c²
TAU_MASS_AMU = TAU_MASS_KG / ATOMIC_MASS_UNIT_KG

# Neutrino masses (approximate upper limits)
ELECTRON_NEUTRINO_MASS_KG = 3.56e-36  # kg (< 2 eV/c²)
ELECTRON_NEUTRINO_MASS_MEV = 2e-6     # MeV/c² (< 2 eV/c²)
ELECTRON_NEUTRINO_MASS_AMU = ELECTRON_NEUTRINO_MASS_KG / ATOMIC_MASS_UNIT_KG

MUON_NEUTRINO_MASS_KG = 3.38e-34      # kg (< 0.19 MeV/c²)
MUON_NEUTRINO_MASS_MEV = 0.19         # MeV/c²
MUON_NEUTRINO_MASS_AMU = MUON_NEUTRINO_MASS_KG / ATOMIC_MASS_UNIT_KG

TAU_NEUTRINO_MASS_KG = 3.20e-32       # kg (< 18 MeV/c²)
TAU_NEUTRINO_MASS_MEV = 18            # MeV/c²
TAU_NEUTRINO_MASS_AMU = TAU_NEUTRINO_MASS_KG / ATOMIC_MASS_UNIT_KG

# Baryon masses (PDG 2022)
LAMBDA_MASS_KG = 1.98e-27             # kg (1115.683 MeV/c²)
LAMBDA_MASS_MEV = 1115.683            # MeV/c²
LAMBDA_MASS_AMU = LAMBDA_MASS_KG / ATOMIC_MASS_UNIT_KG

SIGMA_PLUS_MASS_KG = 2.11e-27         # kg (1189.37 MeV/c²)
SIGMA_PLUS_MASS_MEV = 1189.37         # MeV/c²
SIGMA_PLUS_MASS_AMU = SIGMA_PLUS_MASS_KG / ATOMIC_MASS_UNIT_KG

SIGMA_ZERO_MASS_KG = 2.10e-27         # kg (1192.642 MeV/c²)
SIGMA_ZERO_MASS_MEV = 1192.642        # MeV/c²
SIGMA_ZERO_MASS_AMU = SIGMA_ZERO_MASS_KG / ATOMIC_MASS_UNIT_KG

SIGMA_MINUS_MASS_KG = 2.13e-27        # kg (1197.449 MeV/c²)
SIGMA_MINUS_MASS_MEV = 1197.449       # MeV/c²
SIGMA_MINUS_MASS_AMU = SIGMA_MINUS_MASS_KG / ATOMIC_MASS_UNIT_KG

XI_ZERO_MASS_KG = 2.34e-27            # kg (1314.86 MeV/c²)
XI_ZERO_MASS_MEV = 1314.86            # MeV/c²
XI_ZERO_MASS_AMU = XI_ZERO_MASS_KG / ATOMIC_MASS_UNIT_KG

XI_MINUS_MASS_KG = 2.35e-27           # kg (1321.71 MeV/c²)
XI_MINUS_MASS_MEV = 1321.71           # MeV/c²
XI_MINUS_MASS_AMU = XI_MINUS_MASS_KG / ATOMIC_MASS_UNIT_KG

OMEGA_MINUS_MASS_KG = 2.98e-27        # kg (1672.45 MeV/c²)
OMEGA_MINUS_MASS_MEV = 1672.45        # MeV/c²
OMEGA_MINUS_MASS_AMU = OMEGA_MINUS_MASS_KG / ATOMIC_MASS_UNIT_KG

# Useful combinations for physics calculations
HC = PLANCK_CONSTANT * SPEED_OF_LIGHT  # J*m
HBARC_MEV_FM = 197.3269804  # MeV*fm (ℏc in natural units)

# Astronomical constants
# Sun properties (IAU 2015)
SUN_MASS_KG = 1.98847e30  # kg
M_SUN_KG = SUN_MASS_KG  # Alias for consistency
SUN_RADIUS_M = 6.957e8    # m

# Energy conversions
JOULE_TO_EV = 1 / ELEMENTARY_CHARGE
EV_TO_JOULE = ELEMENTARY_CHARGE
MEV_TO_EV = 1e6
GEV_TO_EV = 1e9
EV_TO_MEV = 1e-6
EV_TO_GEV = 1e-9

# Mass conversions
MASS_KG_TO_EV = 1 / 1.78266192e-36
MASS_KG_TO_MEV = 1 / 1.78266192e-36 / 1000
MASS_EV_TO_KG = 1.78266192e-36

# Solar mass conversion
KG_TO_MSUN = 1 / SUN_MASS_KG
MSUN_TO_KG = SUN_MASS_KG

# Charge conversions
CHARGE_C_TO_E = 1 / ELEMENTARY_CHARGE

# Length conversions
KM_TO_M = 1000.0
M_TO_KM = 1 / KM_TO_M
FM_TO_M = 1e-15
M_TO_FM = 1e15

# Unit conversions for neutron star physics
# Pressure and energy density conversions
MEV_FM3_TO_J_M3 = MEV_TO_EV * EV_TO_JOULE / (FM_TO_M**3)  # MeV/fm³ to J/m³
MEV_FM3_TO_PA = MEV_FM3_TO_J_M3  # Same as J/m³ = Pa
PA_TO_MEV_FM3 = 1 / MEV_FM3_TO_PA

# Additional useful constants for neutron star physics
NEUTRON_DRIP_DENSITY_KG_M3 = 4.3e17  # kg/m³
NUCLEAR_SATURATION_DENSITY_KG_M3 = 2.8e17  # kg/m³
NUCLEAR_SATURATION_DENSITY_FM3 = 0.16  # fm⁻³
FINE_STRUCTURE_CONSTANT = 7.2973525693e-3  # dimensionless
FERMI_COUPLING_CONSTANT = 1.1663787e-5  # GeV⁻²
CRITICAL_MAGNETIC_FIELD_GAUSS = 4.4e13 # Gauss

# Conversion between different density units
KG_M3_TO_FM3 = AVERAGE_NUCLEON_MASS_KG / (NUCLEAR_SATURATION_DENSITY_KG_M3 / NUCLEAR_SATURATION_DENSITY_FM3)
FM3_TO_KG_M3 = 1 / KG_M3_TO_FM3

# Additional aliases for consistency with TOV solver
C_SI = SPEED_OF_LIGHT  # m/s
G_SI = GRAVITATIONAL_CONSTANT  # m³/kg/s²
M_SUN_KG = SUN_MASS_KG  # kg

# Numerical parameters (from QHD-II Fortran code)
DEFAULT_TOLERANCE = 1e-8                    # Convergence tolerance for field equations
DEFAULT_MAX_ITERATIONS = 1000               # Maximum iterations for BROYDN solver

# Baryon density range for calculations 
# Note: These are typical values from nuclear physics literature
# The Fortran code reads RNBINF and RNBSUP from input file
DEFAULT_BARYON_DENSITY_MIN = 0.01           # Minimum baryon density (fm⁻³)
DEFAULT_BARYON_DENSITY_MAX = 1.5            # Maximum baryon density (fm⁻³)

# Nuclear saturation density for reference
NUCLEAR_SATURATION_DENSITY = 0.153          # Nuclear saturation density (fm⁻³)

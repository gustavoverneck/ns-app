# particle.py

from .constants import *
from enum import Enum

class ParticleType(Enum):
    LEPTON = "lepton"
    BARYON = "baryon"
    MESON = "meson"
    GAUGE_BOSON = "gauge_boson"

class LeptonType(Enum):
    CHARGED_LEPTON = "charged_lepton"
    NEUTRINO = "neutrino"

class BaryonType(Enum):
    NUCLEON = "nucleon"
    HYPERON = "hyperon"

class Particle:
    def __init__(self, id: str, mass: float, charge: float, spin: float, 
                 particle_type: ParticleType, subtype=None, generation=None, 
                 isospin=None, isospin_z=None, strangeness=None):
        self.id: str = id
        self.mass: float = mass
        self.charge: float = charge
        self.spin: float = spin
        self.particle_type: ParticleType = particle_type
        self.subtype = subtype
        self.generation = generation
        self.isospin = isospin  # Total isospin I
        self.isospin_z = isospin_z  # Third component of isospin I_z
        self.strangeness = strangeness  # Strangeness quantum number

    def get_name(self) -> str:
        return self.id

    def get_mass(self) -> float:
        return self.mass

    def get_charge(self) -> float:
        return self.charge

    def get_spin(self) -> float:
        return self.spin

    def get_particle_type(self) -> ParticleType:
        return self.particle_type

    def get_subtype(self):
        return self.subtype

    def get_generation(self):
        return self.generation

    def get_isospin(self):
        return self.isospin

    def get_isospin_z(self):
        return self.isospin_z

    def get_strangeness(self):
        return self.strangeness

    def is_lepton(self) -> bool:
        return self.particle_type == ParticleType.LEPTON

    def is_baryon(self) -> bool:
        return self.particle_type == ParticleType.BARYON

    def is_charged_lepton(self) -> bool:
        return (self.particle_type == ParticleType.LEPTON and 
                self.subtype == LeptonType.CHARGED_LEPTON)

    def is_neutrino(self) -> bool:
        return (self.particle_type == ParticleType.LEPTON and 
                self.subtype == LeptonType.NEUTRINO)

    def is_nucleon(self) -> bool:
        return (self.particle_type == ParticleType.BARYON and 
                self.subtype == BaryonType.NUCLEON)

    def is_hyperon(self) -> bool:
        return (self.particle_type == ParticleType.BARYON and 
                self.subtype == BaryonType.HYPERON)

    def is_isospin_singlet(self) -> bool:
        """Check if particle is an isospin singlet (I = 0)"""
        return self.isospin == 0 if self.isospin is not None else False

    def is_isospin_doublet(self) -> bool:
        """Check if particle is in an isospin doublet (I = 1/2)"""
        return self.isospin == 0.5 if self.isospin is not None else False

    def is_isospin_triplet(self) -> bool:
        """Check if particle is in an isospin triplet (I = 1)"""
        return self.isospin == 1.0 if self.isospin is not None else False

    def has_strangeness(self) -> bool:
        """Check if particle has non-zero strangeness"""
        return self.strangeness is not None and self.strangeness != 0


class ParticlesData:
    def __init__(self):
        self.init_electron()
        self.init_muon()
        self.init_tau()
        self.init_electron_neutrino()
        self.init_muon_neutrino()
        self.init_tau_neutrino()
        self.init_proton()
        self.init_neutron()
        self.init_lambda()
        self.init_sigma_plus()
        self.init_sigma_zero()
        self.init_sigma_minus()
        self.init_xi_zero()
        self.init_xi_minus()
        self.init_omega_minus()

    @staticmethod
    def mass_kg_to_natural(mass_kg):
        """Convert mass from kg to natural units (eV)"""
        return mass_kg * MASS_KG_TO_EV

    @staticmethod
    def charge_c_to_natural(charge_c):
        """Convert charge from Coulombs to natural units (in units of e)"""
        return charge_c / ELEMENTARY_CHARGE

    @staticmethod
    def mass_natural_to_kg(mass_ev):
        return mass_ev * MASS_EV_TO_KG

    @staticmethod
    def charge_natural_to_c(charge_e):
        return charge_e * ELEMENTARY_CHARGE

    def init_electron(self):
        self.e = Particle(
            id="Electron",
            mass=ELECTRON_MASS_MEV,  # Use MEV constant
            charge=-1.0,  # in units of e
            spin=0.5,
            particle_type=ParticleType.LEPTON,
            subtype=LeptonType.CHARGED_LEPTON,
            generation=1,
            isospin=0.5,  # Electron doublet
            isospin_z=-0.5,  # Lower component of doublet
            strangeness=0  # Leptons have no strangeness
        )

    def init_muon(self):
        self.mu = Particle(
            id="Muon",
            mass=MUON_MASS_MEV,  # Use MEV constant
            charge=-1.0,  # in units of e
            spin=0.5,
            particle_type=ParticleType.LEPTON,
            subtype=LeptonType.CHARGED_LEPTON,
            generation=2,
            isospin=0.5,  # Muon doublet
            isospin_z=-0.5,  # Lower component of doublet
            strangeness=0  # Leptons have no strangeness
        )

    def init_tau(self):
        self.tau = Particle(
            id="Tau",
            mass=TAU_MASS_MEV,  # Use MEV constant
            charge=-1.0,  # in units of e
            spin=0.5,
            particle_type=ParticleType.LEPTON,
            subtype=LeptonType.CHARGED_LEPTON,
            generation=3,
            isospin=0.5,  # Tau doublet
            isospin_z=-0.5,  # Lower component of doublet
            strangeness=0  # Leptons have no strangeness
        )

    def init_electron_neutrino(self):
        self.ve = Particle(
            id="Electron Neutrino",
            mass=ELECTRON_NEUTRINO_MASS_MEV,  # Use MEV constant
            charge=0.0,  # in units of e
            spin=0.5,
            particle_type=ParticleType.LEPTON,
            subtype=LeptonType.NEUTRINO,
            generation=1,
            isospin=0.5,  # Electron doublet
            isospin_z=0.5,  # Upper component of doublet
            strangeness=0  # Leptons have no strangeness
        )

    def init_muon_neutrino(self):
        self.vmu = Particle(
            id="Muon Neutrino",
            mass=MUON_NEUTRINO_MASS_MEV,  # Use MEV constant
            charge=0.0,  # in units of e
            spin=0.5,
            particle_type=ParticleType.LEPTON,
            subtype=LeptonType.NEUTRINO,
            generation=2,
            isospin=0.5,  # Muon doublet
            isospin_z=0.5,  # Upper component of doublet
            strangeness=0  # Leptons have no strangeness
        )

    def init_tau_neutrino(self):
        self.vtau = Particle(
            id="Tau Neutrino",
            mass=TAU_NEUTRINO_MASS_MEV,  # Use MEV constant
            charge=0.0,  # in units of e
            spin=0.5,
            particle_type=ParticleType.LEPTON,
            subtype=LeptonType.NEUTRINO,
            generation=3,
            isospin=0.5,  # Tau doublet
            isospin_z=0.5,  # Upper component of doublet
            strangeness=0  # Leptons have no strangeness
        )

    def init_proton(self):
        self.p = Particle(
            id="Proton",
            mass=PROTON_MASS_MEV,  # Use MEV constant
            charge=+1.0,  # in units of e
            spin=0.5,
            particle_type=ParticleType.BARYON,
            subtype=BaryonType.NUCLEON,
            generation=1,
            isospin=0.5,  # Nucleon doublet
            isospin_z=0.5,  # Upper component (I_z = +1/2)
            strangeness=0  # Nucleons have no strangeness
        )

    def init_neutron(self):
        self.n = Particle(
            id="Neutron", 
            mass=NEUTRON_MASS_MEV,  # Use MEV constant
            charge=0.0,  # in units of e
            spin=0.5,
            particle_type=ParticleType.BARYON,
            subtype=BaryonType.NUCLEON,
            generation=1,
            isospin=0.5,  # Nucleon doublet
            isospin_z=-0.5,  # Lower component (I_z = -1/2)
            strangeness=0  # Nucleons have no strangeness
        )

    def init_lambda(self):
        self.lambda_baryon = Particle(
            id="Lambda",
            mass=LAMBDA_MASS_MEV,  # Use MEV constant
            charge=0.0,  # in units of e
            spin=0.5,
            particle_type=ParticleType.BARYON,
            subtype=BaryonType.HYPERON,
            generation=2,
            isospin=0.0,  # Lambda is isospin singlet
            isospin_z=0.0,
            strangeness=-1  # Lambda has one strange quark
        )

    def init_sigma_plus(self):
        self.sigma_plus = Particle(
            id="Sigma+",
            mass=SIGMA_PLUS_MASS_MEV,  # Use MEV constant
            charge=+1.0,  # in units of e
            spin=0.5,
            particle_type=ParticleType.BARYON,
            subtype=BaryonType.HYPERON,
            generation=2,
            isospin=1.0,  # Sigma triplet
            isospin_z=1.0,  # I_z = +1
            strangeness=-1  # Sigma has one strange quark
        )

    def init_sigma_zero(self):
        self.sigma_zero = Particle(
            id="Sigma0",
            mass=SIGMA_ZERO_MASS_MEV,  # Use MEV constant
            charge=0.0,  # in units of e
            spin=0.5,
            particle_type=ParticleType.BARYON,
            subtype=BaryonType.HYPERON,
            generation=2,
            isospin=1.0,  # Sigma triplet
            isospin_z=0.0,  # I_z = 0
            strangeness=-1  # Sigma has one strange quark
        )

    def init_sigma_minus(self):
        self.sigma_minus = Particle(
            id="Sigma-",
            mass=SIGMA_MINUS_MASS_MEV,  # Use MEV constant
            charge=-1.0,  # in units of e
            spin=0.5,
            particle_type=ParticleType.BARYON,
            subtype=BaryonType.HYPERON,
            generation=2,
            isospin=1.0,  # Sigma triplet
            isospin_z=-1.0,  # I_z = -1
            strangeness=-1  # Sigma has one strange quark
        )

    def init_xi_zero(self):
        self.xi_zero = Particle(
            id="Xi0",
            mass=XI_ZERO_MASS_MEV,  # Use MEV constant
            charge=0.0,  # in units of e
            spin=0.5,
            particle_type=ParticleType.BARYON,
            subtype=BaryonType.HYPERON,
            generation=2,
            isospin=0.5,  # Xi doublet
            isospin_z=0.5,  # I_z = +1/2
            strangeness=-2  # Xi has two strange quarks
        )

    def init_xi_minus(self):
        self.xi_minus = Particle(
            id="Xi-",
            mass=XI_MINUS_MASS_MEV,  # Use MEV constant
            charge=-1.0,  # in units of e
            spin=0.5,
            particle_type=ParticleType.BARYON,
            subtype=BaryonType.HYPERON,
            generation=2,
            isospin=0.5,  # Xi doublet
            isospin_z=-0.5,  # I_z = -1/2
            strangeness=-2  # Xi has two strange quarks
        )

    def init_omega_minus(self):
        self.omega_minus = Particle(
            id="Omega-",
            mass=OMEGA_MINUS_MASS_MEV,  # Use MEV constant
            charge=-1.0,  # in units of e
            spin=1.5,  # Omega- has spin 3/2
            particle_type=ParticleType.BARYON,
            subtype=BaryonType.HYPERON,
            generation=3,
            isospin=0.0,  # Omega is isospin singlet
            isospin_z=0.0,
            strangeness=-3  # Omega has three strange quarks
        )

    def get_all_particles(self):
        """Return dictionary of all particles"""
        return {
            'electron': self.e,
            'muon': self.mu,
            'tau': self.tau,
            'electron_neutrino': self.ve,
            'muon_neutrino': self.vmu,
            'tau_neutrino': self.vtau,
            'proton': self.p,
            'neutron': self.n,
            'lambda': self.lambda_baryon,
            'sigma_plus': self.sigma_plus,
            'sigma_zero': self.sigma_zero,
            'sigma_minus': self.sigma_minus,
            'xi_zero': self.xi_zero,
            'xi_minus': self.xi_minus,
            'omega_minus': self.omega_minus
        }

    def get_leptons(self):
        """Return dictionary of all leptons"""
        particles = self.get_all_particles()
        return {name: particle for name, particle in particles.items() 
                if particle.is_lepton()}

    def get_baryons(self):
        """Return dictionary of all baryons"""
        particles = self.get_all_particles()
        return {name: particle for name, particle in particles.items() 
                if particle.is_baryon()}

    def get_charged_leptons(self):
        """Return dictionary of all charged leptons"""
        particles = self.get_all_particles()
        return {name: particle for name, particle in particles.items() 
                if particle.is_charged_lepton()}

    def get_neutrinos(self):
        """Return dictionary of all neutrinos"""
        particles = self.get_all_particles()
        return {name: particle for name, particle in particles.items() 
                if particle.is_neutrino()}

    def get_nucleons(self):
        """Return dictionary of all nucleons"""
        particles = self.get_all_particles()
        return {name: particle for name, particle in particles.items() 
                if particle.is_nucleon()}

    def get_hyperons(self):
        """Return dictionary of all hyperons"""
        particles = self.get_all_particles()
        return {name: particle for name, particle in particles.items() 
                if particle.is_hyperon()}

    def get_particles_by_generation(self, generation):
        """Return particles of a specific generation"""
        particles = self.get_all_particles()
        return {name: particle for name, particle in particles.items() 
                if particle.generation == generation}

    def get_isospin_singlets(self):
        """Return particles with isospin I = 0"""
        particles = self.get_all_particles()
        return {name: particle for name, particle in particles.items() 
                if particle.is_isospin_singlet()}

    def get_isospin_doublets(self):
        """Return particles with isospin I = 1/2"""
        particles = self.get_all_particles()
        return {name: particle for name, particle in particles.items() 
                if particle.is_isospin_doublet()}

    def get_isospin_triplets(self):
        """Return particles with isospin I = 1"""
        particles = self.get_all_particles()
        return {name: particle for name, particle in particles.items() 
                if particle.is_isospin_triplet()}

    def get_particles_by_strangeness(self, strangeness):
        """Return particles with specific strangeness"""
        particles = self.get_all_particles()
        return {name: particle for name, particle in particles.items() 
                if particle.strangeness == strangeness}

    def get_strange_particles(self):
        """Return particles with non-zero strangeness"""
        particles = self.get_all_particles()
        return {name: particle for name, particle in particles.items() 
                if particle.has_strangeness()}

    def print_all_particles(self):
        """Print all particle properties"""
        particles = self.get_all_particles()
        print("Particle Properties (Natural Units)")
        print("=" * 110)
        print(f"{'Name':15s} {'Mass (MeV)':>10s} {'Charge':>8s} {'Spin':>6s} "
              f"{'Type':>12s} {'Subtype':>12s} {'Gen':>4s} {'I':>6s} {'I_z':>6s} {'S':>4s}")
        print("-" * 110)
        
        for name, particle in particles.items():
            subtype_str = particle.subtype.value if particle.subtype else "None"
            isospin_str = f"{particle.isospin}" if particle.isospin is not None else "None"
            isospin_z_str = f"{particle.isospin_z}" if particle.isospin_z is not None else "None"
            strangeness_str = f"{particle.strangeness}" if particle.strangeness is not None else "None"
            
            print(f"{particle.id:15s} {particle.mass:>10.3f} "
                  f"{particle.charge:>+8.1f}e {particle.spin:>6.1f} "
                  f"{particle.particle_type.value:>12s} {subtype_str:>12s} "
                  f"{particle.generation:>4d} {isospin_str:>6s} {isospin_z_str:>6s} "
                  f"{strangeness_str:>4s}")

    def print_by_species(self):
        """Print particles organized by species"""
        print("\nParticles by Species")
        print("=" * 60)
        
        print("\nLEPTONS:")
        print("-" * 40)
        leptons = self.get_leptons()
        for name, particle in leptons.items():
            subtype_str = particle.subtype.value if particle.subtype else "None"
            print(f"  {particle.id:15s} (Gen {particle.generation}, {subtype_str}, "
                  f"I={particle.isospin}, I_z={particle.isospin_z}, S={particle.strangeness})")
        
        print("\nBARYONS:")
        print("-" * 40)
        baryons = self.get_baryons()
        for name, particle in baryons.items():
            subtype_str = particle.subtype.value if particle.subtype else "None"
            print(f"  {particle.id:15s} (Gen {particle.generation}, {subtype_str}, "
                  f"I={particle.isospin}, I_z={particle.isospin_z}, S={particle.strangeness})")

    def print_by_isospin(self):
        """Print particles organized by isospin multiplets"""
        print("\nParticles by Isospin Multiplets")
        print("=" * 60)
        
        print("\nISOSPIN SINGLETS (I = 0):")
        print("-" * 40)
        singlets = self.get_isospin_singlets()
        for name, particle in singlets.items():
            print(f"  {particle.id:15s} (I=0, I_z=0, S={particle.strangeness})")
        
        print("\nISOSPIN DOUBLETS (I = 1/2):")
        print("-" * 40)
        doublets = self.get_isospin_doublets()
        for name, particle in doublets.items():
            print(f"  {particle.id:15s} (I=1/2, I_z={particle.isospin_z}, S={particle.strangeness})")
        
        print("\nISOSPIN TRIPLETS (I = 1):")
        print("-" * 40)
        triplets = self.get_isospin_triplets()
        for name, particle in triplets.items():
            print(f"  {particle.id:15s} (I=1, I_z={particle.isospin_z}, S={particle.strangeness})")

    def print_by_strangeness(self):
        """Print particles organized by strangeness"""
        print("\nParticles by Strangeness")
        print("=" * 50)
        
        # Group by strangeness values
        strangeness_values = sorted(set(p.strangeness for p in self.get_all_particles().values() 
                                      if p.strangeness is not None))
        
        for s in strangeness_values:
            particles_with_s = self.get_particles_by_strangeness(s)
            print(f"\nSTRANGENESS S = {s}:")
            print("-" * 30)
            for name, particle in particles_with_s.items():
                print(f"  {particle.id:15s} (Charge={particle.charge:+.1f}e, "
                      f"I={particle.isospin}, I_z={particle.isospin_z})")
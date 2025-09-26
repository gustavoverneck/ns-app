# Neutron Star Solver

A Python package for solving the structure of neutron stars using the Tolman-Oppenheimer-Volkoff (TOV) equations with comprehensive particle physics support.

## Features

- **TBA**
## Installation

### Using uv (recommended)

```bash
# Clone the repository
git clone https://github.com/gustavoverneck/ns-app
cd ns-app

# Set up virtual environment and install dependencies
uv venv
source .venv/bin/activate
uv sync
```

### Using pip

```bash
# Clone the repository
git clone <repository-url>
cd ns-app

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```


## Physical Constants

All calculations use natural units (ℏ = c = 1).

## Particle Database

Includes complete Standard Model particles relevant for neutron stars:

### Leptons
- Charged leptons: electron, muon, tau
- Neutrinos: νₑ, νᵤ, νₜ

### Baryons
- **Nucleons**: proton, neutron
- **Hyperons**: Λ (S=-1), Σ⁺/Σ⁰/Σ⁻ (S=-1), Ξ⁰/Ξ⁻ (S=-2), Ω⁻ (S=-3)

Each particle includes:
- Mass, charge, spin
- Isospin quantum numbers (I, Iᵤ)
- Strangeness quantum number (S)
- Generation and classification

## Requirements

- ...


## License

MIT License

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## References

- ...

## Contact

For questions and contributions, please open an issue on GitHub.
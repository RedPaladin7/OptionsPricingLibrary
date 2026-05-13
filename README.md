# Options Pricing Library

A comprehensive, from-scratch Python library for options pricing, risk management, and quantitative finance. Built with modern Python practices, featuring analytical models, Monte Carlo simulations, finite difference methods, and an interactive Streamlit dashboard.

## Features

### Core Pricing Models
- **Black-Scholes**: Analytical European option pricing with all Greeks
- **Heston Model**: Stochastic volatility model with closed-form and Monte Carlo implementations
- **Monte Carlo**: Generic path-dependent option pricer with variance reduction
- **Local Volatility**: Dupire model calibration and Monte Carlo pricing
- **Finite Difference**: Crank-Nicolson PDE solver for American and barrier options

### Instruments Supported
- European Options (calls/puts)
- American Options (calls/puts)
- Barrier Options (knock-in/knock-out, single/double barriers)
- Asian Options (arithmetic/geometric averaging)

### Risk Management
- **Greeks**: Delta, Gamma, Vega, Theta, Rho, Vanna, Volga, Charm (scalar and surface)
- **P&L Attribution**: Taylor decomposition and backtesting
- **Vega Matrix**: Pillar-by-pillar volatility sensitivity analysis

### Market Data & Calibration
- **Volatility Surface**: SVI parameterization and calibration to option chains
- **Implied Volatility**: Newton-Raphson solver for Black-Scholes IV
- **Live Data**: Yahoo Finance integration for real-time option chains

### Numerical Methods
- **FFT**: Fast Fourier Transform for characteristic function pricing
- **LSMC**: Longstaff-Schwartz algorithm for American options
- **Heston Simulator**: Exact simulation of Heston paths
- **Local Vol Simulator**: Euler discretization for local volatility

## Installation

### Prerequisites
- Python 3.8+
- pip

### Install from Source
```bash
git clone https://github.com/yourusername/OptionsPricingLibrary.git
cd OptionsPricingLibrary
pip install -e .
```

### Dependencies
Core dependencies (automatically installed):
- numpy>=1.24.0
- scipy>=1.10.0
- pandas>=2.0.0

Dashboard dependencies:
- streamlit>=1.32.0
- plotly>=5.18.0
- yfinance>=0.2.36

## Quick Start

### Basic European Option Pricing
```python
from options_lib import BlackScholes, EuropeanOption, MarketData, OptionType

# Create model with 20% volatility
model = BlackScholes(sigma=0.20)

# Define ATM call option
call = EuropeanOption(strike=100, expiry=1.0, option_type=OptionType.CALL)

# Market data
mkt = MarketData(spot=100, rate=0.05, div_yield=0.0)

# Price the option
price = model.price(call, mkt)
print(f"Call price: ${price:.2f}")  # Call price: $10.45
```

### Calculate Greeks
```python
# All Greeks available
delta = model.delta(call, mkt)
gamma = model.gamma(call, mkt)
vega = model.vega(call, mkt)
print(f"Delta: {delta:.3f}, Gamma: {gamma:.3f}, Vega: {vega:.3f}")
```

### Advanced: Heston Model
```python
from options_lib import Heston, HestonParams

# Heston parameters
params = HestonParams(
    v0=0.04,    # Initial variance
    kappa=2.0,  # Mean reversion speed
    v_bar=0.04, # Long-term variance
    xi=0.3,     # Volatility of variance
    rho=-0.7    # Correlation
)

heston_model = Heston(params)
price = heston_model.price(call, mkt)
```

### Monte Carlo Pricing
```python
from options_lib import MonteCarlo
from options_lib.instruments.american import AmericanOption

# American put option
put = AmericanOption(strike=100, expiry=1.0, option_type=OptionType.PUT)

# Monte Carlo with 100k paths
mc_model = MonteCarlo(n_paths=100000, n_steps=252)
price = mc_model.price(put, mkt)
```

## Interactive Dashboard

Launch the Streamlit dashboard for visualization and interactive pricing:

```bash
cd OptionsPricingLibrary
streamlit run dashboard/app.py
```

### Dashboard Pages
1. **Vol Surface**: SVI calibration, 3D volatility surface, risk-neutral density
2. **Greek Surface**: Heatmaps for all Greeks across strike/expiry space
3. **Vega Matrix**: Sensitivity analysis for volatility surface pillars
4. **Model Risk**: Compare local vol vs Black-Scholes, Heston vs Black-Scholes
5. **P&L Attribution**: Taylor expansion decomposition and backtesting

## API Reference

### Main Classes
- `BlackScholes(sigma)`: Analytical European pricer
- `Heston(params)`: Stochastic volatility model
- `MonteCarlo(n_paths, n_steps)`: Path-dependent pricer
- `EuropeanOption(strike, expiry, option_type)`
- `AmericanOption(strike, expiry, option_type)`
- `BarrierOption(strike, expiry, option_type, barrier, barrier_type)`
- `AsianOption(strike, expiry, option_type, average_type)`
- `MarketData(spot, rate, div_yield)`

### Key Methods
- `model.price(instrument, market)`: Price an option
- `model.delta/instrument, market)`: Calculate Greeks
- `implied_vol_bs(price, instrument, market)`: Solve for implied volatility

## Testing

Run the full test suite:
```bash
pytest options_lib/tests/
```

Run specific tests:
```bash
pytest options_lib/tests/test_black_scholes.py -v
```

## Project Structure

```
OptionsPricingLibrary/
├── options_lib/              # Main library package
│   ├── instruments/          # Option contracts
│   ├── models/              # Pricing models
│   ├── numerics/            # Numerical methods
│   ├── market_data/         # Vol surfaces, data fetching
│   ├── risk/                # Greeks, P&L attribution
│   └── tests/               # Unit tests
├── dashboard/               # Streamlit app
│   ├── app.py              # Main dashboard
│   ├── pages/              # Dashboard pages
│   └── requirements.txt    # Dashboard dependencies
├── pyproject.toml          # Package configuration
└── conftest.py             # pytest configuration
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Acknowledgments

Built for educational and research purposes in quantitative finance. Implements algorithms from academic literature including:
- Black-Scholes (1973)
- Heston (1993)
- Dupire (1994)
- Longstaff-Schwartz (2001)
- Gatheral SVI (2004)</content>
<filePath">/home/redpaladin/projects/OptionsPricingLibrary/README.md
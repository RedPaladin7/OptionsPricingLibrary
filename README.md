# Options Pricing Library

A comprehensive, production-ready Python library for quantitative finance, implementing state-of-the-art options pricing models, risk management tools, and numerical methods. Built from scratch with modern Python practices, featuring analytical solutions, Monte Carlo simulations, PDE solvers, and an interactive visualization dashboard.

## What This Library Solves

### The Core Problem: Pricing Complex Derivatives
Traditional Black-Scholes assumes constant volatility, but real markets exhibit:
- **Volatility smiles**: Options at different strikes have different implied volatilities
- **Term structure**: Volatility varies by expiry date
- **Stochastic volatility**: Volatility itself changes over time
- **Path dependence**: Options whose payoff depends on the entire price path (Asians, barriers)
- **Early exercise**: American options can be exercised before expiry

This library provides multiple mathematical frameworks to handle these complexities, each with different strengths and computational approaches.

### Risk Management Challenges
- **Greek calculation**: Understanding sensitivity to market moves
- **P&L attribution**: Explaining daily profits/losses using risk metrics
- **Model risk**: Comparing different pricing models for consistency
- **Hedge effectiveness**: Ensuring portfolios are properly delta-hedged

## Detailed Capabilities

### 1. Analytical Pricing Models

#### Black-Scholes Model
**What it does**: Provides exact closed-form solutions for European options using the geometric Brownian motion assumption.

**Mathematical foundation**:
```
dS = (r - q)S dt + σ S dW
```
Where volatility σ is constant.

**Capabilities**:
- European call/put pricing in milliseconds
- All first-order Greeks (Delta, Gamma, Vega, Theta, Rho)
- Second-order Greeks (Vanna, Volga, Charm)
- Implied volatility calculation via Newton-Raphson

**When to use**: Benchmark pricing, fast calculations, understanding basic option behavior.

#### Heston Stochastic Volatility Model
**What it does**: Models volatility as a stochastic process that correlates with the underlying asset.

**Mathematical foundation**:
```
dS = (r - q)S dt + √v S dW¹
dv = κ(v̄ - v) dt + ξ √v dW²
dW¹ dW² = ρ dt
```

**Parameters**:
- `v0`: Initial variance (not volatility!)
- `kappa`: Mean reversion speed
- `v_bar`: Long-term variance level
- `xi`: Volatility of variance ("vol of vol")
- `rho`: Correlation between spot and volatility

**Capabilities**:
- Semi-closed-form pricing via FFT (Carr-Madan method)
- Captures volatility clustering and leverage effects
- Generates realistic volatility smiles and term structures
- Calibration to market option chains

**Problem solved**: Explains why out-of-the-money puts are expensive (negative skew from ρ < 0).

### 2. Monte Carlo Simulation Engine

**What it does**: Simulates thousands of possible future price paths to estimate option values through statistical sampling.

**Mathematical foundation**:
```
V₀ = e^{-rT} E[payoff(S_T)]
≈ e^{-rT} (1/N) Σ payoff(pathᵢ)
```

**Capabilities**:
- **Variance reduction techniques**:
  - Control variates (uses Black-Scholes as control)
  - Antithetic variates (simulates mirror paths)
- **Path-dependent options**: Asian, barrier, lookback options
- **Early exercise**: American options via Longstaff-Schwartz algorithm
- **Multi-asset options**: Basket, spread options

**Problem solved**: Handles complex payoffs that defy analytical solutions.

### 3. Local Volatility Model (Dupire)

**What it does**: Extracts a deterministic volatility function σ(S,t) that exactly reproduces all market European option prices.

**Mathematical foundation** (Dupire's formula):
```
σ_loc²(K,T) = ∂w/∂T / [1 - k/w ∂w/∂k + ¼(-¼ - 1/w + k²/w²)(∂w/∂k)² + ½ ∂²w/∂k²]
```
Where w(k,T) is the total implied variance from the market surface.

**Capabilities**:
- Perfect calibration to any arbitrage-free volatility surface
- Monte Carlo pricing of exotic options
- Forward smile modeling (how volatility evolves)
- Barrier option pricing with realistic dynamics

**Problem solved**: "What volatility should I use?" - Local vol tells you the exact volatility at any spot/time point.

### 4. Finite Difference PDE Solver

**What it does**: Solves the Black-Scholes partial differential equation numerically on a grid.

**Mathematical foundation** (Crank-Nicolson scheme):
```
[V^{n+1} - V^n]/Δt = ½(L[V^n] + L[V^{n+1}])
```
Where L is the BS operator: (r-q)S ∂V/∂S + ½σ²S² ∂²V/∂S² - rV

**Capabilities**:
- American options with early exercise
- Barrier options with continuous monitoring
- Dividend-paying stocks
- Time-dependent parameters

**Problem solved**: American options and complex boundary conditions.

### 5. Volatility Surface Construction

**What it does**: Fits smooth, arbitrage-free volatility surfaces to market option data.

**SVI Parametrization** (Gatheral 2004):
```
w(k) = a + b[ρ(k-m) + √((k-m)² + σ²)]
```
Where k = ln(K/F) is log-moneyness.

**Capabilities**:
- Calibration to option chains from Yahoo Finance
- Interpolation at any strike/expiry
- Risk-neutral density extraction
- Arbitrage detection and prevention

**Problem solved**: Consistent pricing across all strikes and expiries.

### 6. Risk Management & Greeks

#### Greek Calculation
**What it does**: Computes option sensitivities using analytical formulas or finite differences.

**Available Greeks**:
- **Delta**: Sensitivity to spot price changes
- **Gamma**: Rate of change of delta (convexity)
- **Vega**: Sensitivity to volatility changes
- **Theta**: Time decay
- **Rho**: Sensitivity to interest rates
- **Vanna**: Delta-volatility cross sensitivity
- **Volga**: Volatility convexity

#### Greek Surfaces
Computes Greeks across entire strike/expiry grids for portfolio risk analysis.

#### P&L Attribution
**What it does**: Explains daily P&L using Taylor expansion of Greeks.

**Mathematical foundation**:
```
dV ≈ Δ·dS + ½Γ·dS² + V·dσ + Θ·dt + Vanna·dS·dσ + ½·Volga·dσ²
```

**Capabilities**:
- Decomposes P&L into directional, convexity, and volatility components
- Identifies model risk (unexplained P&L)
- Validates hedging strategies

**Problem solved**: "Why did I make/lose money today?"

### 7. Interactive Dashboard

**Streamlit application** with 5 specialized pages:

1. **Volatility Surface**: SVI calibration, 3D visualization, risk-neutral density
2. **Greek Surfaces**: Heatmaps of all Greeks across strike/expiry space
3. **Vega Matrix**: Pillar-by-pillar volatility sensitivity analysis
4. **Model Risk**: Compare local vol vs Black-Scholes, Heston vs Black-Scholes
5. **P&L Attribution**: Taylor decomposition with backtesting

## Installation & Setup

### Prerequisites
```bash
Python 3.8+
pip
```

### Installation
```bash
git clone <repository-url>
cd OptionsPricingLibrary
pip install -e .
```

### Dependencies
**Core library**:
- numpy>=1.24.0 (numerical computing)
- scipy>=1.10.0 (optimization, special functions)
- pandas>=2.0.0 (data manipulation)

**Dashboard**:
- streamlit>=1.32.0 (web interface)
- plotly>=5.18.0 (interactive charts)
- yfinance>=0.2.36 (market data)

## Comprehensive Usage Guide

### 1. Basic European Option Pricing

```python
from options_lib import BlackScholes, EuropeanOption, MarketData, OptionType

# Initialize model with constant 20% volatility
model = BlackScholes(sigma=0.20)

# Define an at-the-money call option
call = EuropeanOption(
    strike=100.0,      # Strike price
    expiry=1.0,        # Time to expiry in years
    option_type=OptionType.CALL
)

# Market conditions
market = MarketData(
    spot=100.0,        # Current stock price
    rate=0.05,         # Risk-free rate (5%)
    div_yield=0.02     # Dividend yield (2%)
)

# Calculate price
price = model.price(call, market)
print(f"Call price: ${price:.2f}")  # $8.34
```

### 2. Complete Greek Analysis

```python
from options_lib.risk.greeks import GreekEngine

# Create Greek calculation engine
engine = GreekEngine(model)

# Compute all Greeks at once
greeks = engine.all_greeks(call, market)

print("Complete Greek profile:")
for greek, value in greeks.items():
    print(f"{greek.capitalize()}: {value:.4f}")
```

### 3. Stochastic Volatility with Heston

```python
from options_lib import Heston, HestonParams

# Heston parameters (calibrated to typical equity surface)
params = HestonParams(
    v0=0.04,      # Initial variance (σ₀² = 0.04 → 20% vol)
    kappa=2.0,    # Mean reversion speed
    v_bar=0.04,   # Long-term variance
    xi=0.3,       # Volatility of variance
    rho=-0.7      # Negative correlation (leverage effect)
)

# Check Feller condition (variance stays positive)
print(f"Feller satisfied: {params.feller_satisfied}")

heston_model = Heston(params)
price = heston_model.price(call, market)
print(f"Heston price: ${price:.2f}")  # More expensive due to vol uncertainty
```

### 4. Monte Carlo for Path-Dependent Options

```python
from options_lib import MonteCarlo
from options_lib.instruments.asian import AsianOption, AverageType

# Asian option (arithmetic average)
asian_call = AsianOption(
    strike=100.0,
    expiry=1.0,
    option_type=OptionType.CALL,
    average_type=AverageType.ARITHMETIC
)

# Monte Carlo with variance reduction
mc_model = MonteCarlo(
    n_paths=50000,     # Number of simulated paths
    n_steps=252,       # Daily time steps
    use_control_variates=True,  # Variance reduction
    use_antithetic=True         # Additional variance reduction
)

price = mc_model.price(asian_call, market)
print(f"Asian call price: ${price:.2f}")
```

### 5. American Options with Early Exercise

```python
from options_lib.instruments.american import AmericanOption
from options_lib.numerics import CrankNicolson

# American put option
american_put = AmericanOption(
    strike=100.0,
    expiry=1.0,
    option_type=OptionType.PUT
)

# Finite difference solver
fd_solver = CrankNicolson(
    n_spot_steps=200,    # Spatial grid points
    n_time_steps=100     # Time grid points
)

price = fd_solver.price(american_put, market)
print(f"American put price: ${price:.2f}")
```

### 6. Volatility Surface Calibration

```python
from options_lib.market_data.vol_surface import calibrate_vol_surface
from options_lib.market_data.option_chain import fetch_option_chain

# Fetch real market data
ticker = "AAPL"
option_chain = fetch_option_chain(ticker)

# Calibrate SVI surface
vol_surface = calibrate_vol_surface(option_chain)

# Get implied volatility at any point
iv = vol_surface.implied_vol(strike=150.0, expiry=0.5)
print(f"IV at strike 150, 6 months: {iv:.1%}")
```

### 7. Local Volatility for Exotic Options

```python
from options_lib.market_data.local_vol import LocalVolSurface
from options_lib.models.local_vol_mc import LocalVolBarrierPricer
from options_lib.instruments.barrier import BarrierOption, BarrierType

# Create local vol surface from calibrated SVI
local_vol = LocalVolSurface.from_vol_surface(vol_surface)

# Barrier option
barrier_call = BarrierOption(
    strike=100.0,
    expiry=1.0,
    option_type=OptionType.CALL,
    barrier=120.0,
    barrier_type=BarrierType.UP_AND_OUT
)

# Monte Carlo with local vol dynamics
barrier_pricer = LocalVolBarrierPricer(
    local_vol_surface=local_vol,
    n_paths=25000,
    n_steps=252
)

price = barrier_pricer.price(barrier_call, market)
print(f"Barrier call price: ${price:.2f}")
```

### 8. Risk Analysis: Greek Surfaces

```python
from options_lib.risk.greeks import GreekSurface
import numpy as np

# Define grid of strikes and expiries
strikes = np.linspace(80, 120, 21)
expiries = np.linspace(0.1, 2.0, 20)

# Compute delta surface
delta_surface = GreekSurface.compute_surface(
    model=model,
    market=market,
    strikes=strikes,
    expiries=expiries,
    greek_type='delta'
)

# Visualize (in dashboard) or analyze
print(f"ATM delta: {delta_surface.get_value(100, 1.0):.3f}")
```

### 9. P&L Attribution Analysis

```python
from options_lib.risk.pnl_attribution import PLAttributionEngine

# Simulate market moves
initial_market = market
final_market = MarketData(
    spot=105.0,      # +5% spot move
    rate=0.05,       # Rate unchanged
    div_yield=0.02   # Dividend unchanged
)

# Time passed
dt = 1/252  # One trading day

# Calculate P&L attribution
attribution = PLAttributionEngine.attribution(
    model=model,
    instrument=call,
    initial_market=initial_market,
    final_market=final_market,
    dt=dt
)

print("P&L Breakdown:")
print(f"Total P&L: ${attribution.total_pnl:.2f}")
print(f"Delta P&L: ${attribution.delta_pnl:.2f}")
print(f"Gamma P&L: ${attribution.gamma_pnl:.2f}")
print(f"Vega P&L: ${attribution.vega_pnl:.2f}")
print(f"Theta P&L: ${attribution.theta_pnl:.2f}")
print(f"Unexplained: ${attribution.unexplained:.2f}")
```

### 10. Interactive Dashboard

Launch the full visualization suite:

```bash
cd OptionsPricingLibrary
streamlit run dashboard/app.py
```

**Dashboard Features**:
- **Live pricing calculator** with real-time model comparison
- **3D volatility surface** with SVI fitting
- **Greek heatmaps** across strike/expiry space
- **Vega matrix** for portfolio sensitivity analysis
- **Model comparison** (Local Vol vs BS, Heston vs BS)
- **P&L backtesting** with historical attribution

## Advanced Usage Patterns

### Model Calibration Workflow

```python
# 1. Fetch market data
option_chain = fetch_option_chain("SPY")

# 2. Calibrate volatility surface
vol_surface = calibrate_vol_surface(option_chain)

# 3. Fit Heston to the surface
heston_params = calibrate_heston_to_surface(vol_surface, market)

# 4. Create calibrated models
heston_model = Heston(heston_params)
local_vol_surface = LocalVolSurface.from_vol_surface(vol_surface)

# 5. Price exotic options consistently
barrier_price_lv = local_vol_pricer.price(barrier_option, market)
barrier_price_hs = heston_pricer.price(barrier_option, market)

# 6. Compare model risk
model_risk = abs(barrier_price_lv - barrier_price_hs)
```

### Portfolio Risk Management

```python
# Define portfolio of options
portfolio = [
    EuropeanOption(95, 0.5, OptionType.CALL),
    EuropeanOption(105, 0.5, OptionType.PUT),
    BarrierOption(100, 1.0, OptionType.CALL, 120, BarrierType.UP_AND_OUT)
]

# Compute portfolio Greeks
portfolio_greeks = {}
for option in portfolio:
    greeks = engine.all_greeks(option, market)
    for greek in greeks:
        portfolio_greeks[greek] = portfolio_greeks.get(greek, 0) + greeks[greek]

# Analyze risk profile
print(f"Portfolio Delta: {portfolio_greeks['delta']:.2f}")
print(f"Portfolio Gamma: {portfolio_greeks['gamma']:.2f}")
print(f"Portfolio Vega: {portfolio_greeks['vega']:.2f}")
```

## Testing & Validation

Run the comprehensive test suite:

```bash
# All tests
pytest options_lib/tests/ -v

# Specific model tests
pytest options_lib/tests/test_black_scholes.py
pytest options_lib/tests/test_heston.py
pytest options_lib/tests/test_monte_carlo.py

# Numerical method tests
pytest options_lib/tests/test_finite_difference.py
pytest options_lib/tests/test_lsmc_and_localvol_mc.py
```

**Test Coverage**:
- Analytical vs numerical consistency
- Monte Carlo convergence
- Greek accuracy via finite differences
- Arbitrage-free conditions
- Edge cases and error handling

## Performance Characteristics

| Method | European | American | Barrier | Asian | Speed | Accuracy |
|--------|----------|----------|---------|-------|-------|----------|
| Black-Scholes | ✓ | ✗ | ✗ | ✗ | Instant | Exact |
| Heston FFT | ✓ | ✗ | ✗ | ✗ | Fast | Semi-analytic |
| Monte Carlo | ✓ | ✓ | ✓ | ✓ | Slow | Statistical |
| Finite Diff | ✓ | ✓ | ✓ | ✗ | Medium | High |
| Local Vol MC | ✓ | ✓ | ✓ | ✓ | Market-consistent | Market-consistent |

## Architecture & Design

### Modular Structure
```
options_lib/
├── instruments/     # Contract definitions
├── models/         # Pricing engines
├── numerics/       # Mathematical methods
├── market_data/    # Vol surfaces, data fetching
├── risk/          # Greeks, P&L attribution
└── tests/         # Validation suite
```

### Key Design Principles
- **Separation of concerns**: Models, instruments, and numerics are independent
- **Extensibility**: Easy to add new models or instruments
- **Performance**: Vectorized NumPy operations throughout
- **Validation**: Comprehensive error checking and arbitrage detection
- **Documentation**: Every method includes mathematical background

## Applications in Finance

### Trading & Risk Management
- **Option pricing**: Consistent valuation across all products
- **Hedge optimization**: Delta-gamma-vega neutral portfolios
- **Volatility trading**: Surface arbitrage and relative value
- **Exotic pricing**: Structured product valuation

### Research & Development
- **Model comparison**: Local vol vs stochastic vol dynamics
- **Calibration studies**: Parameter stability and market fit
- **Risk analytics**: Portfolio-level Greek exposure
- **Algorithm development**: Custom pricing and hedging strategies

### Educational Use
- **Derivatives theory**: Interactive exploration of option behavior
- **Numerical methods**: Understanding PDEs, Monte Carlo, FFT
- **Stochastic calculus**: Volatility modeling and calibration
- **Risk management**: P&L attribution and hedging concepts

## Contributing

We welcome contributions! Areas of interest:
- New pricing models (SABR, rough volatility)
- Additional instruments (lookback, chooser)
- Performance optimizations
- Documentation improvements
- Bug fixes and testing

### Development Setup
```bash
git clone <repository-url>
cd OptionsPricingLibrary
pip install -e ".[dev]"
pre-commit install
```

### Code Standards
- Type hints throughout
- Comprehensive docstrings with mathematical notation
- Unit tests for all new functionality
- Performance benchmarks for numerical methods

## License

MIT License - see LICENSE file for details.

## References

### Core Literature
- **Black-Scholes (1973)**: "The Pricing of Options and Corporate Liabilities"
- **Heston (1993)**: "A Closed-Form Solution for Options with Stochastic Volatility"
- **Dupire (1994)**: "Pricing with a Smile"
- **Gatheral (2004)**: "A Parsimonious Arbitrage-Free Implied Volatility Parameterization"
- **Carr-Madan (1999)**: "Option Valuation Using the Fast Fourier Transform"

### Numerical Methods
- **Crank-Nicolson**: Standard PDE solver for option pricing
- **Longstaff-Schwartz (2001)**: Least squares Monte Carlo for American options
- **FFT Pricing**: Efficient computation of option prices via characteristic functions

This library implements these methods with production-quality code, comprehensive testing, and practical usability for quantitative finance applications.
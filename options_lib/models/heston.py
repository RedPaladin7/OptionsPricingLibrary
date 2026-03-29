import numpy as np 
from scipy.optimize import minimize, Bounds 
from scipy.interpolate import interp1d
from dataclasses import dataclass, field 
from typing import Optional

from options_lib.models.base import Model
from options_lib.models.black_scholes import BlackScholes
from options_lib.models.implied_vol import implied_vol_bs
from options_lib.instruments.base import Instrument, MarketData, OptionType
from options_lib.instruments.european import EuropeanOption
from options_lib.numerics.fft import carr_madan_fft, interpolate_call_price

@dataclass
class HestonParams:
    v0: float
    kappa: float
    v_bar: float
    xi: float
    rho: float

    def __post_init__(self):
        if self.v0 <= 0:
            raise ValueError(f"v0 must be positive, got {self.v0}")
        if self.kappa <= 0:
            raise ValueError(f"kappa must be positive, got {self.kappa}")
        if self.v_bar <= 0:
            raise ValueError(f"v_bar must be positive, got {self.v_bar}")
        if self.xi <= 0:
            raise ValueError(f"xi must be positive, got {self.xi}")
        if not -1 < self.rho < 1:
            raise ValueError(f"rho must be in (-1, 1), got {self.rho}")
        
    @property
    def feller_satisfied(self) -> bool:
        return 2 * self.kappa * self.v_bar > self.xi ** 2
    
    @property
    def initial_vol(self) -> float:
        return np.sqrt(self.v0)
    
    @property
    def long_run_vol(self) -> float:
        return np.sqrt(self.v_bar)
    
    def __repr__(self) -> str:
        feller = "✓" if self.feller_satisfied else "✗"
        return (
            f"HestonParams("
            f"v0={self.v0:.4f} [vol={self.initial_vol:.1%}], "
            f"kappa={self.kappa:.3f}, "
            f"v_bar={self.v_bar:.4f} [vol={self.long_run_vol:.1%}], "
            f"xi={self.xi:.3f}, "
            f"rho={self.rho:.3f}, "
            f"Feller={feller})"
        )
    
@dataclass
class Heston(Model):
    params: HestonParams
    alpha: float = 1.5
    N: int = 4096 
    eta: float = 0.25 

    def charecteristic_function(
            self, 
            u: np.ndarray,
            T: float,
            S: float,
            r: float,
            q: float
    ) -> np.ndarray:
        kappa = self.params.kappa
        v_bar = self.params.v_bar
        xi = self.params.xi
        rho = self.params.rho 
        v0 = self.params.v0 

        d = np.sqrt(
            (kappa - rho * xi * 1j * u) ** 2
            + xi **2 * (u**2 + 1j * u)
        )

        numer_g = kappa - rho * xi * 1j * u - d
        denom_g = kappa - rho * xi * 1j * u + d 
        g = numer_g / denom_g

        exp_neg_dT = np.exp(-d * T)
        B = (numer_g * (1 - exp_neg_dT)
             / (xi ** 2 * (1 - g * exp_neg_dT)))
        
        log_term = np.log((1 - g * exp_neg_dT) / (1 - g))
        A = ((r - q) * 1j * u * T
             + (kappa * v_bar / xi **2)
             * (numer_g * T - 2 * log_term))
        
        phi = np.exp(A + B * v0 + 1j * u * np.log(S))
        return phi 
    
    def price(
            self, 
            instrument: Instrument,
            market: MarketData
    ) -> float:
        if not isinstance(instrument, EuropeanOption):
            raise NotImplementedError(
                'Heston analytical pricer only supports EuropeanOption.'
            )
        
        S = market.spot 
        K = instrument.strike 
        T = instrument.expiry
        r = market.rate 
        q =  market.div_yield

        def char_fn(u):
            return self.charecteristic_function(u, T, S, r, q)
        
        strikes, call_prices = carr_madan_fft(
            char_fn=char_fn,
            S=S, T=T, r=r, q=q,
            alpha=self.alpha,
            N=self.N,
            eta=self.eta
        )

        call_price = interpolate_call_price(strikes, call_prices, K)

        if instrument.option_type == OptionType.PUT:
            put_price = call_price  - S * np.exp(-q * T) + K * np.exp(-r * T) 
            return float(max(put_price, 0.0))
        
        return float(max(call_price, 0.0))
    
    def price_smile(
            self,
            strikes: np.ndarray,
            expiry: float,
            market: MarketData,
            option_type: OptionType = OptionType.CALL
    ) -> np.ndarray:
        S = market.spot 
        T = expiry
        r = market.rate 
        q = market.div_yield

        def char_fn(u):
            return self.charecteristic_function(u, T, S, r, q)
        
        fft_strikes, fft_prices = carr_madan_fft(
            char_fn=char_fn, 
            S=S, T=T, r=r, q=q,
            alpha=self.alpha,
            N=self.N,
            eta=self.eta
        )

        call_prices = np.interp(strikes, fft_strikes, fft_prices)
        call_prices = np.maximum(call_prices, 0.0)

        if option_type == OptionType.PUT:
            put_prices = call_prices - S * np.exp(-q * T) + strikes * np.exp(-r * T)
            return np.maximum(put_prices, 0.0)
        
        return call_prices
    
    def implied_vol_smile(
        self, 
        strikes: np.ndarray,
        expiry: float,
        market: MarketData
    ) -> np.ndarray:
        call_prices = self.price_smile(strikes, expiry, market, OptionType.CALL)
        iv_smile = np.full(len(strikes), np.nan)
        for i, (K, C) in enumerate(zip(strikes, call_prices)):
            try:
                opt = EuropeanOption(strike=K, expiry=expiry, option_type=OptionType.CALL)
                iv_smile[i] = implied_vol_bs(
                    market_price=C,
                    instrument=opt,
                    market=market
                    sigma_init=np.sqrt(self.params.v0)
                )
            except Exception:
                pass 
        return iv_smile
    
    def calibrate(
        self, 
        market_strikes: np.ndarray,
        market_expiries: np.ndarray,
        market_ivs: np.ndarray,
        market_data: MarketData,
        initial_params: Optional[HestonParams] = None,
        verbose: bool = False 
    ) -> 'Heston':
        if initial_params is None:
            mean_iv = float(np.nanmean(market_ivs))
            initial_params = HestonParams(
                v0 = mean_iv ** 2,
                kappa = 2.0,
                v_bar = mean_iv *8 2,
                xi = 0.3, 
                rho = -0.7
            )

        x0 = np.array([
            initial_params.v0,
            initial_params.kappa,
            initial_params.v_bar,
            initial_params.xi,
            initial_params.rho
        ])

        bounds = Bounds(
            lb=[1e-4, 0.1, 1e-4, 0.01, -0.99],
            ub=[1.0, 10.0, 1.0, 2.0, 0.99]
        )

        iteration = [0]

        def objective(x):
            v0, kappa, v_bar, xi, rho = x 
            try:
                params = HestonParams(v0=v0, kappa=kappa, v_bar=v_bar, xi=xi, rho=rho)
                model = Heston(params, alpha=self.alpha, N=self.N, eta=self.eta)
            except ValueError:
                return 1e6
            
            errors = []
            unique_expiries = np.unique(market_expiries)

            for T in unique_expiries:
                mask = market_expiries == T 
                strikes = market_strikes[mask]
                mkt_ivs = market_ivs[mask]

                try:
                    model_ivs = model.implied_vol_smile(strikes, T, market_data)
                    valid = -np.isnan(model_ivs)
                    if valid.any():
                        errors.extend((model_ivs[valid] - mkt_ivs[valid]) ** 2)
                except Exception:
                    errors.append(1.0)
            if not errors:
                return 1e-6 
            
            rmse = np.sqrt(np.mean(errors))

            iteration[0] += 1 
            if verbose and iteration[0] % 20 == 0:
                print(f"  Iter {iteration[0]:4d}: RMSE = {rmse*100:.4f}%  "
                 f"params = v0={v0:.4f}, κ={kappa:.3f}, "
                 f"v̄={v_bar:.4f}, ξ={xi:.3f}, ρ={rho:.3f}")
            return float(rmse)
        
        if verbose:
            print("Starting Heston calibration...")
            print(f"  {len(market_strikes)} market quotes, "
                  f"{len(np.unique(market_expiries))} expiries")
            
        result = minimize(
            objective, 
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8}
        )

        v0, kappa, v_bar, xi, rho = result.x 
        calibrated_params = HestonParams(
            v0=v0, kappa=kappa, v_bar=v_bar, xi=xi, rho=rho
        )

        if verbose:
            print(f"\nCalibration {'converged' if result.success else 'did not converge'}.")
            print(f"  Final RMSE: {result.fun * 100:.4f}%")
            print(f"  {calibrated_params}")

        return Heston(calibrated_params, alpha=self.alpha, N=self.N, eta=self.eta)
    
    def __repr__(self) -> str:
        return f'Heston({self.params})'
        
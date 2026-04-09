import numpy as np 
from dataclasses import dataclass, field
from typing import Optional

from options_lib.models.base import Model
from options_lib.models.black_scholes import BlackScholes
from options_lib.instruments.european import EuropeanOption
from options_lib.instruments.base import Instrument, MarketData, OptionType
from options_lib.models.heston import Heston, HestonParams

@dataclass 
class Greeks:
    price: float 
    delta: float 
    gamma: float 
    theta: float 
    vega: float 
    rho: float = 0.0
    vanna: float = 0.0
    volga: float = 0.0
    charm: float = 0.0

    def to_dict(self) -> dict:
        return {
            'price': self.price,
            'delta': self.delta,
            'gamma': self.gamma,
            'theta': self.theta,
            'vega': self.vega,
            'rho': self.rho,
            'vanna': self.vanna,
            'volga': self.volga,
            'charm': self.charm
        }
    
    def __repr__(self) -> str:
        return (
            f"Greeks(\n"
            f"  price={self.price:.4f}\n"
            f"  delta={self.delta:.4f}  gamma={self.gamma:.4f}\n"
            f"  vega={self.vega:.4f}   theta={self.theta:.4f}/day\n"
            f"  vanna={self.vanna:.4f} volga={self.volga:.4f}\n"
            f"  charm={self.charm:.6f}/day\n"
            f")"
        )
    
class GreekEngine:
    def __init__(self, model: Model):
        self.model = model 

    def price(self, instrument: Instrument, market: MarketData) -> float:
        return self.model.price(instrument, market)
    
    def delta(self, instrument: Instrument, market: MarketData) -> float:
        if isinstance(self.model, BlackScholes):
            return self.model.delta(instrument, market)
        h = market.spot * 0.01
        up = self.model.price(instrument, MarketData(market.spot+h, market.rate, market.div_yield))
        down = self.model.price(instrument, MarketData(market.spot-h, market.rate, market.div_yield))
        return (up - down) / (2*h)
    
    def gamma(self, instrument: Instrument, market: MarketData) -> float:
        if isinstance(self.model, BlackScholes):
            return self.model.gamma(instrument, market)
        h = market.spot * 0.01
        mid = self.model.price(instrument, market)
        up = self.model.price(instrument, MarketData(market.spot+h, market.rate, market.div_yield))
        down = self.model.price(instrument, MarketData(market.spot-h, market.rate, market.div_yield))
        return (up - 2*mid + down) / h**2
    
    def vega(self, instrument: Instrument, market: MarketData) -> float:
        if isinstance(self.model, BlackScholes):
            return self.model.vega(instrument, market)
        from options_lib.models.heston import Heston
        if isinstance(self.model, Heston):
            h = 0.001 
            params = self.model.params

            v0_up = max((np.sqrt(params.v0)+h)**2, 1e-6)
            v0_down = max((np.sqrt(params.v0)-h)**2, 1e-6)
            params_up = HestonParams(v0=v0_up, kappa=params.kappa, v_bar=params.v_bar, xi=params.xi, rho=params.rho)
            params_down = HestonParams(v0=v0_down, kappa=params.kappa, v_bar=params.v_bar, xi=params.xi, rho=params.rho)

            m_up = Heston(params_up, alpha=self.model.alpha, N=self.model.N, eta=self.model.eta)
            m_down = Heston(params_down, alpha=self.model.alpha, N=self.model.N, eta=self.model.eta)

            p_up = m_up.price(instrument, market)
            p_down = m_down.price(instrument, market)
            return (p_up - p_down) / (2*h)
        raise NotImplementedError(f'Vega not implemented for {type(self.model).__name__}')
    
    def theta(self, instrument: Instrument, market: MarketData) -> float:
        if isinstance(self.model, BlackScholes):
            return self.model.theta(instrument, market)
        
        dt = 1 / 365 
        if instrument.expiry <= dt:
            return 0.0 
        
        inst_tomorrow = instrument.with_expiry(instrument.expiry - dt)
        p_today = self.model.price(instrument, market)
        p_tomorrow = self.model.price(inst_tomorrow, market)
        return p_tomorrow - p_today
    
    def rho(self, instrument: Instrument, market: MarketData) -> float:
        if isinstance(self.model, BlackScholes):
            return self.model.rho(instrument, market)
        h = 0.0001
        up = self.model.price(instrument, MarketData(market.spot, market.rate+h, market.div_yield))
        down = self.model.price(instrument, MarketData(market.spot, market.rate-h, market.div_yield))
        return (up - down) / (2*h) / 100
    
    def vanna(self, instrument: Instrument, market: MarketData) -> float:
        if isinstance(self.model, BlackScholes):
            return self.model.vanna(instrument, market)
        if isinstance(self.model, Heston):
            h = 0.01 
            params = self.model.params
            v0_up = max((np.sqrt(params.v0)+h)**2, 1e-6)
            v0_down = max((np.sqrt(params.v0)-h)**2, 1e-6)
            params_up = HestonParams(v0=v0_up, kappa=params.kappa, v_bar=params.v_bar, xi=params.xi, rho=params.rho)
            params_down = HestonParams(v0=v0_down, kappa=params.kappa, v_bar=params.v_bar)
            m_up = Heston(params_up)
            m_down = Heston(params_down)

            eng_up = GreekEngine(m_up)
            eng_down = GreekEngine(m_down)
            delta_up = eng_up.delta(instrument, market)
            delta_down = eng_down.delta(instrument, market)
            return (delta_up - delta_down) / (2*h)
        raise NotImplementedError(f'Vanna not implemented for {type(self.model).__name__}')
    
    def volga(self, instrument: Instrument, market: MarketData) -> float:
        if isinstance(self.model, BlackScholes):
            return self.model.volga(instrument, market)
        if isinstance(self.model, Heston):
            h = 0.01 
            params = self.model.params 

            def make_model(dv):
                v0 = max((np.sqrt(params.v0)+dv)**2, 1e-6)
                return Heston(HestonParams(v0=v0, kappa=params.kappa, v_bar=params.v_bar, xi=params.xi, rho=params.rho))
            p_up = make_model(+h).price(instrument, market)
            p_down = make_model(-h).price(instrument, market)
            p_mid = self.model.price(instrument, market)
            return (p_up - 2*p_mid + p_down) / h**2
        raise NotImplementedError(f'Volga not implemented for {type(self.model).__name__}')
    
    def charm(self, instrument: Instrument, market: MarketData) -> float:
        if isinstance(self.model, BlackScholes):
            return self.model.charm(instrument, market)
        dt = 1 / 365 
        if instrument.expiry <= dt:
            return 0.0
        inst_tomorrow = instrument.with_expiry(instrument.expiry - dt)
        d_today = self.delta(instrument, market)
        d_tomorrow = self.delta(inst_tomorrow, market)
        return d_tomorrow - d_today
    
    def all_greeks(self, instrument: Instrument, market: MarketData) -> Greeks:
        return Greeks(
            price = self.price(instrument, market),
            delta = self.delta(instrument, market),
            gamma = self.gamma(instrument, market),
            vega  = self.vega(instrument, market),
            theta = self.theta(instrument, market),
            rho   = self.rho(instrument, market),
            vanna = self.vanna(instrument, market),
            volga = self.volga(instrument, market),
            charm = self.charm(instrument, market),
        )
    
@dataclass
class GreekSurface: 
    model       : Model
    market      : MarketData
    strikes     : np.ndarray
    expiries    : np.ndarray
    option_type : OptionType
 
    price_surface : np.ndarray = field(init=False, repr=False)
    delta_surface : np.ndarray = field(init=False, repr=False)
    gamma_surface : np.ndarray = field(init=False, repr=False)
    vega_surface  : np.ndarray = field(init=False, repr=False)
    theta_surface : np.ndarray = field(init=False, repr=False)
    vanna_surface : np.ndarray = field(init=False, repr=False)
    volga_surface : np.ndarray = field(init=False, repr=False)
 
    def __post_init__(self):
        n_T = len(self.expiries)
        n_K = len(self.strikes)
        self.price_surface = np.zeros((n_T, n_K))
        self.delta_surface = np.zeros((n_T, n_K))
        self.gamma_surface = np.zeros((n_T, n_K))
        self.vega_surface  = np.zeros((n_T, n_K))
        self.theta_surface = np.zeros((n_T, n_K))
        self.vanna_surface = np.zeros((n_T, n_K))
        self.volga_surface = np.zeros((n_T, n_K))
 
    def compute(self) -> "GreekSurface":
        engine = GreekEngine(self.model)
 
        for i, T in enumerate(self.expiries):
            for j, K in enumerate(self.strikes):
                try:
                    inst = EuropeanOption(strike=K, expiry=T, option_type=self.option_type)
                    g = engine.all_greeks(inst, self.market)
                    self.price_surface[i, j] = g.price
                    self.delta_surface[i, j] = g.delta
                    self.gamma_surface[i, j] = g.gamma
                    self.vega_surface[i, j]  = g.vega
                    self.theta_surface[i, j] = g.theta
                    self.vanna_surface[i, j] = g.vanna
                    self.volga_surface[i, j] = g.volga
                except Exception:
                    for arr in [self.price_surface, self.delta_surface,
                                self.gamma_surface, self.vega_surface,
                                self.theta_surface, self.vanna_surface,
                                self.volga_surface]:
                        arr[i, j] = np.nan
 
        return self
 
    def get_surface(self, greek: str) -> np.ndarray:
        surfaces = {
            'price' : self.price_surface,
            'delta' : self.delta_surface,
            'gamma' : self.gamma_surface,
            'vega'  : self.vega_surface,
            'theta' : self.theta_surface,
            'vanna' : self.vanna_surface,
            'volga' : self.volga_surface,
        }
        if greek not in surfaces:
            raise ValueError(f"Unknown Greek '{greek}'. Choose from {list(surfaces.keys())}")
        return surfaces[greek]
 
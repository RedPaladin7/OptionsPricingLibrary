import numpy as np 
from dataclasses import dataclass
from typing import Optional

from options_lib.models.base import Model 
from options_lib.models.black_scholes import BlackScholes
from options_lib.instruments.base import Instrument, MarketData
from options_lib.instruments.european import EuropeanOption
from options_lib.risk.greeks import GreekEngine, Greeks

@dataclass
class PnLComponents:
    actual_pnl : float 
    delta_pnl: float 
    gamma_pnl: float 
    vega_pnl: float 
    theta_pnl: float 
    rho_pnl: float 
    vanna_pnl: float 
    volga_pnl: float

    ds: float 
    dsigma: float 
    dt: float 

    @property
    def explained_pnl(self) -> float:
        return self.delta_pnl + self.gamma_pnl + self.vega_pnl + self.theta_pnl + self.vanna_pnl + self.volga_pnl
    
    @property
    def unexplained_pnl(self) -> float:
        return self.actual_pnl - self.explained_pnl
    
    @property
    def explanation_ratio(self) -> float:
        if abs(self.actual_pnl) < 1e-8:
            return 1.0 
        return self.explained_pnl / self.actual_pnl
    
    def summary(self) -> str:
        """Pretty-print the P&L breakdown."""
        lines = [
            f"{'='*50}",
            f"P&L Attribution Summary",
            f"{'='*50}",
            f"Market moves:  dS={self.dS:+.4f}  dσ={self.dSigma:+.4f}  dt={self.dt:.4f}yr",
            f"{'─'*50}",
            f"Actual P&L:    {self.actual_pnl:+.4f}",
            f"{'─'*50}",
            f"Delta P&L:     {self.delta_pnl:+.4f}",
            f"Gamma P&L:     {self.gamma_pnl:+.4f}",
            f"Vega P&L:      {self.vega_pnl:+.4f}",
            f"Theta P&L:     {self.theta_pnl:+.4f}",
            f"Vanna P&L:     {self.vanna_pnl:+.4f}",
            f"Volga P&L:     {self.volga_pnl:+.4f}",
            f"{'─'*50}",
            f"Explained:     {self.explained_pnl:+.4f}",
            f"Unexplained:   {self.unexplained:+.4f}",
            f"Expl. ratio:   {self.explanation_ratio:.1%}",
            f"{'='*50}",
        ]
        return '\n'.join(lines)
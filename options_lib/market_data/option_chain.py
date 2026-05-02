import numpy as np
import pandas as pd 
import warnings 
from dataclasses import dataclass, field 
from typing import Optional
from datetime import datetime, date 
import yfinance as yf 

from options_lib.models.black_scholes import BlackScholes
from options_lib.models.implied_vol import implied_vol_bs
from options_lib.instruments.european import EuropeanOption
from options_lib.instruments.base import MarketData, OptionType

@dataclass
class OptionQuote:
    strike: float
    expiry: float 
    expiry_date: str 
    option_type: str 
    mid: float 
    bid: float 
    ask: float 
    iv: float 
    delta: float 
    open_ineterst: int 
    volume: int 

@dataclass
class OptionChain:
    ticker: str 
    spot: float 
    rate: float 
    div_yield: float 
    quotes: list 
    expiry_dates: list 
    forwards: dict = field(default_factory=dict)

    def get_slice(self, expiry_date: str) -> list:
        return [q for q in self.quotes if q.expiry_date == expiry_date]
    
    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([{
            'strike': q.strike,
            'expiry': q.expiry,
            'expiry_date': q.expiry_date,
            'option_type': q.option_type,
            'mid': q.mid,
            'bid': q.bid,
            'ask': q.ask,
            'iv': q.iv,
            'delta': q.delta,
            'open_interest': q.open_ineterst,
            'volume': q.volume
        } for q in self.quotes])
    
    def iv_surface_data(self) -> tuple:
        strikes = np.array([q.strike for q in self.quotes])
        expiries = np.array([q.expiry for q in self.quotes])
        ivs = np.array([q.iv for q in self.quotes])
        expiry_dates = np.array([q.expiry_date for q in self.quotes])
        return strikes, expiries, ivs, expiry_dates
    
    def summary(self) -> str:
        df = self.to_dataframe()
        lines = [
            f"OptionChain: {self.ticker}",
            f"  Spot:     {self.spot:.2f}",
            f"  Rate:     {self.rate:.2%}",
            f"  Quotes:   {len(self.quotes)} ({df['option_type'].value_counts().to_dict()})",
            f"  Expiries: {len(self.expiry_dates)} ({self.expiry_dates[0]} to {self.expiry_dates[-1]})",
            f"  IV range: {df['iv'].min():.1%} – {df['iv'].max():.1%}",
        ]
        return '\n'.join(lines)
    
def fetch_option_chain(
        ticker: str,
        rate: float = 0.05,
        div_yield: float = 0.0, 
        min_expiry_days: int = 7, 
        max_expiry_days: int = 365,
        min_volume: int = 0, 
        min_open_interest: int = 0,
        max_spread_pct: float = 0.50, 
        delta_range: tuple = (0.05, 0.95),
        use_calls: bool = True,
        use_puts: bool = True, 
        n_expiries: Optional[int] = None
) -> OptionChain:
    ticker_obj = yf.Ticker(ticker)

    hist = ticker_obj.history(period='2d')
    if hist.empty:
        raise ValueError(f'No price data for {ticker}')
    spot = float(hist['Close'].iloc[-1])

    all_expiries = ticker_obj.options 
    if not all_expiries:
        raise ValueError('No options data for {ticker}')
    today = date.today()
    valid_expiries = []
    for exp_str in all_expiries:
        exp_date = datetime.strptime(exp_str, '%Y-%m-%d').date()
        days = (exp_date - today).days
        if min_expiry_days <= days <= max_expiry_days:
            valid_expiries.append(exp_str)
    
    if n_expiries:
        valid_expiries = valid_expiries[:n_expiries]
    
    if not valid_expiries:
        raise ValueError('No valid expiries')
    
    all_quotes = []
    forwards = {}

    for exp_str in valid_expiries:
        exp_date = datetime.strptime(exp_str, '%Y-%m-%d').date()
        T = (exp_date - today).days / 365.0 

        try:
            chain = ticker_obj.option_chain(exp_str)
        except Exception as e:
            warnings.warn(f'Skipping {exp_str}: {e}')
            continue 

        for df, opt_type in [
            (chain.calls if use_calls else pd.DataFrame(), 'call'), 
            (chain.puts if use_puts else pd.DataFrame(), 'put')
        ]: 
            if df.empty:
                continue
        quotes = _clean_option_df(df, opt_type, exp_str, T, spot, rate, div_yield, min_volume, min_open_interest, max_spread_pct, delta_range)
        all_quotes.extend(quotes)
        fwd = _implied_forward(chain.calls, chain.puts, spot, T, rate)
        if fwd:
            forwards[exp_str] = fwd 
    if not all_quotes:
        raise ValueError('No valid quotes')
    expiry_dates = sorted({q.expiry_date for q in all_quotes})
    return OptionChain(
        ticker=ticker,
        spot=spot,
        rate=rate,
        div_yield=div_yield,
        quotes=all_quotes,
        expiry_dates=expiry_dates,
        forwards=forwards
    )


def _clean_option_df(df, opt_type, expiry_date, T, spot, rate, div_yield, min_vol, min_oi, max_spread_pct, delta_range) -> list:
    mkt = MarketData(spot, rate, div_yield)
    otype = OptionType.CALL if opt_type == 'call' else OptionType.PUT
    quotes = []

    for _, row in df.iterrows():
        try:
            K = float(row['strike'])
            bid = float(row.get('bid', 0) or 0)
            ask = float(row.get('ask', 0) or 0)
            oi = int(row.get('openInterest', 0) or 0)
            vol = int(row.get('volume', 0) or 0)

            if bid <= 0 or ask <= 0 or ask < bid:
                continue 
            mid = (bid + ask) / 2.0 
            spread_pct = (ask - bid) / mid 

            if spread_pct > max_spread_pct:
                continue 
            if vol < min_vol or oi < min_oi:
                continue

            disc = np.exp(-rate * T)
            fwd = spot * np.exp((rate - div_yield) * T)
            if opt_type == 'call':
                floor = max(fwd * disc - K * disc, 0)
            else:
                floor = max(K * disc - fwd * disc, 0)
            if mid < floor:
                continue 

            inst = EuropeanOption(strike=K, expiry=T, option_type=otype)
            try:
                iv = implied_vol_bs(mid, inst, mkt, sigma_init=0.25)
            except Exception:
                continue

            if not (0.01 < iv < 5.0):
                continue 
            delta = BlackScholes(sigma=iv).delta(inst, mkt)
            abs_delta = abs(delta)
            if not (delta_range[0] < abs_delta < delta_range[1]):
                continue
            quotes.append(OptionQuote(
                strike=K,
                expiry=T,
                expiry_date=expiry_date,
                option_type=opt_type,
                mid=mid,
                bid=bid,
                ask=ask,
                iv=iv,
                delta=delta,
                open_ineterst=oi,
                volume=vol
            ))
        except Exception:
            continue
    return quotes

def _implied_forward(call_dfs, puts_dfs, spot, T, rate) -> Optional[float]:
    try:
        common = sorted(
            set(call_dfs['strike'].values) & set(puts_dfs['strike'].values)
        )    
        if not common:
            return None 
        K_atm = min(common, key=lambda L: abs(k-spot))
        call_row = call_dfs[call_dfs['strike'] == K_atm].iloc[0]
        put_row = puts_dfs[puts_dfs['strike'] == K_atm].iloc[0]

        C_bid = float(call_row.get('bid', 0) or 0)
        C_ask = float(call_row.get('ask', 0) or 0)
        P_bid = float(put_row.get('bid', 0) or 0)
        P_ask = float(put_row.get('ask', 0) or 0)

        if C_bid <= 0 or P_bid <= 0:
            return None 
        C_mid = (C_bid + C_ask) / 2.0
        P_mid = (P_bid + P_ask) / 2.0
        return float(K_atm + np.exp(rate * T) * (C_mid - P_mid))
    except Exception:
        return None
import pytest 
import numpy as np 
from options_lib.models.black_scholes import BlackScholes
from options_lib.models.implied_vol import implied_vol_bs
from options_lib.instruments.european import EuropeanOption
from options_lib.instruments.base import MarketData, OptionType

@pytest.fixture
def atm_call():
    return EuropeanOption(strike=100.0, expiry=1.0, option_type=OptionType.CALL)

@pytest.fixture
def atm_put():
    return EuropeanOption(strike=100.0, expiry=1.0, option_type=OptionType.PUT)
 
@pytest.fixture
def mkt():
    return MarketData(spot=100.0, rate=0.05, div_yield=0.0)
 
@pytest.fixture
def model():
    return BlackScholes(sigma=0.20)

class TestPricing:
    def test_atm_call_known_value(self, atm_call, mkt, model):
        price = model.price(atm_call, mkt)
        assert(abs(price - 10.456) < 0.001, "")

    def test_put_call_parity(self, mkt, model):
        S, K, T, r, q = 100, 100, 1.0, 0.05, 0.0
        call = EuropeanOption(strike=K, expiry=T, option_type=OptionType.CALL)
        put  = EuropeanOption(strike=K, expiry=T, option_type=OptionType.PUT)
        C = model.price(call, mkt)
        P = model.price(put,  mkt)
        lhs = C - P
        rhs = S * np.exp(-q * T) - K * np.exp(-r * T)
        assert abs(lhs - rhs) < 1e-10, f"Put-call parity violated: {lhs:.6f} != {rhs:.6f}"

    def test_instrinsic_value_floor(self, mkt, model):
        deep_itm = EuropeanOption(strike=50.0, expiry=0.1, option_type=OptionType.CALL)

        
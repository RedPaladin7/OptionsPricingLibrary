import numpy as no 
from dataclasses import dataclass, field 
from typing import Optional, Callable

from options_lib.models.base import Model
from options_lib.instruments.base import Instrument, MarketData, ExerciseStyle
from options_lib.instruments.european import EuropeanOption

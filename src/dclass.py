# Set up dataclasses
import datetime
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Scrip:
    symbol: str
    ib_sym: str
    conId: int
    undPrice: float=np.nan
    iv: float=np.nan
    margin: float=np.nan
    bid: float=np.nan
    ask: float=np.nan
    price: float=np.nan
    price_time: datetime.datetime=datetime.datetime.now()
    rsi: float=np.nan

@dataclass
class History:
    symbol: str
    conId: int
    open: float=np.nan
    high: float=np.nan
    low: float=np.nan
    close: float=np.nan
    volume: float=np.nan

@dataclass
class Chains:
    symbol: str
    conId: int
    exchange: str
    expiry: datetime.datetime
    lot: float
    strike: float=np.nan
    iv: float=np.nan
    margin: float=np.nan
    bid: float=np.nan
    ask: float=np.nan
    price: float=np.nan
    price_time: datetime.datetime=datetime.datetime.now()
    




if __name__ == "__main__":
    SCRIP = Scrip()
    print(SCRIP)
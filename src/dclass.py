# Set up dataclasses
import datetime
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Scrip:
    symlot: pd.DataFrame=pd.DataFrame([])
    ohlcs: pd.DataFrame=pd.DataFrame([])
    chains: pd.DataFrame=pd.DataFrame([])
    undPrice: float=np.nan
    iv: float=np.nan
    margin: float=np.nan
    bid: float=np.nan
    ask: float=np.nan
    price: float=np.nan
    price_time: datetime.datetime=datetime.datetime.now()
    rsi: float=np.nan


if __name__ == "__main__":
    SCRIP = Scrip()
    print(SCRIP)
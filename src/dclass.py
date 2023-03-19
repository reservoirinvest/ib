# Set up dataclasses
import datetime
from dataclasses import dataclass

import numpy as np
import pytz

@dataclass
class NSE_Margin:
    """For default margin dataframe"""
    symbol: str=''
    ib_sym: str=''
    conId: int=0
    secType: str=''
    strike: float=np.nan
    right: str=''
    expiry: str=''
    localSymbol: str=''
    timeVal: datetime.datetime=datetime.datetime.now(tz=pytz.timezone('Asia/Kolkata'))
    iv: float=np.nan
    bid: float=np.nan
    ask: float=np.nan
    price: float=np.nan
    margin: float=np.nan
    rsi: float=np.nan
    comm: float=np.nan
    maxCommission: float=np.nan
    initMarginChange: float=np.nan

    def __post_init__(self):
        """Cleans symbol to make it ready for IB"""
        self.ib_sym = self.symbol[:9].replace("&", "")

if __name__ == "__main__":
    margin = NSE_Margin(symbol='Am God & so are You')
    print(margin)
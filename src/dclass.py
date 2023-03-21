# Set up dataclasses
import datetime
from dataclasses import dataclass
from ib_insync import Contract, Order, OrderState

import numpy as np
import pandas as pd
import pytz

@dataclass
class NSE_Margin:
    """For default margin dataframe"""
    symbol: str=''
    ib_sym: str=''
    conId: int=0
    expiry: datetime.date = np.datetime64('nat')
    strike: float=np.nan
    right: str=''
    undPrice: float=np.nan
    localSymbol: str=''
    oi: int=0
    oiChange: float=np.nan
    pChangeOI: float=np.nan
    volume: int=0
    totalBuyQty: int=0
    totalSellQty: int=0
    pChange: float=np.nan
    opt_iv: float=np.nan
    lastPrice: float=np.nan
    change: float=np.nan
    bidQty: int=0
    bid: float=np.nan
    ask: float=np.nan
    askQty: int=0
    lot: float=np.nan
    contract: object=Contract
    order: object=Order
    margincom: object=OrderState
    timeVal: datetime.datetime=datetime.datetime.now(tz=pytz.timezone('Asia/Kolkata'))
    iv: float=np.nan
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
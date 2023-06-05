# support functions common to nse and snp

import asyncio
import datetime
import logging
import math
from pathlib import Path
from typing import Union
import time

import dateutil
import numpy as np
import pandas as pd
import pytz
import ta
from ib_insync import Contract, MarketOrder
from tqdm.asyncio import tqdm

BAR_FORMAT = "{desc:<21}{percentage:3.0f}%|{bar:15}{r_bar}"
# BAR_FORMAT = "{l_bar}{bar:15}{r_bar}{bar:-10b}"


class Timer:
    """Timer providing elapsed time"""
    def __init__(self, name: str = "") -> None:
        self.name = name
        self._start_time = None

    def start(self):
        """Start a new timer"""
        if self._start_time is not None:
            raise Exception(f"Timer is running. Use .stop() to stop it")

        logging.info(f'{self.name} started at {time.strftime("%d-%b-%Y %H:%M:%S", time.localtime())}')

        self._start_time = time.perf_counter()

    def stop(self) -> None:
        if self._start_time is None:
            raise Exception(f"Timer is not running. Use .start() to start it")

        elapsed_time = time.perf_counter() - self._start_time

        logging.info(f"{self.name} took: " +
            f"{time.strftime('%H:%M:%S', time.gmtime(elapsed_time))} seconds")

        self._start_time = None


def get_dte(dt: Union[datetime.datetime, datetime.date, str], 
            exchange: str,
            time_stamp: bool=False) -> float:
    """Get accurate dte.
    Args: dt as datetime.datetime | datetime.date\n 
          exchange as 'nse'|'snp'\n
          time_stamp boolean gives market close in local timestamp\n
    Rets: dte as float | timestamp in local timzeone"""

    if type(dt) is str:
        dt = pd.to_datetime(dt)

    tz_dict = {'nse': ('Asia/Kolkata', 18), 'snp': ('EST', 16)}
    tz, hr = tz_dict[exchange.lower()]

    mkt_tz = pytz.timezone(tz)
    mkt_close_time = datetime.time(hr, 0)

    now = datetime.datetime.now(tz=mkt_tz)

    mkt_close = datetime.datetime.combine(dt.date(), mkt_close_time).astimezone(mkt_tz)

    dte = (mkt_close-now).total_seconds()/24/3600

    if time_stamp:
        dte = mkt_close

    return dte



def pickle_age(data_path: Path) -> dict:
    """Gets age of the pickles in a dict with relativedelta"""

    # Get all the pickles in data path provided
    pickles = Path(data_path).glob('*.pkl')
    d = {f.name: dateutil.relativedelta.relativedelta(datetime.datetime.now(), 
                 datetime.datetime.fromtimestamp(f.stat().st_mtime)) 
                      for f in pickles}

    return d



def get_closest(df: pd.DataFrame, arg1: str = 'price', arg2: str = 'strike', g: str='symbol', depth: int=1) -> pd.DataFrame:
    """gets the closest `depth` elements for differece of two columns `arg1` and `arg2`"""

    df = df.copy(deep=True) # to prevent modifying the source df

    df_out = df.groupby(g, as_index=False) \
      .apply(lambda x: x.iloc[abs(x[arg1]-x[arg2]) \
      .argsort().iloc[0:depth]]) \
      .reset_index(drop=True)
    
    return df_out 



async def qualifyAsync(ib, contracts: list, BLK_SIZE: int=200, port: int=3000) -> list:

    """Asynchronously qualifies IB contracts at 45 secs per 2k contracts"""

    class AsyncIter:
        """Makes iterable object blocks of contract lists"""    
        def __init__(self, items):    
            self.items = items    

        async def __aiter__(self):    
            for item in self.items:    
                yield item    


    results = dict()

    # ..build the raw blocks from cts
    raw_blks = [contracts[i:i + BLK_SIZE] for i in range(0, len(contracts), BLK_SIZE)]

    async for cblk in AsyncIter(pbar:=tqdm(raw_blks, bar_format=BAR_FORMAT)):

        with await ib.connectAsync(port=port):

            desc1 = f"{cblk[0].symbol}{cblk[0].lastTradeDateOrContractMonth}{cblk[0].strike}{cblk[0].right} : "
            desc2 = f"{cblk[-1].symbol}{cblk[-1].lastTradeDateOrContractMonth}{cblk[-1].strike}{cblk[-1].right}"
            desc = desc1 + desc2

            pbar.set_description(f"{desc1[:9]}-{desc2[:9]}")

            results[desc] = await ib.qualifyContractsAsync(*cblk)

    output = [i for k, v in results.items() for i in v if i.conId>0]

    return output



# class AsyncIter:
#     """Makes iterable object blocks of contract lists"""    
#     def __init__(self, items):    
#         self.items = items    

#     async def __aiter__(self):    
#         for item in self.items:    
#             yield item  
             

async def marginsAsync(ib, contracts: list, orders: list, BLK_SIZE: int=45, timeout: float=5, port: int=3000) -> dict:
    """Gets margins and commissions"""

    raw_blks = [(contracts[i:i + BLK_SIZE], orders[i:i + BLK_SIZE]) for i in range(0, len(contracts), BLK_SIZE)]
    dfs = dict()

    async def wifAsync(ct, o):
        wif = ib.whatIfOrderAsync(ct, o)
        try:
            res = await asyncio.wait_for(wif, timeout=timeout)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            res = None
            logging.error(f"Whatif for {ct.localSymbol} timedout with {timeout} or got cancelled")
        return res

    async for cblk in (pbar := tqdm(raw_blks)):

        pbar.bar_format = BAR_FORMAT

        with await ib.connectAsync(port=port):
        
            cts, ords = cblk    

            wif_tasks = [asyncio.create_task(wifAsync(contract, order), name=contract.conId) for contract, order in zip(cts, ords)]
            
            desc = f"{wif_tasks[0].get_name()[:9]}-{wif_tasks[-1].get_name()[:9]}"
            pbar.set_description(desc)

            res = await asyncio.gather(*wif_tasks)
            
            margins = [{'margin': r.initMarginChange, 
                        'commission': r.commission, 
                        'maxCommission': r.maxCommission} 
                        for r in res if r]
            
            conIds = [c.conId for c in cts]

            results = dict(zip(conIds, margins))

            df = pd.DataFrame(results).transpose()\
                   .rename_axis('conId').reset_index()
            
            df = df.apply(pd.to_numeric)

            dfs[desc] = df

            results = dict()
            cts = []
            ords = []

    return dfs



def get_chain_margins(ib, 
                    df_ch: pd.DataFrame, # option chains needing margins
                    only_priced: bool=False, # If False checks all options
                    port: int=3000) -> pd.DataFrame:

    """Get margins for chains provided\n
    Args: df_ch as chains df with symbol, strike, right and expiry\n
          only_prices flag if True checks only for options with lastPrice\n
          expiry could be a string, datetime.date or datetime.datetime\n\n
    Rets: df with `margin` and `comm` for commission"""

    # get options that have some price
    if only_priced:
        try: # lastPrice is only available for NSE
            df_m = df_ch[df_ch.lastPrice>0].reset_index(drop=True)
        except AttributeError:
            logging.info(f"df chain passed does not have `lastPrice` column")
            df_m = df_ch
    else:
        df_m = df_ch

    # convert expiries to string for ib
    if type(df_m.expiry.iloc[0]) is not str:
        df_m.expiry = df_m.expiry.dt.strftime('%Y%m%d')

    # remove dtes < 0
    dte_mask = df_m.expiry.apply(lambda x: get_dte(x, 'nse')) > 0
    df_m = df_m.loc[dte_mask]

    # add ib_sym to chains df_m
    try:
        df_m.insert(1, 'ib_sym', df_m.symbol.apply(nse2ib_symbol_convert))
    except ValueError: # ib_sym already exists
        pass

    # prepare opt_contracts for chains df_m
    opt_contracts = [Contract(symbol=s, lastTradeDateOrContractMonth=e, strike=k, 
                        right=r, secType ='OPT', exchange='NSE', currency='INR') 
    for s, e, k, r
    in zip(df_m.ib_sym, 
            df_m.expiry, 
            df_m.strike, 
            df_m.right, )]

    opt_results = asyncio.run(qualifyAsync(ib, opt_contracts, port=port))
    logging.info('Option contracts qualified')

    # Make dataframe of successfully qualified opt_contracts
    df_opt_cts = pd.DataFrame(opt_results).assign(contract=opt_results)
    df_opt_cts.rename(columns={'lastTradeDateOrContractMonth': 'expiry', 'symbol': 'ib_sym'}, inplace=True)

    # Prepare df_m for the join to qualified df_opts_cts
    idx_cols = ['ib_sym', 'expiry', 'strike', 'right']

    # merge the contract
    df_tgt = df_m.set_index(idx_cols).join(df_opt_cts.set_index(idx_cols), lsuffix='DROP')\
        .filter(regex="^(?!.*DROP)")\
            .dropna(subset=['contract']).reset_index()
    df_tgt = df_tgt.assign(order=[MarketOrder('SELL', lot) for lot in df_tgt.lot]).reset_index(drop=True)

    # get margins for option opt_contracts
    dfs = asyncio.run(marginsAsync(ib, df_tgt.contract, df_tgt.order, port=port))

    # clean outrageous margins and commission
    df_mcom = pd.concat(dfs.values(), ignore_index=True, axis=0)
    df_mcom['comm']= df_mcom[['commission', 'maxCommission']].min(axis=1)
    comm = np.where(df_mcom.comm > 1e7, np.nan, df_mcom.comm)
    margin = np.where(df_mcom.margin > 1e7, np.nan, df_mcom.margin)
    df_mcom = df_mcom.assign(comm=comm, margin=margin).drop(columns=['commission', 'maxCommission'], axis=1)

    # join with a clever use of filter and regex
    df_tgt = df_tgt.set_index('conId').join(df_mcom.set_index('conId'), lsuffix='DROP').filter(regex="^(?!.*DROP)").reset_index()

    return df_tgt



def get_und_margins(ib, df_ch: pd.DataFrame, 
                    df_prices: Union[None, pd.DataFrame] = None, 
                    port: int=3000) -> pd.DataFrame:
    """Generate underlying margins for options closest to underlying prices\n
    Args: df_ch as chains df with latest `undPrice`\n
          df_prices if provided need to have `last` column\n
    """
    # remove negative dtes
    df_ch = df_ch[df_ch.dte > 0]

    # add ib_sym to chains df_ch
    try:
        df_ch.insert(1, 'ib_sym', df_ch.symbol.apply(nse2ib_symbol_convert))
    except ValueError: # ib_sym already exists
        pass

    # Populate latest underlying prices, if provided
    if not None:
        undPrice = df_ch.symbol.map(df_prices.set_index('symbol')['last'].to_dict())
        df_ch = df_ch.assign(undPrice=undPrice)

    # Get the options closest to undPrice
    df_margin = get_closest(df_ch, arg1='undPrice')

    # get the margins
    df_und_margins = get_chain_margins(ib, df_margin, port=port)

    # saves if path is given
    # if savepath:
    #     df_und_margins.to_pickle(savepath)

    return df_und_margins



def find_dir_path_in_cwd(find_dir: str) -> Path:
    """Finds directory in current path and returns its path"""

    find_in = Path.cwd().parts

    try:
        dir_idx = find_in.index(find_dir)
    except ValueError:
        raise Exception(f"{find_dir} not found in {Path.cwd()}. Cannot proceed.")
    found = find_in[:dir_idx+1]
    return Path(*found)


def calcsdmult_df(lastPrice, df):
    """Back calculate standard deviation MULTIPLE against undPrice for given price. Needs dataframes.

    Args:
        (price) as series of strike prices whose sd needs to be known in float.   
        (df) as a dataframe with undPrice, dte and iv columns in float

    Returns:
        Series of std deviation multiple as float

    """
    sdevmult = (lastPrice - df.undPrice) / (
        (df.dte / 365).apply(math.sqrt) * df.iv * df.undPrice)
    return abs(sdevmult)



def get_rsi(df_ohlcs: pd.DataFrame, # df with ascending series of close, indexed on symbols,
            days: int=14, # no of days for the rsi
           ) -> pd.Series:
    '''Gets RSI for no of days specified
    '''

    if df_ohlcs.index.name == 'symbol':
        df_ohlcs.reset_index(inplace=True) # ensures that there is a symbol column
        # df_ohlcs = df_ohlcs.reset_index() 
        

    if len(df_ohlcs.symbol.unique()) > 1:
        df = df_ohlcs.set_index('date').sort_index(ascending=True)
        df = df.groupby('symbol').close.apply(lambda x: ta.momentum.RSIIndicator(close=x, window=days).rsi().iloc[-1])
        rsi = df.rename('rsi')
    else:
        rsi = ta.momentum.RSIIndicator(close=df_ohlcs.close,  window=days).rsi().iloc[-1]
        
    return rsi


def dates_split(start_date: datetime.date, 
                end_date: datetime.date, 
                intv: int=50) -> tuple:    
    """Splits dates into tuples of intervals"""


    if end_date < start_date:
        logging.error(f"End date:{end_date} cannot be earlier than Start date:{start_date} ")
        return None

    blks = ((end_date - start_date)/intv).days

    for _ in range(blks):
        end = start_date + datetime.timedelta(days=intv)
        yield (start_date, end)
        start_date = end + datetime.timedelta(days=1)
    
    # the last chunk
    if start_date < end_date:
        yield (start_date, end_date)


if __name__ == "__main__":

    print(find_dir_path_in_cwd('ib'))
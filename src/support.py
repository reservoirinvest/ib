import asyncio
import datetime
import logging
from pathlib import Path
from typing import Union

import dateutil
import pandas as pd
import pytz
from tqdm.asyncio import tqdm

BAR_FORMAT = "{desc:<10}{percentage:3.0f}%|{bar:25}{r_bar}{bar:-10b}"

def get_dte(dt: Union[datetime.datetime, datetime.date], exchange: str) -> float:
    """Get accurate dte"""

    tz_dict = {'nse': ('Asia/Kolkata', 18), 'snp': ('EST', 16)}
    tz, hr = tz_dict[exchange.lower()]

    mkt_tz = pytz.timezone(tz)
    mkt_close_time = datetime.time(hr, 0)

    now = datetime.datetime.now(tz=mkt_tz)

    mkt_close = datetime.datetime.combine(dt.date(), mkt_close_time).astimezone(mkt_tz)

    dte = (mkt_close-now).total_seconds()/24/3600

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



async def qualifyAsync(ib, contracts: list, BLK_SIZE: int=2002, port: int=3000) -> list:

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

            desc = f"{cblk[0].symbol}{cblk[0].lastTradeDateOrContractMonth}{cblk[0].strike}{cblk[0].right} : "
            desc = desc + f"{cblk[-1].symbol}{cblk[-1].lastTradeDateOrContractMonth}{cblk[-1].strike}{cblk[-1].right}"

            pbar.set_description(desc)

            results[desc] = await ib.qualifyContractsAsync(*cblk)

    output = [i for k, v in results.items() for i in v if i.conId>0]

    return output



class AsyncIter:
    """Makes iterable object blocks of contract lists"""    
    def __init__(self, items):    
        self.items = items    

    async def __aiter__(self):    
        for item in self.items:    
            yield item   

async def marginsAsync(ib, contracts: list, orders: list, BLK_SIZE: int=44, timeout: float=5, port: int=3000) -> dict:
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
            
            desc = f"{wif_tasks[0].get_name()} to {wif_tasks[-1].get_name()}"
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



def find_dir_path_in_cwd(find_dir: str) -> Path:
    """Finds directory in current path and returns its path"""

    find_in = Path.cwd().parts

    try:
        dir_idx = find_in.index(find_dir)
    except ValueError:
        raise Exception(f"{find_dir} not found in {Path.cwd()}. Cannot proceed.")
    found = find_in[:dir_idx+1]
    return Path(*found)

if __name__ == "__main__":
    dt = datetime.datetime(2023, 5, 13) # x is to be converted to mkt close
    exchange = 'nse'

    print(get_dte(dt, exchange))
# Generates data

import asyncio
import logging
import logging.config
from pathlib import Path

import pandas as pd
import yaml
from ib_insync import IB, Contract, MarketOrder
from tqdm import tqdm

from nse.nse import get_nse_chain, get_nse_hist, get_nse_syms, get_prices
from support import (get_closest, get_dte, pickle_age,
                     qualifyAsync, marginsAsync)

root = Path.cwd().parent

# config log
with open(root / 'config' / 'log.yml', 'r') as f:
    log_config = yaml.safe_load(f.read())
    logging.config.dictConfig(log_config)

log = logging.getLogger('ib_log')

# configuration for the program
with open(root / 'config' / 'config.yml', 'r') as f:
    config = yaml.safe_load(f.read())['NSE']

BAR_FORMAT = config['BAR_FORMAT']
days_old = 0.75 # set how many days old
chk_file_path = root / 'data' / 'master'

ib = IB() # initialize

def make_syms(savepath: str="") -> pd.DataFrame:
    """Generates symbols F&O symbols from NSE website"""

    df_syms = get_nse_syms()

    if savepath:
        df_syms.to_pickle(savepath)

    return df_syms



def make_chains(df_syms: pd.DataFrame, savepath: str='') -> pd.DataFrame:
    """Generates option chains for all NSE F&Os with live prices"""

    dfs = []

    tq_scrips = tqdm(df_syms.symbol, bar_format=BAR_FORMAT)
    for s in tq_scrips:
        tq_scrips.set_description(f"{s}")
        try:
            dfs.append(get_nse_chain(s))
        except KeyError as e:
            log.error(f"{s} has error {e}")

    # assemble the chains
    nse_chains = pd.concat(dfs,ignore_index=True)

    # get the dtes for the chains

    exchange = df_syms.exchange.iloc[0]

    dte = nse_chains.expiry.apply(lambda x: get_dte(x, exchange))
    dte.name = 'dte'

    # insert the dtes
    try:
        nse_chains.insert(5, 'dte', dte)
    except ValueError:
        log.warning('dte already in nse_chains, will be refreshed')
        nse_chains = nse_chains.assign(dte=dte)

    if savepath:
        nse_chains.to_pickle(savepath)

    return nse_chains


def get_und_margins(df_syms: pd.DataFrame, df_ch: pd.DataFrame, savepath: str='') -> pd.DataFrame:
    """Generate underlying margins"""

    # dictionary map
    m_di = df_syms.set_index('symbol').ib_sym.to_dict()


if __name__ == "__main__":
    
    df_syms = get_nse_syms()

    df_chains = make_chains(pd.DataFrame(df_syms.iloc[0]))

    print(df_chains)
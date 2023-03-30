# Generates data

import asyncio
import logging
import logging.config
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from ib_insync import IB, Contract, MarketOrder
from tqdm import tqdm

from nse import get_nse_chain, get_nse_hist, get_nse_syms, get_prices
from support import (find_dir_path_in_cwd, get_chain_margins, get_closest,
                     get_dte, get_und_margins, marginsAsync, Timer,
                     nse2ib_symbol_convert, qualifyAsync)

root = Path.cwd().parent

# config log
with open(root / 'config' / 'log.yml', 'r') as f:
    log_config = yaml.safe_load(f.read())
    logging.config.dictConfig(log_config)

log = logging.getLogger('ib_log')
logging.getLogger('ib_insync').setLevel(logging.WARNING) # Disable debug logs

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



def make_history(df_syms: pd.DataFrame, savepath: str= '', hist_days: int=365):
    """Generates history"""

    # convert to scrips with is_index = True for Indexes like NIFTY
    scrips = df_syms.set_index('symbol')['secType'].eq('IND').to_dict()

    dfs = []

    tq_scrips = tqdm(scrips.items(), bar_format=BAR_FORMAT)
    for k, v in tq_scrips:
        tq_scrips.set_description(f"{k}")
        try:
            dfs.append(get_nse_hist(k, v, hist_days))
        except KeyError as e:
            log.error(f"{k} has error {e}")

    # assemble the hists
    nse_hists = pd.concat(dfs,ignore_index=True)        

    if savepath:
        nse_hists.to_pickle(savepath)

    return nse_hists



if __name__ == "__main__":

    g_time = Timer("generate.py")
    g_time.start()

    root = find_dir_path_in_cwd('ib')
    datapath = root / 'data' / 'master'

    port = 3000 # 3000 for live, 4002 for paper
    
    df_syms = get_nse_syms()
    logging.info('Symbols generated')

    ## Chains....
    ## ***********

    # df_chains = make_chains(pd.DataFrame(df_syms), 
    #                         savepath=datapath / 'nse_chains.pkl')
    # logging.info('Chains made') 

    df_chains = pd.read_pickle(datapath / 'nse_chains.pkl')

    ## History...
    ## ***********

    # df_hist = make_history(df_syms, savepath=datapath / 'nse_hists.pkl')
    # logging.info('Histores made')

    df_hist = pd.read_pickle(datapath / 'nse_hists.pkl')
    
    ## Underling margins...
    ## ***********
    
    df_prices = get_prices()
    df_unds = get_und_margins(ib, df_chains, 
                    df_prices, 
                    port=port)
    df_unds.to_pickle(datapath / 'nse_margins.pkl')

    logging.info('Underlying margins extracted')
    
    df_unds = pd.read_pickle(datapath / 'nse_margins.pkl')

    ## All options with margins...
    ## *************

    df_opt_margins = get_chain_margins(ib, df_ch=df_chains, 
                                       only_priced=True, port=port)
    
    df_opt_margins.to_pickle(datapath / 'nse_opt_margins.pkl')
    logging.info('All NSE Options with margins generated')

    g_time.stop()
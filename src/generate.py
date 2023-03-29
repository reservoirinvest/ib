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
from support import (find_dir_path_in_cwd, get_closest, get_dte, marginsAsync,
                     qualifyAsync)

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



def get_und_margins(ib, df_syms: pd.DataFrame, df_ch: pd.DataFrame, savepath: str='') -> pd.DataFrame:
    """Generate underlying margins"""

    # dictionary map
    m_di = df_syms.set_index('symbol').ib_sym.to_dict()

    # Get the latest underlying prices
    df_prices = get_prices()
    undPrice = df_ch.symbol.map(df_prices.set_index('symbol')['last'].to_dict())
    df_ch = df_ch.assign(undPrice=undPrice)

    # Get the options closest to undPrice
    df_margin = get_closest(df_ch, arg1='undPrice')

    # add ib_sym to df_margin
    df_margin.insert(1, 'ib_sym', df_margin.symbol.map(m_di))

    # qualify and build underlying contracts and orders
    contracts = [Contract(symbol=s, lastTradeDateOrContractMonth=e, strike=k, 
                        right=r, secType ='OPT', exchange='NSE', currency='INR') 
    for s, e, k, r
    in zip(df_margin.ib_sym, 
            df_margin.expiry.dt.strftime('%Y%m%d'), 
            df_margin.strike, 
            df_margin.right, )]

    # prepare orders
    orders = [MarketOrder("SELL", lot) for lot in df_margin.lot]

    # qualify contracts
    contracts = asyncio.run(qualifyAsync(ib, contracts))

    # remove unqualified contracts from df_margin
    q_cons = {c.symbol: c for c in contracts if c.conId}
    df_margin = df_margin.assign(contract=df_margin.ib_sym.map(q_cons), order=orders)\
                        .dropna(subset=['contract']).reset_index(drop=True)

    # Get the underlying margins

    results = asyncio.run(marginsAsync(ib, df_margin.contract, df_margin.order))
    conId = [c.conId for c in df_margin.contract]
    df_margin.insert(0, 'conId', conId)

    df_undcom = pd.concat(results, ignore_index=True)
    df_margin = df_margin.set_index('conId').join(df_undcom.set_index('conId')).drop(columns='maxCommission').reset_index()

    if savepath:
        df_margin.to_pickle(savepath)

    return df_margin


def get_opt_margins(ib, df_syms: pd.DataFrame, df_ch: pd.DataFrame, df_margin: pd.DataFrame, savepath: str='') -> pd.DataFrame:

    """Get option margins for all options in the chains that have underlying margins"""

    # dictionary map
    m_di = df_syms.set_index('symbol').ib_sym.to_dict()

    # remove unnecessary unds from chains
    df_ch = df_ch[df_ch.symbol.isin(df_margin.symbol)]

    # get options that have some price
    df_m = df_ch[df_ch.lastPrice>0].reset_index(drop=True)

    # add ib_sym to chains df_m
    df_m.insert(1, 'ib_sym', df_m.symbol.map(m_di))

    # prepare opt_contracts for chains df_m
    opt_contracts = [Contract(symbol=s, lastTradeDateOrContractMonth=e, strike=k, 
                        right=r, secType ='OPT', exchange='NSE', currency='INR') 
    for s, e, k, r
    in zip(df_m.ib_sym, 
            df_m.expiry.dt.strftime('%Y%m%d'), 
            df_m.strike, 
            df_m.right, )]

    opt_results = asyncio.run(qualifyAsync(ib, opt_contracts))

    # Make dataframe of successfully qualified opt_contracts
    df_opt_cts = pd.DataFrame(opt_results).assign(contract=opt_results)
    df_opt_cts.rename(columns={'lastTradeDateOrContractMonth': 'expiry', 'symbol': 'ib_sym'}, inplace=True)

    # Prepare df_m for the join to qualified df_opts_cts
    df_tgt = df_m.assign(expiry = df_m.expiry.dt.strftime('%Y%m%d'))

    idx_cols = ['ib_sym', 'expiry', 'strike', 'right']

    df_tgt = df_tgt.set_index(idx_cols).join(df_opt_cts.set_index(idx_cols)[['conId', 'contract']]).dropna(subset=['contract']).reset_index()
    df_tgt = df_tgt.assign(order=[MarketOrder('SELL', lot) for lot in df_tgt.lot])

    # get margins for option opt_contracts
    dfs = asyncio.run(marginsAsync(ib, df_tgt.contract, df_tgt.order, BLK_SIZE=44, timeout=5))
    df_mcom = pd.concat(dfs.values(), ignore_index=True, axis=0)

    df_tgt = df_tgt.set_index('conId').join(df_mcom.set_index('conId')).drop(columns='maxCommission').reset_index()

    if savepath:
        df_tgt.to_pickle(savepath)

    return df_tgt

if __name__ == "__main__":

    root = find_dir_path_in_cwd('ib')

    # datapath = Path(r'C:\Users\kashi\python\ib\data\master')
    datapath = root / 'data' / 'master'
    
    df_syms = get_nse_syms()
    logging.info('Symbols generated')

    df_chains = make_chains(pd.DataFrame(df_syms), 
                            savepath=datapath / 'nse_chains.pkl')
    logging.info('Chains made')

    # df_chains = pd.read_pickle(datapath / 'nse_chains.pkl')

    df_hist = make_history(df_syms, savepath=datapath / 'nse_hists.pkl')
    logging.info('Histores made')

    # df_hist = pd.read_pickle(datapath / 'nse_hists.pkl')
    
    df_unds = get_und_margins(ib, df_syms, df_chains, 
                              savepath=datapath / 'nse_margins.pkl')
    logging.info('Underlying margins extracted')
    
    df_opt_margins = get_opt_margins(ib, df_syms=df_syms, df_ch=df_chains, df_margin=df_unds, savepath=datapath / 'nse_opt_margins.pkl')
    logging.info('All NSE Options with margins generated')

    print(df_opt_margins)
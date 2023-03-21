# Generates main data

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
BAR_FORMAT = "{desc:<10}{percentage:3.0f}%|{bar:25}{r_bar}{bar:-10b}"

ib = IB() # initialize

# config log
with open(root / 'config' / 'log.yml', 'r') as f:
    log_config = yaml.safe_load(f.read())
    logging.config.dictConfig(log_config)

log = logging.getLogger('ib_log')

# configuration for the program
with open(root / 'config' / 'config.yml', 'r') as f:
    config = yaml.safe_load(f.read())['NSE']

days_old = 0.75 # set how many days old
chk_file_path = root / 'data' / 'master'

# GENERATE NSE SYMBOLS

df_syms = get_nse_syms()

log.info('Finished getting df_syms')

# BUILD OPTION CHAINS

try:
    chain_age_delta = pickle_age(chk_file_path)['nse_chains.pkl']
    chain_age = chain_age_delta.days + chain_age_delta.hours/24 + chain_age_delta.minutes/24/60
except KeyError:
    chain_age = days_old + 1 # force regeneration

old_chains = chain_age > days_old
dfs = []

if old_chains:
    tq_scrips = tqdm(df_syms.symbol, bar_format=BAR_FORMAT)
    for s in tq_scrips:
        tq_scrips.set_description(f"{s}")
        try:
            dfs.append(get_nse_chain(s))
        except KeyError as e:
            log.error(f"{s} has error {e}")
    
    log.info('Option Chains regenerated')

else:
    dfs = [pd.read_pickle(chk_file_path / 'nse_chains.pkl')]

# assemble the chains
nse_chains = pd.concat(dfs,ignore_index=True)

# store the nse_chains if new
if old_chains:
    data_path = root / 'data' / 'master' / 'nse_chains.pkl'
    nse_chains.to_pickle(data_path)

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

# GENERATE HISTORY

# convert to scrips with is_index = True for Indexes like NIFTY
scrips = df_syms.set_index('symbol')['secType'].eq('IND').to_dict()

try:
    hist_age_delta = pickle_age(chk_file_path)['nse_hists.pkl']
    hist_age = hist_age_delta.days + hist_age_delta.hours/24 + hist_age_delta.minutes/24/60
except KeyError:
    hist_age = days_old + 1 # regenerate

old_hists = hist_age > days_old

dfs = []
hist_days = 365

if old_hists:
    tq_scrips = tqdm(scrips.items(), bar_format=BAR_FORMAT)
    for k, v in tq_scrips:
        tq_scrips.set_description(f"{k}")
        try:
            dfs.append(get_nse_hist(k, v, hist_days))
        except KeyError as e:
            log.error(f"{k} has error {e}")
            pass
        
else:
    dfs = [pd.read_pickle(chk_file_path / 'nse_hists.pkl')]

# assemble the hists
nse_hists = pd.concat(dfs,ignore_index=True)

# store the nse_hists if new
if old_hists:
    data_path = root / 'data' / 'master' / 'nse_hists.pkl'
    nse_hists.to_pickle(data_path)
    log.info('Histories regenerated for underlyings')

# GENERATE MARGINS FOR UNDERLYINGS

try:
    margin_age_delta = pickle_age(chk_file_path)['nse_margins.pkl']
    margin_age = margin_age_delta.days + margin_age_delta.hours/24 + margin_age_delta.minutes/24/60
except KeyError:
    margin_age = days_old + 1 # force regeneration

old_margins = margin_age > days_old

# dictionary map
m_di = df_syms.set_index('symbol').ib_sym.to_dict()

if old_margins:
    # prepare chains
    df_ch = nse_chains

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
    # cos = [(c, o) for c, o in zip(df_margin.contract, df_margin.order)]

    results = asyncio.run(marginsAsync(ib, df_margin.contract, df_margin.order))

    # clean up those results which are empty
    cdi = {c.conId: c.symbol for c in df_margin.contract}
    clean_results = {cdi[k]: v for k, v in results.items() if v}
    df_margin = df_margin.assign(margincom=df_margin.ib_sym.map(clean_results))\
                        .dropna(subset=['margincom'])\
                        .reset_index(drop=True)

    # save underlying margins
    df_margin.to_pickle(root / 'data' / 'master' / 'nse_margins.pkl')

    log.info('Generated margins for underlyings')

else:
    df_margin = pd.read_pickle(root / 'data' / 'master' / 'nse_margins.pkl')

# GENERATE MARGINS FOR ALL OPTIONS

df_ch = nse_chains

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

# save option margins
df_tgt.to_pickle(root / 'data' / 'master' / 'nse_opt_margins.pkl')

log.info('Generated margins for all options')


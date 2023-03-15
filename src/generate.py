# Generates main data

import logging
import logging.config
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

from nse.nse import get_nse_syms
from nse.nse import get_nse_chain, get_nse_hist
from support import get_dte, pickle_age

root = Path.cwd()
bar_format = "{desc:<10}{percentage:3.0f}%|{bar:25}{r_bar}{bar:-10b}"

with open(root / 'config' / 'log.yml', 'r') as f:
    config = yaml.safe_load(f.read())
    logging.config.dictConfig(config)

log = logging.getLogger('ib_log')

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
    tq_scrips = tqdm(df_syms.symbol, bar_format=bar_format)
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
    tq_scrips = tqdm(scrips.items(), bar_format=bar_format)
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




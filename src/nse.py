# contains all functions specific to nse

# !!! to-do
# [] remove get_history dependancy. Make your own!
# [] compute targets from standard deviations.
# [] compute margins
# [] set prices
# [] check positions
# [] place trades

import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
from nsepy import get_history
from tqdm import tqdm
import yaml
from typing import Union

from support import get_dte

log = logging.getLogger('log')

# prevent urllib DEBUG connectionpool logs from nsepython requests
logging.getLogger('urllib3').setLevel(logging.WARNING)

# Get constants from YAML file
with open('refmap.yml', 'rb') as ref:
    def_dict = yaml.safe_load(ref)

BAR_FORMAT = def_dict['BAR_FORMAT']

INDICES = def_dict['NSE']['INDICES']

HEADERS = def_dict['NSE']['HEADER']
INDEX_HEADERS = def_dict['NSE']['INDEX_HIST_HEADER']

INDEX_COLS = def_dict['NSE']['INDEX_HIST']
INDEX_SYM_MAP = def_dict['NSE']['INDEX_SYM_MAP']

EQUITY_HIST = def_dict['NSE']['EQUITY_HIST']
OPT_HIST = def_dict['NSE']['OPT_HIST']

def nsefetch(payload):
    try:
        output = requests.get(payload,headers=HEADERS).json()
    except ValueError:
        s = requests.Session()
        output = s.get("http://nseindia.com",headers=HEADERS)
        output = s.get(payload,headers=HEADERS).json()
    return output


def nsesymbolpurify(symbol):
    symbol = symbol.replace('&','%26') #URL Parse for Stocks Like M&M Finance
    return symbol

def nse_optionchain_scrapper(symbol):
    symbol = nsesymbolpurify(symbol)
    if any(x in symbol for x in INDICES):
        payload = nsefetch('https://www.nseindia.com/api/option-chain-indices?symbol='+symbol)
    else:
        payload = nsefetch('https://www.nseindia.com/api/option-chain-equities?symbol='+symbol)
    return payload


def nse2ib_symbol_convert(s: str) -> str:
    """Convert NSE symbols to IB compatible symbols"""
    
    res = s[:9].replace("&", "")
    if res == 'NIFTY':
        res = 'NIFTY50'

    return res


def nse_json(url: str, 
             data: str = '',
             HEADERS: str = HEADERS):
    
    """
    Fetch json from nse for the url provided.
    
    For index, data has to be provided as a dictionary wrapped in string.
    """

    if data: # only index has data for request.post

        result = requests.post(url=url, headers=INDEX_HEADERS,  data=str(data)).json()
        
        # needs eval to pull dictionary out of string
        x = eval(list(result.values())[0])

        output = x

    else:

        try:
            output = requests.get(url, headers=HEADERS, data=data).json()

        except ValueError:
            s=requests.Session()
            output = s.get("http://nseindia.com",headers=HEADERS)
            output = s.get(url,headers=HEADERS, data=data)
            try:
                output = output.json()
            except ValueError:
                output = None

    return output



def get_nse_syms() -> pd.DataFrame:
    """Generates symbols for nse with expiry months having lots"""

    # get symbols url
    url = "https://www.nseindia.com/api"
    url = url + "/equity-stockIndices?index=SECURITIES%20IN%20F%26O"

    # get the json for stocks
    njs = nse_json(url, HEADERS=HEADERS)
    equities = [njs['data'][x]['symbol'] for x in range(len(njs['data']))]
    nselist = INDICES + equities

    df_syms = pd.DataFrame({'nse_symbol': nselist})

    # introduce `secType`
    df_syms.insert(1, 'secType', 
                    np.where(df_syms.nse_symbol.str.contains('NIFTY'), 'IND', 'STK'))

    # introduce `exchange`
    df_syms.insert(2, 'exchange', 'NSE')

    # make ib friendly symbols
    df_syms.insert(1, 'ib_symbol', df_syms.nse_symbol.apply(nse2ib_symbol_convert))

    # df = df_syms.rename(columns={'symbol':'nse_symbol'})

    return df_syms


def nse_get_fno_lot_sizes(
        symbol: str = '', 
        dt: str = pd.to_datetime('today')):
    """
    gets fno lot size for nse

    :param symbol: <str>. Needs NSE friendly symbol
    :param dt: <str>. Date of expiry - e.g. 'Jan-2023'.
               Converts to pd period M.   
               Gives all expiries if left blank.

    :returns:  

       if symbol, dt is provided - returns lotsize in int 
       else returns pd dataframe

    """

    url="https://archives.nseindia.com/content/fo/fo_mktlots.csv"

    if dt:
        try:
            dt = pd.to_datetime(dt).to_period('M')
        except Exception:
            log.error(f"cannot figure out what dt = '{dt}' means for {symbol}")
            dt = ''

    payload = pd.read_csv(url)

    lots_df = payload[list(payload)[1:5]]

    # strip whitespace from columns and make it lower case
    lots_df.columns = lots_df.columns.str.strip().str.lower() 

    # strip all string contents of whitespaces
    lots_df = lots_df.applymap(lambda x: x.strip() if type(x) is str else x)

    # remove 'Symbol' row
    lots_df = lots_df[lots_df.symbol != 'Symbol']

    # melt the expiries into rows
    lots_df = lots_df.melt(id_vars=['symbol'], var_name='expiryM', value_name='lot').dropna()

    # remove rows without lots
    lots_df = lots_df[~(lots_df.lot == '')]

    # convert expiry to period
    lots_df = lots_df.assign(expiryM=pd.to_datetime(lots_df.expiryM, format='%b-%y').dt.to_period('M'))

    # convert lots to integers
    lots_df = lots_df.assign(lot=pd.to_numeric(lots_df.lot, errors='coerce'))
    
    # convert & to %26 #!!! is this needed?
    # lots_df = lots_df.assign(symbol=lots_df.symbol.str.replace('&', '%26'))
    
    output = lots_df.reset_index(drop=True)

    if symbol:
        output = output.loc[output.symbol == symbol]
        if dt:
            output = output.loc[output.expiryM == dt]
        
        if output.empty:
            output = np.nan
        elif len(output) == 1:
            output = output.lot.iloc[0]

    return output


def get_nse_chain(symbol: str) -> pd.DataFrame:
        """Get Option Chains for a symbol"""

        scraped = nse_optionchain_scrapper(symbol)

        raw_dict = scraped['records']['data']
        dfs = [pd.DataFrame.from_dict(raw_dict[i]).transpose()[2:] 
                for i in range(len(raw_dict))]

        df = pd.concat(dfs).drop(columns='identifier').rename_axis('right').reset_index()

        float_cols = ['strikePrice', 'pchangeinOpenInterest', 'impliedVolatility', 
                'lastPrice', 'change', 'pChange', 'bidprice', 'askPrice' , 
                'underlyingValue' ]
        int_cols = ['openInterest', 'changeinOpenInterest', 'totalTradedVolume', 
                'totalBuyQuantity', 'totalSellQuantity', 'bidQty', 'askQty']
        df[float_cols] = df[float_cols].astype('float32')
        df[int_cols] = df[int_cols].astype('int64')
        df['expiryDate'] = pd.to_datetime(df.expiryDate)

        colmap = {'underlying': 'symbol', 'expiryDate': 'expiry', 'strikePrice': 'strike',  
                'right': 'right','underlyingValue': 'undPrice', 'openInterest': 'oi',
                'changeinOpenInterest': 'oiChange', 'pchangeinOpenInterest': 'pChangeOI',
                'totalTradedVolume': 'volume', 'totalBuyQuantity': 'totalBuyQty', 
                'totalSellQuantity': 'totalSellQty', 'pChange': 'pChange', 
                'impliedVolatility': 'opt_iv',   'lastPrice': 'lastPrice', 'change': 'change',
                'bidQty': 'bidQty', 'bidprice': 'bid', 'askPrice': 'ask', 'askQty': 'askQty',
                }

        df = df.rename(columns=colmap)[colmap.values()]

        # map the lots

        df_lots = nse_get_fno_lot_sizes()
        df['expiryM'] = pd.to_datetime(df.expiry, format='%b-%y').dt.to_period('M')

        map_cols = ['symbol', 'expiryM']

        df = df.set_index(map_cols) \
        .join(df_lots.set_index(map_cols)) \
        .reset_index() \
        .drop('expiryM', axis=1)

        df = df.assign(opt_iv = df.opt_iv/100)
        df = df.assign(right=df.right.str[:1])
        df = df.sort_values(['expiry', 'right', 'strike'], 
                        ascending=[True, False, True]).reset_index(drop=True)
        
        df = df.rename(columns={'symbol': 'nse_symbol'})

        return df


def make_chains(symbols: Union[pd.Series, list], savepath: str = '') -> pd.DataFrame:
    """Generates option chains for all NSE F&Os with live prices"""

    dfs = []
    tq_scripts = tqdm(symbols, bar_format = BAR_FORMAT)

    for s in tq_scripts:
        tq_scripts.set_description(f"{s}")
        try:
            dfs.append(get_nse_chain(s))
        except KeyError as e:
            log.error(f"{s} has error {e}")

    # assemble the chains
    nse_chains = pd.concat(dfs, ignore_index=True)

    # get the dtes for the chains
    exchange = 'NSE'
    dte = nse_chains.expiry.apply(lambda x: get_dte(x, exchange))

    # insert the dtes
    try:
        nse_chains.insert(5, 'dte', dte)
    except ValueError:
        log.warning('dte already in nse_chains, will be refreshed')
        nse_chains = nse_chains.assign(dte=dte)

    nse_chains = nse_chains.rename(columns={'symbol':'nse_symbol'})

    if savepath:
        nse_chains.to_pickle(savepath)

    return nse_chains


def clean_prices(df: pd.DataFrame) -> pd.DataFrame:
    "Cleans up NSE prices to give numeric values and times"

    # stage a `mother` numeric df, removing commas from price text
    df = df.apply(lambda x: 
                    pd.to_numeric(x.astype(str).str.replace(',',''), 
                                errors='ignore'))

    df = df.dropna(axis=1)

    # remove unnecessary columns if present
    searchfor = '|'.join(['meta', 'chart', 'indexOrder', 'timeVal', 
                        'percChange', 'identifier'])
    col_loc = ~df.columns.str.contains(searchfor)
    df = df.loc[:, col_loc]

    # protect the texts
    protect_cols = ['symbol', 'series']

    df_clean = df.drop(protect_cols, axis=1, errors="ignore")

    # Make the datetimes
    date_cols = [s for s in df_clean.columns if 'date' in s]
    df_dates = df[date_cols].apply(pd.to_datetime, axis=1, errors='coerce')

    # Make the numerics
    df_nums = df[df_clean.columns.difference(df_dates.columns)]\
                        .apply(pd.to_numeric, errors='coerce', axis=1)\
                            .dropna(axis=1)

    # Join outputs
    df_out = df.loc[:, df.columns.difference(df_clean.columns)].join(df_dates).join(df_nums)

    df_out = df_out.rename(columns={'symbol': 'nse_symbol'})

    return df_out


def equity_prices() -> pd.DataFrame:
    """Gets all live equity prices"""

    url1 = "https://www.nseindia.com/api"
    url1 = url1 + "/equity-stockIndices?index=SECURITIES%20IN%20F%26O"

    # this url does not work outside market hours. #!!! check if this is faster
    url2 = "https://www1.nseindia.com/live_market/dynaContent/live_watch/stock_watch/foSecStockWatch.json"


    for u in [url1, url2]:

        js = nse_json(u)

        if js:
            break

    # make df of equity_prices
    x = []

    for item in js['data']:
        df = pd.DataFrame.from_dict([item])
        x.append(df)
    df = pd.concat(x, ignore_index=True)

    df = clean_prices(df) # handle obj -> numerics, times

    return df



def index_prices(important: bool = True) -> pd.DataFrame:
    """Gets all live index prices 
    (equity index only. No bonds, etc)"""

    url = 'https://iislliveblob.niftyindices.com/jsonfiles/LiveIndicesWatch.json'
    js = nse_json(url) # works!

    x = []
    for item in js['data']:
        df = pd.DataFrame.from_dict([item])
        x.append(df)
    df = pd.concat(x, ignore_index=True)

    # keep only equities
    df = df[(df.indexType == 'eq') & (df.yearHigh != '-')].reset_index(drop=True)

    # symbol map for fno index (except `INDIAVIX`)
    df.insert(0, 'symbol', df.indexName.map(INDEX_SYM_MAP).fillna(df.indexName))

    if important:
        df = df[df.symbol.isin(INDEX_SYM_MAP.values())]

    df = clean_prices(df) # handle obj -> numerics, times

    return df



def get_prices() -> pd.DataFrame:
    """Get the underlying prices"""

    df_eq = equity_prices()
    df_ix = index_prices()

    # Do column conversions to standardize
    with open('refmap.yml', 'rb') as ref:
        def_dict = yaml.safe_load(ref)
        
    eq_cols_dict = def_dict['NSE']['EQUITY']
    ix_cols_dict = def_dict['NSE']['INDEX']

    df_e = df_eq.rename(columns=eq_cols_dict).loc[:, eq_cols_dict.values()]
    df_i = df_ix.rename(columns=ix_cols_dict).loc[:, ix_cols_dict.values()]

    df = pd.concat([df_e, df_i], ignore_index=True)
    df['localTime'] = pd.Timestamp.now(tz='Asia/Calcutta')

    return df



def clean_index_hist(x: dict, index_columns: str=INDEX_COLS) -> pd.DataFrame:
        
        """Cleans index json history and makes it df"""

        df = pd.DataFrame.from_records(x)

        # clean the column names
        df1 = df.iloc[:, 1:].rename(columns=index_columns)

        # clean the column types
        df2 = df1.iloc[:, 1].apply(pd.to_datetime)
        df2 = (df2 + pd.Timedelta(hours=16)).dt.tz_localize('Asia/Calcutta')

        df3 = df1.iloc[:, 2:].apply(pd.to_numeric)
        df4 = pd.concat([df1.iloc[:, 0], df2, df3], axis=1)

        # symbol map for fnos (except `INDIAVIX`)
        df4.insert(0, 'nse_symbol', df4.symbol.map(INDEX_SYM_MAP).fillna(df4.symbol))

        return df4



def clean_eq_hist(eq_json: dict, cols = EQUITY_HIST) -> pd.DataFrame:

    """Cleans equity json history and makes it df"""

    x = list(eq_json.values())[0]
    df = pd.DataFrame.from_records(x)

    EQUITY_HIST = def_dict['NSE']['EQUITY_HIST']
    df = df[EQUITY_HIST.keys()]
    df = df.rename(columns=EQUITY_HIST)
    df = df.assign(Date=(pd.to_datetime(df.Date)+ pd.Timedelta(hours=16))\
                .dt.tz_localize('Asia/Calcutta'))
    
    return df


def clean_opt_hist(opt_json: dict, 
                      OPT_HIST: dict = OPT_HIST) -> pd.DataFrame:
    
    """Cleans option history and makes it a df"""
    
    # prepare the mother df
    x = list(opt_json.values())[0]
    df = pd.DataFrame.from_records(x)

    OPT_HIST = def_dict['NSE']['OPT_HIST']
    df = df[OPT_HIST.keys()]
    df = df.rename(columns=OPT_HIST)

    # clean the dates and numerics
    df_dates = df.iloc[:, 2:4].apply(pd.to_datetime)

    dte = (df_dates['Expiry'].sub(df_dates['Date']).dt.days)
    dte.rename('dte', inplace=True)

    df_nums = df.iloc[:, 4:]

    df_out = pd.concat([df.iloc[:, :2], df_dates, dte, df_nums], axis=1)

    return df_out


def is_nse_open() -> bool:

    """Gives a True if nse is open"""

    url = 'https://nseindia.com/api/marketStatus'
    js = nse_json(url) # works!
    nse_is_open = js[list(js.keys())[0]][0]['marketStatus'] not in ['Close', 'Closed']

    return nse_is_open



if __name__ == "__main__":
    # df = get_nse_syms()
    # df = nse_get_fno_lot_sizes()
    # df = get_nse_chain('SBIN')
    # df = make_chains(pd.Series(['NIFTY', 'SBIN']))
    # df = equity_prices()
    # df = index_prices()
    # df = get_prices()

    

    print(df)
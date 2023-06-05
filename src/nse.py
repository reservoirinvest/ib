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

from support import get_dte

BAR_FORMAT = "{desc:<21}{percentage:3.0f}%|{bar:15}{r_bar}"
log = logging.getLogger('log')

indices = ['NIFTY','FINNIFTY','BANKNIFTY']

# prevent urllib DEBUG connectionpool logs from nsepython requests
logging.getLogger('urllib3').setLevel(logging.WARNING)

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:74.0) Gecko/20100101 Firefox/74.0'
}

def nsefetch(payload):
    try:
        output = requests.get(payload,headers=headers).json()
    except ValueError:
        s =requests.Session()
        output = s.get("http://nseindia.com",headers=headers)
        output = s.get(payload,headers=headers).json()
    return output


def nsesymbolpurify(symbol):
    symbol = symbol.replace('&','%26') #URL Parse for Stocks Like M&M Finance
    return symbol

def nse_optionchain_scrapper(symbol):
    symbol = nsesymbolpurify(symbol)
    if any(x in symbol for x in indices):
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
    
    # convert & to %26
    lots_df = lots_df.assign(symbol=lots_df.symbol.str.replace('&', '%26'))
    
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


def nse_json(url: str):
    """Fetch json from nse for the url provided"""
    try:
        output = requests.get(url,headers=headers).json()

    except ValueError:
        try:
            s=requests.Session()
            output = s.get("http://nseindia.com",headers=headers)
            output = s.get(url,headers=headers).json()

        except ValueError: # for csv loads generating JSONDecodeError
            output = requests.get(url).text
        
    return output



def get_nse_syms() -> pd.DataFrame:
    """Generates symbols for nse with expiry months having lots"""

    # get symbols url
    url = "https://www.nseindia.com/api"
    url = url + "/equity-stockIndices?index=SECURITIES%20IN%20F%26O"

    # get the json for stocks
    njs = nse_json(url)
    equities = [njs['data'][x]['symbol'] for x in range(len(njs['data']))]
    nselist = indices + equities

    df_syms = pd.DataFrame({'symbol': nselist})

    # introduce `secType`
    df_syms.insert(1, 'secType', 
                    np.where(df_syms.symbol.str.contains('NIFTY'), 'IND', 'STK'))

    # introduce `exchange`
    df_syms.insert(2, 'exchange', 'NSE')

    # make ib friendly symbols
    df_syms.insert(1, 'ib_sym', df_syms.symbol.apply(nse2ib_symbol_convert))

    return df_syms



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

        colmap = { 'underlying': 'symbol', 'expiryDate': 'expiry', 'strikePrice': 'strike',  
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

        # df = df.assign(lot=nse_get_fno_lot_sizes(symbol=symbol))

        df = df.assign(opt_iv = df.opt_iv/100)
        df = df.assign(right=df.right.str[:1])
        df = df.sort_values(['expiry', 'right', 'strike'], 
                        ascending=[True, False, True]).reset_index(drop=True)

        return df

def make_chains(df_syms: pd.DataFrame, savepath: str = '') -> pd.DataFrame:
    """Generates option chains for all NSE F&Os with live prices"""
    dfs = []
    tq_scripts = tqdm(df_syms.symbol, bar_format = BAR_FORMAT)

    for s in tq_scripts:
        tq_scripts.set_description(f"{s}")
        try:
            dfs.append(get_nse_chain(s))
        except KeyError as e:
            log.error(f"{s} has error {e}")

    # assemble the chains
    nse_chains = pd.concat(dfs, ignore_index=True)

    # get the dtes for the chains
    exchange = df_syms.exchange.iloc[0]
    dte = nse_chains.expiry.apply(lambda x: get_dte(x, exchange))

    # insert the dtes
    try:
        nse_chains.insert(5, 'dte', dte)
    except ValueError:
        log.warning('dte already in nse_chains, will be refreshed')
        nse_chains = nse_chains.assign(dte=dte)

    if savepath:
        nse_chains.to_json(savepath, date_format = 'iso')

    return nse_chains


def get_nse_hist(symbol: str, 
        is_index: bool,
        days: int=365 ) -> pd.DataFrame:
    
    end = datetime.now().date()
    start = end-timedelta(days=days)
    cols = {'Date': 'date',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
            'Turnover': 'turnover',
            'Symbol': 'symbol',
            'Series': 'series',
            'Prev Close': 'prev_cls',
            'Last': 'last',
            'VWAP': 'vwap',
            'Trades': 'trades',
            'Deliverable Volume': 'd_volume',
            '%Deliverble': 'pct_delv'}

    # call
    data = get_history(symbol=symbol, start=start, end=end, index=is_index).reset_index()

    if data.empty:
            logging.error(f'Could not generate history for {symbol}')
            data = None 
            return data

    data.insert(0, 'date', data.Date.apply(pd.to_datetime)) 
    data.drop('Date', axis=1, inplace=True) # remove old Date field

    # convert to end of day India time
    date=(data.date + pd.Timedelta(hours=16)).dt.tz_localize('Asia/Calcutta')
    data = data.assign(date=date)

    # if index, put in symbol column
    if is_index:
            data.insert(0, 'Symbol', symbol)

    data.rename(columns=cols, inplace=True)

    return data



def clean_prices(df: pd.DataFrame) -> pd.DataFrame:
    "Cleans up NSE prices to give numeric values and times"

    # stage a numeric df, removing commas from price text
    df = df.apply(lambda x: 
                  pd.to_numeric(x.astype(str).str.replace(',',''), 
                                errors='ignore'))
    df2 = df.apply(pd.to_numeric, errors='coerce')
    df2 = df2.dropna(axis=1, how='all') \
          .drop(columns=['timeVal', 'percChange', 'indexOrder'], errors='ignore')

    # Prep the mother df
    df1 = df.drop(columns=list(df2.columns))

    # convert xDt to date dict
    try: # to get equityies only
        s = pd.Series(df.xDt.unique())
        di = pd.to_datetime(s, errors='coerce').set_axis(s).to_dict()
        df1 = df1.assign(xDt = df1.xDt.map(di))
    except AttributeError:
        pass

    df_final = pd.concat([df1, df2], axis=1)

    return df_final



def equity_prices() -> pd.DataFrame:
    """Gets all live equity prices"""

    url = "https://www1.nseindia.com/live_market/dynaContent/live_watch/stock_watch/foSecStockWatch.json"

    js = nse_json(url)
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
    di = {'NIFTY 50': 'NIFTY', 'NIFTY BANK': 'BANKNIFTY', 'INDIA VIX': 
          'INDIAVIX', 'NIFTY FIN SERVICE': 'FINNIFTY',
          'NIFTY MIDCAP 50': 'MIDCPNIFTY',}
    df.insert(0, 'symbol', df.indexName.map(di).fillna(df.indexName))

    if important:
        df = df[df.symbol.isin(di.values())]

    df = clean_prices(df) # handle obj -> numerics, times

    return df



def is_nse_open() -> bool:
    """Gives a True if nse is open"""

    url = 'https://nseindia.com/api/marketStatus'
    js = nse_json(url) # works!
    nse_is_open = js[list(js.keys())[0]][0]['marketStatus'] != 'Closed'

    return nse_is_open



def get_prices() -> pd.DataFrame:
    """Get the underlying prices"""

    df_eq = equity_prices()
    df_ix = index_prices()

    df = pd.concat([df_eq, df_ix], ignore_index=True)
    ltp = df['last'].combine_first(df.ltP)
    df = df.assign(last=ltp, ltP=ltp)

    # harmonize time to time value
    df.timeVal = pd.to_datetime(df.timeVal).dt.tz_localize('Asia/Calcutta')
    df.timeVal = df.timeVal.max()

    return df



if __name__ == "__main__":
    # df = get_nse_syms()
    # df = nse_get_fno_lot_sizes()
    # df = get_nse_chain('NIFTY')
    # df = make_chains(pd.DataFrame({'symbol': ['RELIANCE', 'NIFTY']}).assign(exchange='NSE'))
    df = get_prices()


    # df = get_nse_hist('M&M', True, 365) # !!! Not working !!!
    

    print(df)
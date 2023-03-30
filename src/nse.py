import logging
from datetime import datetime, timedelta
from io import StringIO

import numpy as np
import pandas as pd
import requests
from nsepy import get_history
from nsepython import nse_get_fno_lot_sizes, nse_optionchain_scrapper

from .support import nse2ib_symbol_convert

# prevent urllib DEBUG connectionpool logs from nsepython requests
logging.getLogger('urllib3').setLevel(logging.WARNING)

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:74.0) Gecko/20100101 Firefox/74.0'
}



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



def get_nse_syms(onlyWithHist: bool=True) -> pd.DataFrame:
    """Generates symbols for nse with expiry months having lots"""

    url = "https://www1.nseindia.com/content/fo/fo_mktlots.csv"

    try:
        req = requests.get(url)
        if req.status_code == 404:
            print(f"\n{url} URL contents not correct. 404 error!")
        df_syms = pd.read_csv(StringIO(req.text))
    except requests.ConnectionError as e:
        print(f"Connection Error {e}")
    except pd.errors.ParserError as e:
        print(f"Parser Error {e}")

    df_syms = df_syms[list(df_syms)[1:5]]

    # strip whitespace from columns and make it lower case
    df_syms.columns = df_syms.columns.str.strip().str.lower()

    # strip all string contents of whitespaces
    df_syms = df_syms.applymap(lambda x: x.strip()
                                        if type(x) is str else x)

    # remove 'Symbol' row
    df_syms = df_syms[df_syms.symbol != "Symbol"]

    # drop symbols not able to generate history!
    if onlyWithHist:
        searchfor = ['MIDCPNIFTY', 'FINNIFTY']
        drop_me = df_syms[df_syms.symbol.str.contains('|'.join(searchfor))].index
        df_syms.drop(drop_me, inplace=True)

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
        df = df.assign(lot=nse_get_fno_lot_sizes(symbol=symbol))
        df = df.assign(opt_iv = df.opt_iv/100)
        df = df.assign(right=df.right.str[:1])
        df = df.sort_values(['expiry', 'right', 'strike'], 
                        ascending=[True, False, True]).reset_index(drop=True)

        return df



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
    df = get_nse_syms()
    # df = get_nse_chain('NIFTY')
    # df = get_nse_hist('M&M', True, 365)

    print(df)
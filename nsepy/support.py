# Old working support file

import datetime
import inspect
import re
import ast

import pandas as pd
import pytz
import requests
from bs4 import BeautifulSoup

from commons import *


#Constants

with open(r'C:\Users\kashi\python\zArchive\zOthers\symbol_count.txt', 'r') as f:
    s = f.read()
symbol_count = ast.literal_eval(s)

headers = {
    'Connection': 'keep-alive',
    'Cache-Control': 'max-age=0',
    'DNT': '1',
    'Upgrade-Insecure-Requests': '1',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.79 Safari/537.36',
    'Sec-Fetch-User': '?1',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
    'Sec-Fetch-Site': 'none',
    'Sec-Fetch-Mode': 'navigate',
    'Accept': '*/*',
    'Accept-Encoding': 'gzip, deflate, br',
    'Accept-Language': 'en-US,en;q=0.9,hi;q=0.8',
    'Connection': 'keep-alive',
    'Host': 'www1.nseindia.com',
    'X-Requested-With': 'XMLHttpRequest'
}

opt_headers = {
    'Connection': 'keep-alive',
    'Cache-Control': 'max-age=0',
    'DNT': '1',
    'Upgrade-Insecure-Requests': '1',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.79 Safari/537.36',
    'Sec-Fetch-User': '?1',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
    'Sec-Fetch-Site': 'none',
    'Sec-Fetch-Mode': 'navigate',
    'Accept-Encoding': 'gzip, deflate, br',
    'Accept-Language': 'en-US,en;q=0.9,hi;q=0.8',
}

session = requests.Session()

indices = ['NIFTY','FINNIFTY','BANKNIFTY']

dd_mmm_yyyy = StrDate.default_format(format="%d-%b-%Y")
dd_mm_yyyy = StrDate.default_format(format="%d-%m-%Y")

EQUITY_SCHEMA = [str, str,
                 dd_mmm_yyyy,
                 float, float, float, float,
                 float, float, float, int, float,
                 int, int, float]
EQUITY_HEADERS = ["Symbol", "Series", "Date", "Prev Close",
                  "Open", "High", "Low", "Last", "Close", "VWAP",
                  "Volume", "Turnover", "Trades", "Deliverable Volume",
                  "%Deliverble"]
EQUITY_SCALING = {"Turnover": 100000,
                  "%Deliverble": 0.01}

INDEX_SCHEMA = [dd_mmm_yyyy,
                float, float, float, float,
                int, float]
INDEX_HEADERS = ['Date',
                 'Open', 'High', 'Low', 'Close',
                 'Volume', 'Turnover']
INDEX_SCALING = {'Turnover': 10000000}

OPTION_SCHEMA = [str, dd_mmm_yyyy, dd_mmm_yyyy, str, float,
                 float, float, float, float,
                 float, float, int, float,
                 float, int, int, float]
OPTION_HEADERS = ['Symbol', 'Date', 'Expiry', 'Option Type', 'Strike Price',
                  'Open', 'High', 'Low', 'Close',
                  'Last', 'Settle Price', 'Number of Contracts', 'Turnover',
                  'Premium Turnover', 'Open Interest', 'Change in OI', 'Underlying']
OPTION_SCALING = {"Turnover": 100000,
                  "Premium Turnover": 100000}

VIX_INDEX_SCHEMA = [dd_mmm_yyyy,
                    float, float, float, float,
                    float, float, float]
VIX_INDEX_HEADERS = ['Date',
                     'Open', 'High', 'Low', 'Close',
                     'Previous', 'Change', '%Change']
VIX_SCALING = {'%Change': 0.01}

FUTURES_SCHEMA = [str, dd_mmm_yyyy, dd_mmm_yyyy,
                  float, float, float, float,
                  float, float, int, float,
                  int, int, float]

FUTURES_HEADERS = ['Symbol', 'Date', 'Expiry',
                   'Open', 'High', 'Low', 'Close',
                   'Last', 'Settle Price', 'Number of Contracts', 'Turnover',
                   'Open Interest', 'Change in OI', 'Underlying']
FUTURES_SCALING = {"Turnover": 100000}

URLFetchSession = partial(URLFetch, session=session,
                          headers=headers)

derivative_history_url = partial(
    URLFetchSession(
        url='http://www1.nseindia.com/products/dynaContent/common/productsSymbolMapping.jsp?',
        headers = {**headers, **{'Referer': 'https://www1.nseindia.com/products/content/derivatives/equities/historical_fo.htm'}}
        #headers = (lambda a,b: a.update(b) or a)(headers.copy(),{'Referer': 'https://www1.nseindia.com/products/content/derivatives/equities/historical_fo.htm'})
        ),
    segmentLink=9,
    symbolCount='')

index_vix_history_url = URLFetchSession(
    url='http://www1.nseindia.com/products/dynaContent/equities/indices/hist_vix_data.jsp')

index_history_url = URLFetchSession(
    url='http://www1.nseindia.com/products/dynaContent/equities/indices/historicalindices.jsp')

equity_history_url_full = URLFetchSession(
    url='http://www1.nseindia.com/products/dynaContent/common/productsSymbolMapping.jsp')

equity_history_url = partial(equity_history_url_full,
                             dataType='PRICEVOLUMEDELIVERABLE',
                             segmentLink=3, dateRange="")

symbol_count_url = URLFetchSession(
    url='http://www1.nseindia.com/marketinfo/sym_map/symbolCount.jsp')


def nsesymbolpurify(symbol):
    symbol = symbol.replace('&','%26') #URL Parse for Stocks Like M&M Finance
    return symbol

def nsefetch(payload, headers=headers):
    try:
        output = requests.get(payload,headers=headers).json()
    except ValueError:
        s =requests.Session()
        output = s.get("http://nseindia.com",headers=headers)
        output = s.get(payload,headers=headers).json()
    return output

def symlist():

    positions = nsefetch('https://www.nseindia.com/api/equity-stockIndices?index=SECURITIES%20IN%20F%26O')

    nselist=[]

    i=0
    for x in range(i, len(positions['data'])):
        nselist=nselist+[positions['data'][x]['symbol']]

    return nselist

def nse_optionchain_scrapper(symbol):
    symbol = nsesymbolpurify(symbol)
    if any(x in symbol for x in indices):
        payload = nsefetch('https://www.nseindia.com/api/option-chain-indices?symbol='+symbol, headers=opt_headers)
    else:
        payload = nsefetch('https://www.nseindia.com/api/option-chain-equities?symbol='+symbol, headers=opt_headers)
    return payload

def nse_get_fno_lot_sizes(symbol="all",mode="list"):
    url="https://archives.nseindia.com/content/fo/fo_mktlots.csv"

    if(mode=="list"):
        s=requests.get(url).text
        res_dict = {}
        for line in s.split('\n'):
          if line != '' and re.search(',', line) and (line.casefold().find('symbol') == -1):
              (code, name) = [x.strip() for x in line.split(',')[1:3]]
              res_dict[code] = int(name)
        if(symbol=="all"):
            return res_dict
        if(symbol!=""):
            return res_dict[symbol.upper()]

    if(mode=="pandas"):
        payload = pd.read_csv(url)
        if(symbol=="all"):
            return payload
        else:
            payload = payload[(payload.iloc[:, 1] == symbol.upper())]
            return payload

def nse_opts(symbol: str) -> pd.DataFrame:

    scraped = nse_optionchain_scrapper(symbol)
    raw_dict = scraped['records']['data']
    dfs = [pd.DataFrame.from_dict(raw_dict[i]).transpose()[2:] 
                for i in range(len(raw_dict))]

    df = pd.concat(dfs).drop(columns='identifier')\
                   .rename_axis('right').reset_index()

    float_cols = ['strikePrice', 'pchangeinOpenInterest', 'impliedVolatility', 
                'lastPrice', 'change', 'pChange', 'bidprice', 'askPrice' , 
                'underlyingValue' ]
    int_cols = ['openInterest', 'changeinOpenInterest', 'totalTradedVolume', 
                'totalBuyQuantity', 'totalSellQuantity', 'bidQty', 'askQty']
    df[float_cols] = df[float_cols].astype('float32')
    df[int_cols] = df[int_cols].astype('int64')
    df = df.assign(expiryDate=pd.to_datetime(df.expiryDate))

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


    # ..get accurate dte
    nse_tz = pytz.timezone('Asia/Kolkata')
    now = datetime.datetime.now(tz=nse_tz).timestamp()
    nse_tz_expiry = df.expiry.apply(lambda x: nse_tz.localize(datetime.datetime.combine(x.date(), datetime.time(18,0))).timestamp())
    dte = (nse_tz_expiry.apply(datetime.datetime.fromtimestamp) - datetime.datetime.fromtimestamp(now)).apply(datetime.timedelta.total_seconds)/24/3600
    df = df.assign(dte=dte)

    return df

def get_history(symbol, start, end, index=False, futures=False, option_type="",
                expiry_date=None, strike_price="", series='EQ'):
    """This is the function to get the historical prices of any security (index,
        stocks, derviatives, VIX) etc.

        Args:
            symbol (str): Symbol for stock, index or any security
            start (datetime.date): start date
            end (datetime.date): end date
            index (boolean): False by default, True if its a index
            futures (boolean): False by default, True for index and stock futures
            expiry_date (datetime.date): Expiry date for derivatives, Compulsory for futures and options
            option_type (str): It takes "CE", "PE", "CA", "PA" for European and American calls and puts
            strike_price (int): Strike price, Compulsory for options
            series (str): Defaults to "EQ", but can be "BE" etc (refer NSE website for details)

        Returns:
            pandas.DataFrame : A pandas dataframe object 

        Raises:
            ValueError: 
                        1. strike_price argument missing or not of type int when options_type is provided
                        2. If there's an Invalid value in option_type, valid values-'CE' or 'PE' or 'CA' or 'CE'
                        3. If both futures='True' and option_type='CE' or 'PE'
    """
    frame = inspect.currentframe()
    args, _, _, kwargs = inspect.getargvalues(frame)
    del(kwargs['frame'])
    start = kwargs['start']
    end = kwargs['end']
    if (end - start) > datetime.timedelta(130):
        kwargs1 = dict(kwargs)
        kwargs2 = dict(kwargs)
        kwargs1['end'] = start + datetime.timedelta(130)
        kwargs2['start'] = kwargs1['end'] + datetime.timedelta(1)

        t1 = ThreadReturns(target=get_history, kwargs=kwargs1)
        t2 = ThreadReturns(target=get_history, kwargs=kwargs2)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        return pd.concat((t1.result, t2.result))
    else:
        return get_history_quanta(**kwargs)

def get_history_quanta(**kwargs):
    url, params, schema, headers, scaling = validate_params(**kwargs)
    df = url_to_df(url=url,
                   params=params,
                   schema=schema,
                   headers=headers, scaling=scaling)
    return df

def url_to_df(url, params, schema, headers, scaling={}):
    resp = url(**params)
    bs = BeautifulSoup(resp.text, 'lxml')
    tp = ParseTables(soup=bs,
                     schema=schema,
                     headers=headers, index="Date")
    df = tp.get_df()
    for key, val in six.iteritems(scaling):
        df[key] = val * df[key]
    return df

def validate_params(symbol, start, end, index=False, futures=False, option_type="",
                    expiry_date=None, strike_price="", series='EQ'):
    """
                symbol = "SBIN" (stock name, index name and VIX)
                start = date(yyyy,mm,dd)
                end = date(yyyy,mm,dd)
                index = True, False (True even for VIX)
                ---------------
                futures = True, False
                option_type = "CE", "PE", "CA", "PA"
                strike_price = integer number
                expiry_date = date(yyyy,mm,dd)
    """

    params = {}

    if start > end:
        raise ValueError('Please check start and end dates')

    if (futures and not option_type) or (not futures and option_type):  # EXOR
        params['symbol'] = symbol
        params['dateRange'] = ''
        params['optionType'] = 'select'
        params['strikePrice'] = ''
        params['fromDate'] = start.strftime('%d-%b-%Y')
        params['toDate'] = end.strftime('%d-%b-%Y')
        url = derivative_history_url

        try:
            params['expiryDate'] = expiry_date.strftime("%d-%m-%Y")
        except AttributeError as e:
            raise ValueError(
                'Derivative contracts must have expiry_date as datetime.date')

        option_type = option_type.upper()
        if option_type in ("CE", "PE", "CA", "PA"):
            if not isinstance(strike_price, int) and not isinstance(strike_price, float):
                raise ValueError(
                    "strike_price argument missing or not of type int or float")
            # option specific
            if index:
                params['instrumentType'] = 'OPTIDX'
            else:
                params['instrumentType'] = 'OPTSTK'
            params['strikePrice'] = strike_price
            params['optionType'] = option_type
            schema = OPTION_SCHEMA
            headers = OPTION_HEADERS
            scaling = OPTION_SCALING
        elif option_type:
            # this means that there's an invalid value in option_type
            raise ValueError(
                "Invalid value in option_type, valid values-'CE' or 'PE' or 'CA' or 'CE'")
        else:
            # its a futures request
            if index:
                if symbol == 'INDIAVIX':
                    params['instrumentType'] = 'FUTIVX'
                else:
                    params['instrumentType'] = 'FUTIDX'
            else:
                params['instrumentType'] = 'FUTSTK'
            schema = FUTURES_SCHEMA
            headers = FUTURES_HEADERS
            scaling = FUTURES_SCALING
    elif futures and option_type:
        raise ValueError(
            "select either futures='True' or option_type='CE' or 'PE' not both")
    else:  # its a normal request

        if index:
            if symbol == 'INDIAVIX':
                params['fromDate'] = start.strftime('%d-%b-%Y')
                params['toDate'] = end.strftime('%d-%b-%Y')
                url = index_vix_history_url
                schema = VIX_INDEX_SCHEMA
                headers = VIX_INDEX_HEADERS
                scaling = VIX_SCALING
            else:
                if symbol in DERIVATIVE_TO_INDEX:
                    params['indexType'] = DERIVATIVE_TO_INDEX[symbol]
                else:
                    params['indexType'] = symbol
                params['fromDate'] = start.strftime('%d-%m-%Y')
                params['toDate'] = end.strftime('%d-%m-%Y')
                url = index_history_url
                schema = INDEX_SCHEMA
                headers = INDEX_HEADERS
                scaling = INDEX_SCALING
        else:
            params['symbol'] = symbol
            params['series'] = series
            params['symbolCount'] = get_symbol_count(symbol)
            params['fromDate'] = start.strftime('%d-%m-%Y')
            params['toDate'] = end.strftime('%d-%m-%Y')
            url = equity_history_url
            schema = EQUITY_SCHEMA
            headers = EQUITY_HEADERS
            scaling = EQUITY_SCALING

    return url, params, schema, headers, scaling

def get_symbol_count(symbol):
    try:
        return symbol_count[symbol]
    except:
        cnt = symbol_count_url(symbol=symbol).text.lstrip().rstrip()
        symbol_count[symbol] = cnt
        return cnt

def nse_most_active(type="securities",sort="value"):
    payload = nsefetch("https://www.nseindia.com/api/live-analysis-most-active-"+type+"?index="+sort+"", headers=opt_headers)
    payload = pd.DataFrame(payload["data"])
    return payload

def nse_get_top_losers():
    positions = nsefetch('https://www.nseindia.com/api/equity-stockIndices?index=SECURITIES%20IN%20F%26O', headers=opt_headers)
    df = pd.DataFrame(positions['data'])
    df = df.sort_values(by="pChange")
    return df.head(5)

def nse_get_top_gainers():
    positions = nsefetch('https://www.nseindia.com/api/equity-stockIndices?index=SECURITIES%20IN%20F%26O', headers=opt_headers)
    df = pd.DataFrame(positions['data'])
    df = df.sort_values(by="pChange" , ascending = False)
    return df.head(5)

def nse_get_advances_declines(mode="pandas"):
    try:
        if(mode=="pandas"):
            positions = nsefetch('https://www.nseindia.com/api/equity-stockIndices?index=SECURITIES%20IN%20F%26O', headers=opt_headers)
            return pd.DataFrame(positions['data'])
        else:
            return nsefetch('https://www.nseindia.com/api/equity-stockIndices?index=SECURITIES%20IN%20F%26O', headers=opt_headers)
    except:
        print("\nPandas is not working for some reason.\n")
        return nsefetch('https://www.nseindia.com/api/equity-stockIndices?index=SECURITIES%20IN%20F%26O', headers=opt_headers)

def nse_preopen(key="NIFTY",type="pandas"):
    payload = nsefetch("https://www.nseindia.com/api/market-data-pre-open?key="+key+"", headers=opt_headers)
    if(type=="pandas"):
        payload = pd.DataFrame(payload['data'])
        payload  = pd.json_normalize(payload['metadata'])
        return payload
    else:
        return payload

def nse_preopen_movers(key="FO",filter=1.5):
    preOpen_gainer=nse_preopen(key)
    return preOpen_gainer[preOpen_gainer['pChange'] >1.5],preOpen_gainer[preOpen_gainer['pChange'] <-1.5]

if __name__ == "__main__":

    symbol = 'RELIANCE'
    period = 365 # days
    end = datetime.datetime.now().date()
    start = end - datetime.timedelta(days=period)

    df_history = get_history(symbol, start=start, end=end)
    df_opts = nse_opts(symbol)
    
    print(df_history)
    print("\n\n")
    print(df_opts)

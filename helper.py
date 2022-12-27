# Helper files 

import datetime
import re
from pathlib import Path

import pandas as pd
import pytz
import requests
import yaml

root = Path.cwd()
config_path = root / 'conf' / 'config.yml'


# Constants from configuration file
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

headers = config['headers']


# Constants
indices = ['NIFTY','FINNIFTY','BANKNIFTY']

def nse_fetch(payload: str, headers: dict):
    """Fetch payload"""
    try:
        output = requests.get(payload,headers=headers).json()
    except ValueError:
        s = requests.Session()
        output = s.get("http://nseindia.com",headers=headers)
        output = s.get(payload,headers=headers).json()
    return output



def nse_symbol_purify(symbol: str) -> str:
    """URL Parse for Stocks Like M&M Finance"""
    symbol = symbol.replace('&','%26')
    return symbol.upper()



def nse_fnos() -> set:
    """Get FNO symbols"""
    
    payload = 'https://www.nseindia.com/api/equity-stockIndices?index=SECURITIES%20IN%20F%26O'
    positions = nse_fetch(payload=payload, headers=headers)

    data = positions['data']
    symbols = {data[i]['symbol'] for i in range(len(data))}

    symbols |= set(indices)
    
    return symbols



def nse_optionchain_scrapper(symbol):

    symbol = nse_symbol_purify(symbol)
    
    if any(x in symbol for x in indices):
        payload = nse_fetch('https://www.nseindia.com/api/option-chain-indices?symbol='+symbol, headers=headers)
    else:
        payload = nse_fetch('https://www.nseindia.com/api/option-chain-equities?symbol='+symbol, headers=headers)
    
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
    """Gets all nse options for the symbol"""

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
            'totalTradedVolume': 'volume', 'totalBuyQuantity': 'buyQty', 
            'totalSellQuantity': 'sellQty', 'pChange': 'pChange', 
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

    

if __name__ == "__main__":
    symbol = 'reliance'
    print(nse_opts(symbol).head())
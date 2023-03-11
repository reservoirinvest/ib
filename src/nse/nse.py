import logging
from io import StringIO

import numpy as np
import pandas as pd
import requests
from nsepython import nse_get_fno_lot_sizes, nse_optionchain_scrapper

# prevent urllib DEBUG connectionpool logs from nsepython requests
logging.getLogger('urllib3').setLevel(logging.WARNING) 

def get_nse_syms() -> pd.DataFrame:
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

    # introduce `secType`
    df_syms.insert(1, 'secType', np.where(df_syms.symbol.str.contains('NIFTY'), 'IND', 'STK'))

    # introduce `exchange`
    df_syms.insert(2, 'exchange', 'NSE')

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

if __name__ == "__main__":
    # df = get_nse_syms()
    df = get_nse_chain('NIFTY')
    print(df)
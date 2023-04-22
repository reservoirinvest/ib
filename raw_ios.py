# testing history extraction of data from nse website

import datetime
import requests
import json

from nsepython import equity_history_virgin, derivative_history_virgin, expiry_list, index_history

# Extracted IO URLs for equity and option histories using request.get() with common headers for both
raw_eq_hist_io = 'https://www.nseindia.com/api/historical/cm/equity?symbol=SBIN&series=[%22EQ%22]&from=02-03-2023&to=21-04-2023'
raw_opt_eq_hist_io = 'https://www.nseindia.com/api/historical/fo/derivatives?&from=02-03-2023&to=21-04-2023&optionType=PE&strikePrice=500.00&expiryDate=27-Apr-2023&instrumentType=OPTSTK&symbol=SBIN'
raw_opt_index_hist_io = "https://www.nseindia.com/api/historical/fo/derivatives?symbol=NIFTY"



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
    'Accept-Encoding': 'gzip, deflate, br',
    'Accept-Language': 'en-US,en;q=0.9,hi;q=0.8',
}


# Extracted IO for indexes using request.post() using index headers, different from eq and opts
data = str({'name':'NIFTY BANK','startDate':'02-Mar-2023','endDate':'21-Apr-2023'})
url = 'https://niftyindices.com/Backpage.aspx/getHistoricaldatatabletoString'

index_headers = {
    'Connection': 'keep-alive',
    'sec-ch-ua': '" Not;A Brand";v="99", "Google Chrome";v="91", "Chromium";v="91"',
    'Accept': 'application/json, text/javascript, */*; q=0.01',
    'DNT': '1',
    'X-Requested-With': 'XMLHttpRequest',
    'sec-ch-ua-mobile': '?0',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.77 Safari/537.36',
    'Content-Type': 'application/json; charset=UTF-8',
    'Origin': 'https://niftyindices.com',
    'Sec-Fetch-Site': 'same-origin',
    'Sec-Fetch-Mode': 'cors',
    'Sec-Fetch-Dest': 'empty',
    'Referer': 'https://niftyindices.com/reports/historical-data',
    'Accept-Language': 'en-US,en;q=0.9,hi;q=0.8',
}

index_history = requests.post(url=url, headers=index_headers,  data=str(data)).json()

# Parameters
symbol = 'SBIN'
series = 'EQ'

end_date = datetime.datetime.now().date()
days_of_history = 50
intv = 50 # interval chunks

# Stage the equity dates
eq_start_date = (end_date - datetime.timedelta(days = days_of_history))\
    .strftime('%d-%m-%Y')
eq_end_date = end_date.strftime('%d-%m-%Y')

eq_headers = {
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

# Stage the option history dates
opt_start_date = (end_date - datetime.timedelta(days = days_of_history))\
    .strftime('%d-%m-%Y')
opt_end_date = end_date.strftime('%d-%m-%Y')
instrumentType = 'options'

strike = 500
right = 'PE'

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

# Stage index history dates
index_start_date = (end_date - datetime.timedelta(days = days_of_history))\
    .strftime('%d-%b-%Y')
index_end_date = end_date.strftime('%d-%b-%Y')



if __name__ == "__main__":

    ## ... for Equity history
    # eq_result = requests.get(raw_eq_hist_io, 
    #                       headers=eq_headers).json()
    # print(f"\n raw_eq_history_io = {raw_eq_hist_io}\n\neq_result = {eq_result}")

    # history = equity_history_virgin(symbol=symbol, series=series, start_date=eq_start_date, end_date=eq_end_date)
    # print(history)

    ###----------------------------------------------------------

    ## ... for option history
    ## gives list output ['27-Apr-2023', '25-May-2023', '29-Jun-2023']
    # expiry = expiry_list(symbol=symbol) 
    # expiry= expiry[0]

    # opt_history = derivative_history_virgin(symbol=symbol, start_date=opt_start_date, 
    #                                         end_date=opt_end_date, instrumentType=instrumentType, 
    #                                         expiry_date=expiry, strikePrice=strike, optionType=right)
    # print(opt_history)

    # opt_result = requests.get(raw_opt_eq_hist_io, 
    #                       headers=opt_headers).json()
    # print(f"\n raw_opt_history_io = {raw_opt_eq_hist_io}\n\nopt_result = {opt_result}")

    ###----------------------------------------------------------

    ## ... for index history
    # symbol = 'NIFTY BANK' # Needs to be correct for BANKNIFTY, NIFTYIT, etc
    # print(index_history(symbol, index_start_date, index_end_date))

    # index_history_result = requests.post('https://niftyindices.com/Backpage.aspx/getHistoricaldatatabletoString', 
    #                                      headers=index_headers,  data=str(data)).json()
    # print(index_history_result)

    ###----------------------------------------------------------

    ## ... for index option history

    # symbol = 'NIFTY BANK'

    # gives list output ['27-Apr-2023', '25-May-2023', '29-Jun-2023']
    # expiry = '27-Apr-2023'

    # index_opt_history = derivative_history_virgin(symbol=symbol, start_date=opt_start_date, 
    #                                         end_date=opt_end_date, instrumentType='Index options', 
    #                                         expiry_date=expiry, strikePrice=strike, optionType='Put')
    # print(index_opt_history)

    result = requests.get(raw_opt_index_hist_io, 
                          headers=headers)
    print(result)

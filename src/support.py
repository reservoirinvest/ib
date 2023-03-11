import datetime
from collections import defaultdict
from pathlib import Path
from typing import Union

import dateutil
import pytz


def get_dte(dt: Union[datetime.datetime, datetime.date], exchange: str) -> float:
    """Get accurate dte"""

    tz_dict = {'nse': ('Asia/Kolkata', 18), 'snp': ('EST', 16)}
    tz, hr = tz_dict[exchange.lower()]

    mkt_tz = pytz.timezone(tz)
    mkt_close_time = datetime.time(hr, 0)

    now = datetime.datetime.now(tz=mkt_tz)

    mkt_close = datetime.datetime.combine(dt.date(), mkt_close_time).astimezone(mkt_tz)

    dte = (mkt_close-now).total_seconds()/24/3600

    return dte

def pickle_age(data_path: Path) -> dict:
    """Gets age of the pickles in a dict with relativedelta"""

    # Get all the pickles in data path provided
    pickles = Path(data_path).glob('*.pkl')
    d = {f.name: dateutil.relativedelta.relativedelta(datetime.datetime.now(), datetime.datetime.fromtimestamp(f.stat().st_mtime)) for f in pickles}

    return d

if __name__ == "__main__":
    dt = datetime.datetime(2023, 5, 13) # x is to be converted to mkt close
    exchange = 'nse'

    print(get_dte(dt, exchange))
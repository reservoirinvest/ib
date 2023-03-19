import asyncio
import logging
from pathlib import Path
from typing import Callable, Coroutine, Union

import numpy as np
import pandas as pd
from ib_insync import IB, Contract, util
from tqdm import tqdm

from src.dclass import NSE_Margin

BAR_FORMAT = "{desc:<10}{percentage:3.0f}%|{bar:25}{r_bar}{bar:-10b}"


def pre_process(cts):
    """Generates tuples for input to the engine"""

    try:
        symbol = cts.symbol
        output = ((cts, None), )

    except AttributeError as ae1:  # it's an iterable!
        try:
            symbols = [c.symbol for c in cts]

            if len(symbols) == 1:
                output = ((cts[0], None), )
            else:
                output = ((c, None) for c in cts)

        except AttributeError as ae2:  # 2nd value is MarketOrder!
            try:
                output = tuple(cts)
            except:
                logging.error(f"Unknown error in {ae2}")
                output = None

    return tuple(output)


# .make name for symbol being processed by the engine
def make_name(cts):
    """Generates name for contract(s)"""
    try:
        output = [
            c.symbol + c.lastTradeDateOrContractMonth[-4:] + c.right +
            str(c.strike) + ".." for c in cts
        ]

    except TypeError as te:  # single non-iterable element
        if cts != "":  # not empty!
            output = (cts.symbol + cts.lastTradeDateOrContractMonth[-4:] +
                      cts.right + str(cts.strike))
        else:
            output = cts

    except AttributeError as ae1:  # multiple (p, s) combination
        try:
            output = [
                c[0].symbol + c[0].lastTradeDateOrContractMonth[-4:] +
                c[0].right + str(c[0].strike) + ".." for c in cts
            ]
        except TypeError as te2:
            output = (cts[0].symbol +
                      cts[0].lastTradeDateOrContractMonth[-4:] + cts[0].right +
                      str(cts[0].strike))

    return output


async def nse_margin(ib: IB, co, **kwargs) -> pd.DataFrame:
    """Optimal execAsync: CONCURRENT=200 with TIMEOUT=FILL_DELAY=5.5"""

    df_margin = pd.DataFrame([NSE_Margin])
    df_empty = df_margin.iloc[0:0]

    try:
        FILL_DELAY = kwargs["FILL_DELAY"]
    except KeyError as ke:
        logging.error(
            f"Warning: No FILL_DELAY supplied!. 1.5 second default is taken"
        )
        FILL_DELAY = 1.5


    try:
        ct, o = pre_process(co)
    except ValueError as ve:
        logging.error(
            f"Error: {co} co supplied is incorrect! It should be a tuple(contract, order)"
        )
        df = df_empty

    async def wifAsync(ct, o):
        wif = ib.whatIfOrderAsync(ct, o)
        await asyncio.sleep(FILL_DELAY)
        return wif

    wif = await wifAsync(ct, o)

    if wif.done():

        res = wif.result()

        try:

            df = (util.df([ct]).iloc[:, :6].rename(
                columns={"lastTradeDateOrContractMonth": "expiry"}))

            df = df.join(
                util.df(
                    [res])[["initMarginChange", "maxCommission",
                            "commission"]])

        except TypeError as e:

            logging.error(f"Error: Unknown type of contract: {ct}, order: {o}" +
                  f" in margin wif: {wif}" +
                  f" giving error{e}")

            df = df_empty

        except IndexError as e:

            logging.error(f"Error: Index error for contract: {ct}, order: {o}" +
                  f" in margin wif: {wif}" +
                  f" giving error{e}")

            df = df_empty

        except KeyError as e:

            logging.error(f"Error: Key error for contract: {ct}, order: {o}" +
                  f" in margin wif: {wif}" +
                  f" giving error{e}")

            df = df_empty

    else:

        logging.error(
            f"Error: wif could not complete for contract: {ct.localSymbol}" +
            f". Try by increasing FILL_DELAY from > {FILL_DELAY} secs")

        df = df_empty

    # post-processing df
    df = df.assign(secType=ct.secType,
                   conId=ct.conId,
                   localSymbol=ct.localSymbol,
                   symbol=ct.symbol)
    df = df.assign(
        comm=df[["commission", "maxCommission"]].min(axis=1),
        margin=df.initMarginChange.astype("float"),
    )

    # Correct unrealistic margin and commission
    df = df(
        margin=np.where(df.margin > 1e7, np.nan, df.margin),
        comm=np.where(df.comm > 1e7, np.nan, df.comm),
    )

    # df = df[[
    #     "conId",
    #     "secType",
    #     "symbol",
    #     "strike",
    #     "right",
    #     "expiry",
    #     "localSymbol",
    #     "margin",
    #     "comm",
    # ]]

    cols_to_use = df.columns.difference(df_margin.columns)
    df = pd.merge(df, df_margin[cols_to_use], left_index=True, right_index=True, how='outer')

    return df


async def executeAsync(
    ib: IB(),
    algo: Callable[..., Coroutine],  # coro name
    cts: Union[Contract, pd.Series, list, tuple],  # list of contracts
    CONCURRENT: int = 44,  # to prevent overflows put 44 * (TIMEOUT-1)
    TIMEOUT: None = None,  # if None, no progress messages shown
    post_process: Callable[[set, Path, str],
                           pd.DataFrame] = None,  # If checkpoint is needed
    DATAPATH: Path = None,  # Necessary for post_process
    OP_FILENAME: str = "",  # output file name
    SHOW_TQDM: bool = True,  # Show tqdm bar instead of individual messages
    REUSE: bool = False,  # Reuse the OP_FILENAME supplied
    **kwargs,
) -> pd.DataFrame:

    tasks = set()
    results = set()
    remaining = pre_process(cts)
    last_len_tasks = 0  # tracking last length for error catch
    done = set()

    output = pd.DataFrame([]) # Corrected error from get_covers if it doesn't contain any scrips

    # Set pbar
    if SHOW_TQDM:
        pbar = tqdm(
            total=len(remaining),
            desc=f"{algo.__name__}: ",
            bar_format=BAR_FORMAT,
            ncols=80,
            leave=False,
        )

    # Get the results
    while len(remaining):

        # Tasks limited by concurrency
        if len(remaining) <= CONCURRENT:

            tasks.update(
                asyncio.create_task(algo(ib, c, **kwargs), name=make_name(c))
                for c in remaining)

        else:

            tasks.update(
                asyncio.create_task(algo(ib, c, **kwargs), name=make_name(c))
                for c in remaining[:CONCURRENT])

        # Execute tasks
        while len(tasks):

            done, tasks = await asyncio.wait(tasks,
                                             timeout=TIMEOUT,
                                             return_when=asyncio.ALL_COMPLETED)

            # Remove dones from remaining
            done_names = [d.get_name() for d in done]
            remaining = [
                c for c in remaining if make_name(c) not in done_names
            ]

            # Update results and checkpoint
            results.update(done)

            # Checkpoint the results
            if post_process:

                output = post_process(
                    results=results,
                    DATAPATH=DATAPATH,
                    REUSE=REUSE,
                    LAST_RUN=False,
                    OP_FILENAME=OP_FILENAME,
                )

                if not output.empty:
                    REUSE = False  # for second run onwards

            else:
                output = results

            if TIMEOUT:

                if remaining:

                    if SHOW_TQDM:
                        pbar.update(len(done))

                    else:
                        logging.info(
                            f"\nDone {algo.__name__} for {done_names[:2]} {len(results)} out of {len(cts)}. Pending {[make_name(c) for c in remaining][:2]}"
                        )

                # something wrong. Task is not progressing
                if (len(tasks) == last_len_tasks) & (len(tasks) > 0):
                    logging.warning(
                        f"\n @ ALERT @: Tasks failing. Pending {len(tasks)} tasks such as {[t.get_name() for t in tasks][:3]}... will be killed in 5 seconds !\n"
                    )
                    dn, pend = await asyncio.wait(tasks, timeout=5.0)
                    if len(dn) > 0:
                        results.update(dn)

                    tasks.difference_update(dn)
                    tasks.difference_update(pend)

                    [t.cancel() for t in tasks]

                    pend_names = [p.get_name() for p in pend]
                    # remove pending from remaining
                    remaining = [
                        c for c in remaining if make_name(c) not in pend_names
                    ]

                # re-initialize last length of tasks
                last_len_tasks = len(tasks)

    # Make the final output, based on REUSE status

    if OP_FILENAME:
        df = post_process(
            results=set(),  # Empty dataset
            DATAPATH=DATAPATH,
            REUSE=REUSE,
            LAST_RUN=True,
            OP_FILENAME=OP_FILENAME,
        )
    else:
        df = output

    if SHOW_TQDM:

        pbar.update(len(done))
        pbar.refresh()
        pbar.close()

    return df
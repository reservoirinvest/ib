#%%
# Determine status - load imports

import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from pprint import pprint
from typing import Union

import numpy as np
import pandas as pd
from ib_async import Contract, Option, Order, util
from loguru import logger
from pyprojroot import here
from tqdm import tqdm

from build import (
    atm_margin,
    calculate_atm_margin,
    chains_n_unds,
    delete_pkl_files,
    do_i_refresh,
    get_dte,
    get_ib_connection,
    get_option_chains,
    get_pickle,
    get_prec,
    get_prices_snapshot,
    get_qualified_symbols,
    get_volatilities_snapshot,
    how_many_days_old,
    load_config,
    pickle_me,
    volatilities,
)

#%%
# Config and constants

ROOT = here()

ACTIVESTATUS = os.getenv("ACTIVESTATUS", "").split(",")

pd.set_option('display.max_columns', None)

@dataclass
class OpenOrder:
    conId: int = 0
    symbol: str = "Dummy"
    secType: str = "STK"
    expiry: datetime = datetime.now()
    strike: float = 0.0
    right: str = None
    orderId: int = 0
    contract: Contract = None
    order: Order = None
    permId: int = 0
    action: str = "SELL"
    qty: float = 0.0
    lmtPrice: float = 0.0
    status: str = None

    def empty(self):
        return pd.DataFrame([self.__dict__]).iloc[0:0]

#%%
# Utility functions
def clean_ib_util_df(
    contracts: Union[list, pd.Series],
    eod=True,
    ist=False,
) -> Union[pd.DataFrame, None]:
    """Cleans ib_async's util.df to keep only relevant columns"""
    if isinstance(contracts, pd.Series):
        ct = contracts.to_list()
    elif not isinstance(contracts, list):
        logger.error(
            f"Invalid type for contracts: {type(contracts)}. Must be list or pd.Series."
        )
        return None
    else:
        ct = contracts

    try:
        udf = util.df(ct)
    except (AttributeError, ValueError) as e:
        logger.error(f"Error creating DataFrame from contracts: {e}")
        return None

    if udf is None or udf.empty:
        return None

    udf = udf[
        [
            "symbol",
            "conId",
            "secType",
            "lastTradeDateOrContractMonth",
            "strike",
            "right",
        ]
    ]
    udf.rename(columns={"lastTradeDateOrContractMonth": "expiry"}, inplace=True)

    if len(udf.expiry.iloc[0]) != 0:
        udf["expiry"] = udf["expiry"].apply(util.formatIBDatetime)
    else:
        udf["expiry"] = pd.NaT

    udf["contract"] = ct
    return udf

#%%
# Functions to get financials, portforlio and orders
def get_ib_portfolio(account: str) -> pd.DataFrame:
    """
    Get the IB portfolio for the specified account and return it as a DataFrame.
    
    Args:
        account: The account code (e.g., from .env US_ACCOUNT or SG_ACCOUNT)
    
    Returns:
        DataFrame with portfolio items
    """
    if not account:
        raise ValueError(f"Account '{account}' not found")
    
    ib = get_ib_connection('SNP', account=account)
    try:
        portfolio_items = ib.portfolio()
        upf = util.df(portfolio_items)
        contract_df = util.df(list(upf.contract)).iloc[:, :6]
        upf = contract_df.join(upf.drop(columns=["account", "contract"]))
        
        upf = upf.rename(
            columns={
                "lastTradeDateOrContractMonth": "expiry",
                "marketPrice": "mktPrice",
                "marketValue": "mktVal",
                "averageCost": "avgCost",
                "unrealizedPNL": "unPnL",
                "realizedPNL": "rePnL",
            }
        )
        
        df_pf = upf.drop_duplicates(keep="last")
        return df_pf
    
    finally:
        if ib and ib.isConnected():
            ib.disconnect()
            print("Disconnected from IB\n")



def get_financials(account: str = "") -> dict:
    """
    Get account financial values for the specified account or all consolidated accounts.
    
    Args:
        account: The account code (e.g., from .env US_ACCOUNT or SG_ACCOUNT).
                 If empty or None, returns aggregated values for all accounts.
    
    Returns:
        dict: Current net liquidation value, cash, margins, and excess liquidity with values rounded to 2 decimals
    """
    ib = get_ib_connection('SNP', account=account or None)
    try:
        if account:
            # Fetch values for specific account
            df_acc = util.df(ib.accountValues(account=account))
        else:
            # Fetch and aggregate values for all accounts
            df_acc = util.df(ib.accountValues())  # No account specified
            if not df_acc.empty:
                # Aggregate numeric values by tag, assuming sum for financial metrics
                df_acc = df_acc.groupby('tag').agg({
                    'value': lambda x: pd.to_numeric(x, errors='coerce').sum(),
                    'currency': 'first'  # Keep first currency (assumes same currency)
                }).reset_index()

        d_map = {
            "NetLiquidation": "net liquidation value",
            "StockMarketValue": "stocks",
            "TotalCashBalance": "cash",
            "Cushion": "cushion",
            "InitMarginReq": "initial margin",
            "MaintMarginReq": "maintenance margin",
            "UnrealizedPnL": "unrealized pnl",
            "RealizedPnL": "realized pnl",
            "LookAheadAvailableFunds": "funds available to trade",
            "ExcessLiquidity": "excess liquidity",
        }

        # Filter and set tag as categorical to match d_map order
        df_out = df_acc[df_acc['tag'].isin(d_map.keys())].copy()
        
        acc = df_out.set_index("tag")['value'].apply(float).to_dict()
        
        # Calculate Cushion as ExcessLiquidity / NetLiquidation
        net_liquidation = acc.get("NetLiquidation", 0)
        excess_liquidity = acc.get("ExcessLiquidity", 0)
        acc["Cushion"] = (excess_liquidity / net_liquidation) if net_liquidation != 0 else 0

        acc = {d_map.get(k): round(v, 2) for k, v in acc.items()}

        return acc
    
    finally:
        if ib and ib.isConnected():
            ib.disconnect()
            print("Disconnected from IB\n")


def get_open_orders(account: str = "", is_active: bool = False) -> pd.DataFrame:
    """
    Get open orders for the specified account.
    
    Args:
        account: The account code (e.g., from .env US_ACCOUNT or SG_ACCOUNT).
                 If empty or None, returns orders labeled with 'ALL'.
        is_active: If True, only return orders with active status
    
    Returns:
        DataFrame with open order details, including account and order columns
    """
    ib = get_ib_connection('SNP', account=account or None)
    try:
        trades = ib.reqAllOpenOrders()  # Fetch all open orders
        dfo = OpenOrder().empty()

        if trades:
            all_trades_df = (
                clean_ib_util_df([t.contract for t in trades])
                .join(util.df(t.orderStatus for t in trades))
                .join(util.df(t.order for t in trades), lsuffix="_")
            )
            
            # Filter by account if provided
            if account:
                all_trades_df['account'] = pd.Series([t.order.account for t in trades])
                all_trades_df = all_trades_df[all_trades_df['account'] == account]
            
            # Add account and order columns just before creating dfo
            account_name = 'ALL' if not account else account
            all_trades_df['account'] = account_name  # Assign account or 'ALL'
            order = pd.Series([t.order for t in trades], name="order")[all_trades_df.index]
            all_trades_df = all_trades_df.assign(order=order)
            
            all_trades_df.rename(
                {"lastTradeDateOrContractMonth": "expiry",
                 "totalQuantity": "qty"}, axis="columns", inplace=True
            )
            
            if "symbol" not in all_trades_df.columns:
                if "contract" in all_trades_df.columns:
                    all_trades_df["symbol"] = all_trades_df["contract"].apply(
                        lambda x: x.symbol
                    )
                else:
                    raise ValueError(
                        "Neither 'symbol' nor 'contract' column found in the DataFrame"
                    )
            
            # Move account column to first position
            cols = ['account'] + [col for col in all_trades_df.columns if col != 'account']
            all_trades_df = all_trades_df[cols]
            
            dfo = all_trades_df[dfo.columns]
            
            if is_active:
                dfo = dfo[dfo.status.isin(ACTIVESTATUS)]
        
        if "state" not in dfo.columns:
            dfo = dfo.assign(state="unknown")
        
        return dfo
    
    except Exception as e:
        print(f"Error fetching open orders for {account or 'ALL'}: {str(e)}")
        return OpenOrder().empty()  # Return empty DataFrame on error
    
    finally:
        if ib and ib.isConnected():
            ib.disconnect()
            print("Disconnected from IB\n")

#%%
# Functions to classify portfolios, open orders and update unds

def classify_pf(pf):
    """
    Classifies trading strategies in a portfolio based on option and stock positions.

    Parameters:
    pf (pd.DataFrame): Portfolio DataFrame containing columns:
        - symbol: Ticker symbol
        - secType: Security type ('STK' or 'OPT')
        - right: Option right ('C', 'P', or '0' for stocks)
        - expiry: Option expiration date
        - strike: Option strike price
        - position: Position size (positive or negative)

    Returns:
    pd.DataFrame: Original DataFrame with added 'state' column containing classifications
    """
    # Create a copy to avoid modifying the original DataFrame
    pf = pf.copy()
    
    # Add dte column for options
    if 'expiry' in pf.columns and 'dte' not in pf.columns:
        pf['dte'] = pf.expiry.apply(lambda x: get_dte(x) if pd.notnull(x) else None)
        
    pf["state"] = "tbd"

    # First, classify all options
    option_mask = pf.secType == "OPT"
    
    # Classify protecting options (long calls or long puts)
    protecting_mask = option_mask & (
        ((pf.right == "C") & (pf.position > 0)) |  # Long call
        ((pf.right == "P") & (pf.position > 0))   # Long put
    )
    pf.loc[protecting_mask, "state"] = "protecting"
    
    # Classify sowed options (short options that are not part of a spread)
    sowed_mask = option_mask & (pf.position < 0)  # All short options
    pf.loc[sowed_mask, "state"] = "sowed"
    
    # Now classify covering options (short calls that are part of a spread)
    # These will override the 'sowed' classification
    covering_mask = option_mask & (pf.position < 0) & (
        ((pf.right == "C") | (pf.right == "P"))  # Short call or put
    )
    # Only mark as covering if there's a corresponding long position
    has_long = pf[pf.position > 0].groupby('symbol').size()
    covering_mask = covering_mask & pf.symbol.isin(has_long.index)
    pf.loc[covering_mask, "state"] = "covering"
    
    # Now classify stocks based on their options
    stock_mask = pf.secType == "STK"
    
    # Get symbols with protecting and covering options
    symbols_with_protecting = set(pf[pf.state == "protecting"].symbol.unique())
    symbols_with_covering = set(pf[pf.state == "covering"].symbol.unique())
    
    # Classify stocks
    pf.loc[stock_mask & 
           pf.symbol.isin(symbols_with_protecting) & 
           ~pf.symbol.isin(symbols_with_covering), "state"] = "uncovered"
           
    pf.loc[stock_mask & 
           ~pf.symbol.isin(symbols_with_protecting) & 
           pf.symbol.isin(symbols_with_covering), "state"] = "unprotected"
           
    pf.loc[stock_mask & 
           pf.symbol.isin(symbols_with_protecting) & 
           pf.symbol.isin(symbols_with_covering), "state"] = "zen"
           
    pf.loc[stock_mask & 
           (pf.state == "tbd") & 
           (pf.position != 0), "state"] = "exposed"

    # Classify orphaned options (long options without corresponding stock)
    # Get symbols that have stock positions using the existing stock_mask
    has_stock = set(pf[stock_mask].symbol.unique())

    # Mark as orphaned if:
    # 1. It's an option
    # 2. It's a long position
    # 3. The symbol doesn't have any stock position
    pf.loc[
        option_mask & 
        (pf.position > 0) & 
        ~pf.symbol.isin(has_stock),
        "state"
    ] = "orphaned"
    
    # For any remaining unclassified positions
    pf.loc[pf.state == "tbd", "state"] = "unclassified"
    
    return pf


def classify_open_orders(df_openords, pf):
    """
    Classify open orders based on their characteristics and portfolio context.

    Parameters:
    df_openords (pd.DataFrame): DataFrame of open orders
    pf (pd.DataFrame): Portfolio DataFrame

    Returns:
    pd.DataFrame: Open orders DataFrame with added 'state' column
    """
    if df_openords is None or df_openords.empty:
        return df_openords

    # Create a copy to avoid modifying the original DataFrame
    df = df_openords.copy()

    # Initialize status column
    df["state"] = "unclassified"

    # Identify option orders
    opt_orders = df[df.secType == "OPT"]

    # 'covering' - option SELL order with underlying stock position
    covering_mask = (opt_orders.action == "SELL") & (
        # Call option with positive stock position
        (
            (opt_orders.right == "C")
            & (
                opt_orders.symbol.isin(
                    pf[(pf.secType == "STK") & (pf.position > 0)].symbol
                )
            )
        )
        |
        # Put option with negative stock position
        (
            (opt_orders.right == "P")
            & (
                opt_orders.symbol.isin(
                    pf[(pf.secType == "STK") & (pf.position < 0)].symbol
                )
            )
        )
    )

    df.loc[covering_mask[covering_mask].index, "state"] = "covering"

    # 'protecting' - option BUY order with underlying stock position
    protecting_mask = (
        (opt_orders.action == "BUY")
        & (
            # Put option protecting long stock position
            ((opt_orders.right == "P") & (opt_orders.symbol.isin(pf[(pf.secType == "STK") & (pf.position > 0)].symbol)))
            |
            # Call option protecting short stock position
            ((opt_orders.right == "C") & (opt_orders.symbol.isin(pf[(pf.secType == "STK") & (pf.position < 0)].symbol)))
        )
    )
    df.loc[protecting_mask[protecting_mask].index, "state"] = "protecting"

    # 'sowing' - option SELL order without underlying stock position
    sowing_mask = (opt_orders.action == "SELL") & (
        ~opt_orders.symbol.isin(pf[(pf.secType == "STK")].symbol)
    )
    df.loc[sowing_mask[sowing_mask].index, "state"] = "sowing"

    # 'reaping' - option BUY order with matching existing option position
    reaping_mask = opt_orders.apply(
        lambda row: (
            row.action == "BUY"
            and not pf[
                (pf.secType == "OPT")
                & (pf.symbol == row.symbol)
                & (pf.right == row.right)
                & (pf.strike == row.strike)
            ].empty
        ),
        axis=1,
    )
    df.loc[reaping_mask[reaping_mask].index, "state"] = "reaping"

    # 'de-orphaning' - option SELL order with matching existing option position
    de_orphaning_mask = opt_orders.apply(
        lambda row: (
            row.action == "SELL"
            and not pf[
                (pf.secType == "OPT")
                & (pf.symbol == row.symbol)
                & (pf.right == row.right)
                & (pf.strike == row.strike)
            ].empty
        ),
        axis=1,
    )
    df.loc[de_orphaning_mask[de_orphaning_mask].index, "state"] = "de-orphaning"

    # 'straddling' - two option BUY orders for same symbol not in portfolio
    # Group by symbol and count BUY actions
    straddle_symbols = (
        opt_orders[(opt_orders.action == "BUY")]
        .groupby("symbol")
        .filter(lambda x: len(x) >= 2)["symbol"]
        .unique()
    )

    straddle_mask = (
        (opt_orders.action == "BUY")
        & (opt_orders.symbol.isin(straddle_symbols))
        & (~opt_orders.symbol.isin(pf.symbol))
    )
    df.loc[straddle_mask[straddle_mask].index, "state"] = "straddling"

    return df


def update_unds_status(df_unds:pd.DataFrame, 
                    df_pf:pd.DataFrame, 
                    df_openords: pd.DataFrame) -> pd.DataFrame:
    """
    Update underlying symbols status based on portfolio and open orders.

    Parameters:
    df_unds (pd.DataFrame): Underlying symbols DataFrame
    df_pf (pd.DataFrame): Portfolio DataFrame

    Returns:
    pd.DataFrame: Updated underlying symbols DataFrame with 'state' column
    """
    df_unds = df_unds.drop(columns=['mktPrice', 'state', ], errors='ignore').merge(
        df_pf[df_pf["secType"] == "STK"][["symbol", "mktPrice", 'state']],
        on="symbol",
        how="left",
        suffixes=("", "_new"),
    )

    # update status from df_pf for stock symbols
    stk_symbols = df_pf[df_pf.secType == "STK"].symbol
    stk_state_dict = dict(
        zip(
            df_pf.loc[df_pf.secType == "STK", "symbol"],
            df_pf.loc[df_pf.secType == "STK", "state"],
        )
    )

    df_unds.loc[df_unds.symbol.isin(stk_symbols), "state"] = \
            df_unds.loc[df_unds.symbol.isin(stk_symbols)].symbol.map(stk_state_dict)

    # ..update status for symbols not in df_pf
    df_unds.loc[~df_unds.symbol.isin(df_pf.symbol.unique()), "state"] = "virgin"

    # Zen conditions
    zen_symbols = set()

    # 1. Symbols with both covering and protecting positions are zen
    for symbol, group in df_openords.groupby("symbol"):
        if len(group) == 2 and {"covering", "protecting"}.issubset(set(group.state)):
            zen_symbols.add(symbol)
        else:
            group = df_pf[df_pf.symbol == symbol]
            if len(group) == 2 and {"covering", "protecting"}.issubset(
                set(group.state)
            ):
                zen_symbols.add(symbol)

    # 2. Symbols with 'straddled' portfolio state
    straddled_symbols = df_pf[df_pf.state == "straddled"].symbol
    zen_symbols.update(straddled_symbols)

    # 3. Symbols with short 'sowing' order
    sowing_symbols = df_openords[df_openords.state == "sowing"].symbol
    zen_symbols.update(sowing_symbols)

    # 4. Unprotected with protecting order
    unprotected_with_protect = df_pf[
        (df_pf.state == "unprotected")
        & df_pf.symbol.isin(df_openords[df_openords.state == "protecting"].symbol)
    ].symbol
    zen_symbols.update(unprotected_with_protect)

    # 5. Uncovered with covering order
    uncovered_with_cover = df_pf[
        (df_pf.state == "uncovered")
        & df_pf.symbol.isin(df_openords[df_openords.state == "covering"].symbol)
    ].symbol
    zen_symbols.update(uncovered_with_cover)

    # 6. Long 'orphaned' position with 'de-orphaning' order
    orphaned_with_deorphan = df_pf[
        (df_pf.state == "orphaned")
        & df_pf.symbol.isin(df_openords[df_openords.state == "de-orphaning"].symbol)
    ].symbol
    zen_symbols.update(orphaned_with_deorphan)

    # 7. Short 'sowed' position with 'reaping' order
    sowed_with_reap = df_pf[
        (df_pf.state == "sowed")
        & df_pf.symbol.isin(df_openords[df_openords.state == "reaping"].symbol)
    ].symbol
    zen_symbols.update(sowed_with_reap)

    # 8. Short 'orphaned' position with 'virgin' order
    orphaned_with_virgin = df_pf[
        (df_pf.state == "orphaned")
        & ~df_pf.symbol.isin(df_openords[df_openords.state == "virgin"].symbol)
    ].symbol
    zen_symbols.update(orphaned_with_virgin)

    # Update status for zen symbols
    df_unds.loc[df_unds.symbol.isin(zen_symbols), "state"] = "zen"

    # Unreaped: Symbol has a short option position with no open 'reaping' order
    unreaped_symbols = df_pf[
        (df_pf.state == "sowed")
        & ~df_pf.symbol.isin(df_openords[df_openords.state == "reaping"].symbol)
    ].symbol

    # Update status for unreaped symbols
    df_unds.loc[df_unds.symbol.isin(unreaped_symbols), "state"] = "unreaped"

    # Unprotected: Symbol has an exposed state with only one 'covering' order
    unprotected_symbols = []
    for symbol in df_pf[df_pf.state == "unprotected"].symbol:
        openord_group = df_openords[df_openords.symbol == symbol]
        if len(openord_group) == 1 and openord_group.iloc[0].state == "covering":
            unprotected_symbols.append(symbol)

    # Update status for unprotected symbols
    df_unds.loc[df_unds.symbol.isin(unprotected_symbols), "state"] = "unprotected"

    # Uncovered: Symbol has an exposed state with only one 'protecting' order
    uncovered_symbols = []
    for symbol in df_unds[df_unds.state == "exposed"].symbol:
        openord_group = df_openords[df_openords.symbol == symbol]
        if len(openord_group) == 1 and openord_group.iloc[0].state == "protecting":
            uncovered_symbols.append(symbol)

    # Update status for uncovered symbols
    df_unds.loc[df_unds.symbol.isin(uncovered_symbols), "state"] = "uncovered"

    # Orphaned: Symbol has an 'orphaned' state with no open orders
    orphaned_symbols = df_pf[(df_pf.state == "orphaned") & ~df_pf.symbol.isin(df_openords.symbol)].symbol

    # Update status for orphaned symbols
    df_unds.loc[df_unds.symbol.isin(orphaned_symbols), "state"] = "orphaned"

    # Classify short stock positions without covering/protecting options as 'exposed'
    # Get all short stock positions from portfolio
    short_stocks = df_pf[(df_pf.secType == 'STK') & (df_pf.position < 0)]['symbol']
    
    # Find short stocks that don't have covering or protecting options
    exposed_short_stocks = []
    for symbol in short_stocks:
        # Check if there are any covering or protecting options in portfolio or open orders
        has_covering = (df_pf.symbol == symbol) & (df_pf.state == 'covering')
        has_protecting = (df_pf.symbol == symbol) & (df_pf.state == 'protecting')
        has_covering_orders = (df_openords.symbol == symbol) & (df_openords.state == 'covering')
        has_protecting_orders = (df_openords.symbol == symbol) & (df_openords.state == 'protecting')
        
        if not (has_covering.any() or has_protecting.any() or 
                has_covering_orders.any() or has_protecting_orders.any()):
            exposed_short_stocks.append(symbol)
    
    # Update status for exposed short stocks
    df_unds.loc[df_unds.symbol.isin(exposed_short_stocks), "state"] = "exposed"

    return df_unds

#%%
# Test Functions
if __name__ == "__main__":

    if do_i_refresh(my_path=ROOT / 'data' / 'df_unds.pkl', max_days=1):
        df_chains, df_unds = chains_n_unds()
    else:
        df_chains= get_pickle(path=ROOT / 'data' / 'df_chains.pkl')
        df_unds= get_pickle(path=ROOT / 'data' / 'df_unds.pkl')

    #%%
    # Get financials for US, SG and consolidated accounts
    us_fin = get_financials(account=os.getenv("US_ACCOUNT", ""))
    sg_fin = get_financials(account=os.getenv("SG_ACCOUNT", ""))
    fin = get_financials()

    print("\nConsolidated Financials:")
    pprint(fin)
    print("\nUS Account Financials:")
    pprint(us_fin)
    print("\nSG Account Financials:")
    pprint(sg_fin)

    #%%
    # Get portfolio and open orders for US and SG accounts
    ACCOUNT = "US_ACCOUNT"  # Change to US_ACCOUNT or SG_ACCOUNT as needed
    print(f"\nUsing account: {ACCOUNT}\n")

    df_pf = get_ib_portfolio(account=os.getenv(ACCOUNT, ""))
    df_pf = classify_pf(df_pf)
    print('df_pf\n')
    print(f"{df_pf.to_string()}\n")

    df_openords = get_open_orders(account=os.getenv(ACCOUNT, ""), is_active=True)
    df_openords = classify_open_orders(df_openords, df_pf)

    print('\ndf_openords\n')
    print(f"{df_openords.drop(columns=['contract', 'order']).to_string()}\n")

    df_unds = update_unds_status(df_unds, df_pf, df_openords)
    print(f"\ndf_unds {df_unds.head().to_string()}")

# %%

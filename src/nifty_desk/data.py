"""Data layer for the NIFTY Research Desk.

Two feeds:

* Index history (^NSEI, ^NSEBANK) via yfinance, cached as plain CSVs
  (``data/nsei_daily.csv`` / ``data/nsebank_daily.csv`` in the repo) with
  Date,Open,High,Low,Close,Volume columns. A cache younger than
  ``max_age_hours`` is used as-is; anything older (or missing) triggers a
  fresh download and an atomic cache rewrite.
* Option chains from the official NSE F&O bhavcopy, parsed by the vendored
  greeks-lab engine (``engines.chain.load_bhavcopy`` / ``extract_chain``).

No data is ever fabricated: if yfinance is unreachable and no cache exists,
``load_index_history`` raises.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from .engines import chain as _chain

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"

_TICKERS = {
    "NIFTY": ("^NSEI", DATA_DIR / "nsei_daily.csv"),
    "BANKNIFTY": ("^NSEBANK", DATA_DIR / "nsebank_daily.csv"),
}
HISTORY_START = "2015-01-01"


def _cache_fresh(path: Path, max_age_hours: float) -> bool:
    return path.exists() and (time.time() - path.stat().st_mtime) < max_age_hours * 3600


def _fetch_yahoo(ticker: str) -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download(ticker, start=HISTORY_START, auto_adjust=False,
                      progress=False, threads=False)
    if raw is None or raw.empty:
        raise RuntimeError(f"yfinance returned no data for {ticker}")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    out = raw.reset_index()[["Date", "Open", "High", "Low", "Close", "Volume"]]
    out["Date"] = pd.to_datetime(out["Date"]).dt.tz_localize(None)
    return out.dropna(subset=["Close"]).sort_values("Date").reset_index(drop=True)


def load_index_history(symbol: str = "NIFTY", max_age_hours: float = 12.0,
                       force_refresh: bool = False) -> pd.DataFrame:
    """Daily OHLCV for ``symbol`` ('NIFTY' or 'BANKNIFTY') from 2015-01-01.

    Serves the on-disk cache when it is fresh; otherwise downloads via
    yfinance and rewrites the cache.
    """
    if symbol not in _TICKERS:
        raise ValueError(f"unknown symbol {symbol!r}; use one of {list(_TICKERS)}")
    ticker, cache_path = _TICKERS[symbol]
    if not force_refresh and _cache_fresh(cache_path, max_age_hours):
        df = pd.read_csv(cache_path, parse_dates=["Date"])
        return df.sort_values("Date").reset_index(drop=True)
    df = _fetch_yahoo(ticker)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(".csv.tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(cache_path)
    return df


def fetch_all(max_age_hours: float = 12.0, force_refresh: bool = False) -> dict[str, pd.DataFrame]:
    """Both index histories, keyed 'NIFTY' / 'BANKNIFTY'."""
    return {sym: load_index_history(sym, max_age_hours, force_refresh)
            for sym in _TICKERS}


def load_chain(source, symbol: str = "NIFTY", expiry=None, min_days: int = 1) -> pd.DataFrame:
    """Load one option chain from a bhavcopy path/zip via the vendored parser.

    ``expiry=None`` selects the nearest expiry with at least ``min_days``
    left (the greeks-lab headline-chain convention: the expiring series on
    expiry day itself is never used).
    """
    raw = _chain.load_bhavcopy(source)
    return _chain.extract_chain(raw, symbol=symbol, expiry=expiry,
                                min_days=min_days)

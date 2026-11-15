"""Fetch S&P 500 constituent returns for the last 5 years via polars-yahoo."""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "polars-yahoo>=0.2.0",
#     "pandas",
#     "numpy",
#     "lxml",
#     "requests",
#     "pyarrow",
# ]
# ///

import io
from pathlib import Path

import pandas as pd
import polars_yahoo as yh
import requests

# ── 1. S&P 500 tickers from Wikipedia ─────────────────────────────────────────

resp = requests.get(
    "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    headers={"User-Agent": "Mozilla/5.0 (research script)"},
    timeout=30,
)
resp.raise_for_status()
tables = pd.read_html(io.StringIO(resp.text))
tickers = tables[0]["Symbol"].str.replace(".", "-", regex=False).tolist()
print(f"Found {len(tickers)} tickers in the S&P 500 index")

# ── 2. Download 5 years of adjusted close prices ──────────────────────────────

# adj_close is Yahoo's split- and dividend-adjusted close, the same series
# yfinance returned as "Close" under auto_adjust=True. Symbols Yahoo cannot
# serve are skipped with a warning; the missing-data filter below would have
# dropped them anyway.
bars = yh.download(tickers, period=None, start="2021-06-01", end="2026-06-01")
raw = (
    bars.pivot(on="ticker", index="date", values="adj_close")
    .sort("date")
    .to_pandas()
    .set_index("date")
    .sort_index(axis=1)
)
raw.index = pd.DatetimeIndex(raw.index, name="Date")

print(f"\nRaw download: {raw.shape[0]} trading days x {raw.shape[1]} tickers")

# ── 3. Clean: keep tickers with <5% missing days, forward-fill short gaps ─────

threshold = 0.05
missing_frac = raw.isna().mean()
keep = missing_frac[missing_frac <= threshold].index
raw = raw[keep].ffill()

print(f"After cleaning: {raw.shape[0]} days x {raw.shape[1]} assets")

# ── 4. Returns ────────────────────────────────────────────────────────────────

pct_returns = raw.pct_change().dropna()
print(f"Return matrix shape: {pct_returns.shape}  (T={pct_returns.shape[0]}, N={pct_returns.shape[1]})")

# ── 5. Save ───────────────────────────────────────────────────────────────────

out_dir = Path(__file__).resolve().parents[1] / "data"

pct_returns.to_parquet(out_dir / "sp500_pct_returns.parquet")
print("\nSaved data/sp500_pct_returns.parquet")

print(f"Date range: {pct_returns.index[0].date()} → {pct_returns.index[-1].date()}")
print(f"Assets: {pct_returns.shape[1]}")
print(f"Trading days: {pct_returns.shape[0]}")

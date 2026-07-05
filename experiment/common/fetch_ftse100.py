"""Fetch FTSE 100 daily returns via yfinance and save to parquet."""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "yfinance",
#     "pandas",
#     "pyarrow",
# ]
# ///

import yfinance as yf

# FTSE 100 constituents as of 2025 (with .L suffix for LSE)
# Source: Wikipedia / FTSE Russell June 2025
FTSE100_TICKERS = [
    "AAF.L",
    "AAL.L",
    "ABF.L",
    "ADM.L",
    "AHT.L",
    "AIRP.L",
    "ALC.L",
    "ANTO.L",
    "AO.L",
    "APA.L",
    "AUTO.L",
    "AV.L",
    "AZN.L",
    "BA.L",
    "BARC.L",
    "BATS.L",
    "BBOX.L",
    "BEZ.L",
    "BKG.L",
    "BLND.L",
    "BME.L",
    "BNZL.L",
    "BP.L",
    "BSRT.L",
    "BT.L",
    "CLLN.L",
    "CPG.L",
    "CRH.L",
    "CRDA.L",
    "DCC.L",
    "DGE.L",
    "DPLM.L",
    "EDV.L",
    "ELN.L",
    "ENT.L",
    "EXPN.L",
    "FCIT.L",
    "FERG.L",
    "FLTR.L",
    "FRES.L",
    "GLEN.L",
    "GSK.L",
    "HIK.L",
    "HL.L",
    "HLMA.L",
    "HLN.L",
    "HSBA.L",
    "ICP.L",
    "IGG.L",
    "III.L",
    "IMB.L",
    "INF.L",
    "ITV.L",
    "JD.L",
    "KGF.L",
    "LAND.L",
    "LGEN.L",
    "LLOY.L",
    "LMP.L",
    "LSEG.L",
    "MKS.L",
    "MNDI.L",
    "MNG.L",
    "MRO.L",
    "NG.L",
    "NXT.L",
    "OCL.L",
    "PCF.L",
    "PHNX.L",
    "PRU.L",
    "PSH.L",
    "PSN.L",
    "PSON.L",
    "RB.L",
    "RDSA.L",
    "REL.L",
    "RIO.L",
    "RKT.L",
    "RMV.L",
    "RR.L",
    "RS1.L",
    "RSA.L",
    "SBRY.L",
    "SDR.L",
    "SGE.L",
    "SGRO.L",
    "SJP.L",
    "SKG.L",
    "SMDS.L",
    "SMIN.L",
    "SMT.L",
    "SN.L",
    "SPX.L",
    "SSE.L",
    "STAN.L",
    "STJ.L",
    "SVT.L",
    "TSCO.L",
    "TW.L",
    "UU.L",
    "VOD.L",
    "WEIR.L",
    "WMH.L",
    "WPP.L",
    "WTB.L",
]

START = "2019-06-01"
END = "2026-05-30"
MAX_MISSING_FRAC = 0.10

print(f"Downloading {len(FTSE100_TICKERS)} tickers from {START} to {END}...")
raw = yf.download(
    FTSE100_TICKERS,
    start=START,
    end=END,
    auto_adjust=True,
    progress=True,
)["Close"]

print(f"\nRaw shape: {raw.shape}")

# Drop tickers with too much missing data
missing_frac = raw.isna().mean()
keep = missing_frac[missing_frac <= MAX_MISSING_FRAC].index
dropped = set(raw.columns) - set(keep)
if dropped:
    print(f"Dropping {len(dropped)} tickers with >{MAX_MISSING_FRAC * 100:.0f}% missing: {sorted(dropped)}")
raw = raw[keep]

# Forward-fill residual gaps (delisted / bank-holiday gaps)
raw = raw.ffill()

# Drop any remaining rows with NaN (leading rows before first listing)
raw = raw.dropna()

# Compute percentage returns
returns = raw.pct_change().dropna()

print(f"\nFinal returns shape: {returns.shape}")
print(f"Date range: {returns.index[0].date()} to {returns.index[-1].date()}")
print(f"n={returns.shape[1]} assets, T={returns.shape[0]} days, n/T={returns.shape[1] / returns.shape[0]:.3f}")

# Remove ticker suffix for cleaner column names
returns.columns = [t.replace(".L", "") for t in returns.columns]

out = "data/ftse100_pct_returns.parquet"
returns.to_parquet(out)
print(f"\nSaved to {out}")

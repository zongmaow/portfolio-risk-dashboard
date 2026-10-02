"""Market data access via yfinance.

Kept free of Streamlit so it can be reused in scripts/tests;
the app layer adds caching on top.
"""

from pathlib import Path

import pandas as pd
import yfinance as yf

# Frozen sample: the price snapshot the README and papers report on.
# Adjusted closes for the default portfolio plus SPY, 2022-01-03 to
# 2026-10-01, downloaded once from Yahoo Finance and committed to the
# repo so every reported number is exactly reproducible.
FROZEN_PRICES_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "frozen_prices.csv"
)
FROZEN_START = "2022-01-03"
FROZEN_END = "2026-10-01"

# Fallback sector map for the most common tickers, used when
# yfinance's info endpoint is slow or unavailable.
SECTOR_FALLBACK = {
    "AAPL": "Technology",
    "MSFT": "Technology",
    "NVDA": "Technology",
    "GOOGL": "Communication Services",
    "META": "Communication Services",
    "AMZN": "Consumer Discretionary",
    "TSLA": "Consumer Discretionary",
    "JPM": "Financial Services",
    "BAC": "Financial Services",
    "V": "Financial Services",
    "XOM": "Energy",
    "CVX": "Energy",
    "JNJ": "Healthcare",
    "UNH": "Healthcare",
    "LLY": "Healthcare",
    "PG": "Consumer Staples",
    "KO": "Consumer Staples",
}


def download_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Download adjusted-close prices. Returns a DataFrame (columns = tickers)."""
    tickers = [t.strip().upper() for t in tickers if t.strip()]
    if not tickers:
        raise ValueError("No tickers provided.")
    data = yf.download(
        tickers, start=start, end=end, auto_adjust=True, progress=False
    )
    closes = data["Close"] if "Close" in data else data
    if isinstance(closes, pd.Series):  # single ticker comes back as Series
        closes = closes.to_frame(tickers[0])
    closes.columns = [str(c).upper() for c in closes.columns]
    closes = closes.dropna(how="all").ffill().dropna()
    if closes.empty:
        raise ValueError(f"No price data downloaded for {tickers}.")
    return closes


def load_frozen_prices(tickers: list[str] | None = None) -> pd.DataFrame:
    """Load the frozen price snapshot instead of downloading live data.

    Returns adjusted closes for the requested tickers (all snapshot
    columns when ``tickers`` is None). Raises ValueError if a requested
    ticker is not in the snapshot.
    """
    prices = pd.read_csv(FROZEN_PRICES_PATH, index_col=0, parse_dates=True)
    prices.index.name = "Date"
    if tickers is not None:
        tickers = [t.strip().upper() for t in tickers if t.strip()]
        missing = [t for t in tickers if t not in prices.columns]
        if missing:
            raise ValueError(f"Not in the frozen snapshot: {missing}.")
        prices = prices[tickers]
    return prices


def get_sectors(tickers: list[str]) -> dict[str, str]:
    """Best-effort GICS sector lookup per ticker."""
    sectors: dict[str, str] = {}
    for t in tickers:
        t = t.strip().upper()
        sector = SECTOR_FALLBACK.get(t)
        if sector is None:
            try:
                info = yf.Ticker(t).info
                sector = info.get("sector") or "Unknown"
            except Exception:
                sector = "Unknown"
        sectors[t] = sector
    return sectors


def sector_exposure(
    weights: pd.Series, sectors: dict[str, str]
) -> pd.DataFrame:
    """Aggregate portfolio weights by sector."""
    w = weights / weights.sum()
    df = pd.DataFrame(
        {"weight": w, "sector": [sectors.get(t, "Unknown") for t in w.index]}
    )
    return df.groupby("sector")["weight"].sum().sort_values(ascending=False)

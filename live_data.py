"""
Live market data via yfinance.
"""
import numpy as np
import pandas as pd

try:
    import yfinance as yf
    _YF_AVAILABLE = True
except ImportError:
    _YF_AVAILABLE = False


def fetch(ticker: str, period: str = "1y") -> dict | None:
    if not _YF_AVAILABLE:
        return None
    try:
        t    = yf.Ticker(ticker)
        hist = t.history(period=period)
        if len(hist) < 5:
            return None

        close   = hist["Close"]
        S       = float(close.iloc[-1])
        S_prev  = float(close.iloc[-2])
        rets    = np.log(close / close.shift(1)).dropna()

        info = {}
        try:
            info = t.info or {}
        except Exception:
            pass

        return {
            "ticker":     ticker.upper(),
            "name":       info.get("longName", ticker.upper()),
            "price":      S,
            "change":     S - S_prev,
            "change_pct": (S - S_prev) / S_prev * 100,
            "sigma_1m":   float(rets.tail(21).std()  * np.sqrt(252)),
            "sigma_3m":   float(rets.tail(63).std()  * np.sqrt(252)),
            "sigma_1y":   float(rets.std()            * np.sqrt(252)),
            "high_52w":   float(close.tail(252).max()),
            "low_52w":    float(close.tail(252).min()),
            "beta":       info.get("beta"),
            "currency":   info.get("currency", "USD"),
            "hist":       hist,
            "returns":    rets,
        }
    except Exception:
        return None


def risk_free_rate() -> float:
    """Approximate US 3-month T-bill rate via ^IRX."""
    try:
        t = yf.Ticker("^IRX")
        h = t.history(period="5d")
        if len(h) > 0:
            return float(h["Close"].iloc[-1]) / 100
    except Exception:
        pass
    return 0.05

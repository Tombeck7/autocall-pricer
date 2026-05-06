"""
Live market data via yfinance.
Fetches: price, historical vol, implied vol from option chain, fundamentals.
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

        close  = hist["Close"]
        S      = float(close.iloc[-1])
        S_prev = float(close.iloc[-2])
        rets   = np.log(close / close.shift(1)).dropna()

        info = {}
        try:
            info = t.info or {}
        except Exception:
            pass

        # ── Implied vol from option chain ─────────────────────────────────────
        iv_chain = _get_atm_iv(t, S)

        # ── Fundamentals ──────────────────────────────────────────────────────
        market_cap = info.get("marketCap")
        if market_cap:
            if market_cap >= 1e12:
                mc_str = f"{market_cap/1e12:.2f}T"
            elif market_cap >= 1e9:
                mc_str = f"{market_cap/1e9:.2f}B"
            else:
                mc_str = f"{market_cap/1e6:.0f}M"
        else:
            mc_str = None

        return {
            # Identity
            "ticker":        ticker.upper(),
            "name":          info.get("longName", ticker.upper()),
            "sector":        info.get("sector"),
            "industry":      info.get("industry"),
            "currency":      info.get("currency", "USD"),
            "exchange":      info.get("exchange"),

            # Price
            "price":         S,
            "prev_close":    S_prev,
            "change":        S - S_prev,
            "change_pct":    (S - S_prev) / S_prev * 100,
            "open":          float(hist["Open"].iloc[-1]),
            "high_day":      float(hist["High"].iloc[-1]),
            "low_day":       float(hist["Low"].iloc[-1]),
            "volume":        float(hist["Volume"].iloc[-1]),
            "avg_volume":    info.get("averageVolume"),

            # Range
            "high_52w":      float(close.tail(252).max()),
            "low_52w":       float(close.tail(252).min()),

            # Historical vol
            "sigma_1m":      float(rets.tail(21).std()  * np.sqrt(252)),
            "sigma_3m":      float(rets.tail(63).std()  * np.sqrt(252)),
            "sigma_1y":      float(rets.std()            * np.sqrt(252)),

            # Implied vol (from option chain)
            "iv_chain":      iv_chain,   # dict: {expiry: {strike: iv, ...}, ...}
            "iv_atm_near":   iv_chain.get("atm_near"),
            "iv_atm_far":    iv_chain.get("atm_far"),

            # Fundamentals
            "pe_ratio":      info.get("trailingPE"),
            "fwd_pe":        info.get("forwardPE"),
            "pb_ratio":      info.get("priceToBook"),
            "eps":           info.get("trailingEps"),
            "div_yield":     info.get("dividendYield"),
            "beta":          info.get("beta"),
            "market_cap":    market_cap,
            "market_cap_str": mc_str,
            "target_price":  info.get("targetMeanPrice"),
            "analyst_rating": info.get("recommendationKey"),

            # Raw
            "hist":    hist,
            "returns": rets,
        }
    except Exception as e:
        return None


def _get_atm_iv(ticker_obj, S: float) -> dict:
    """Extract ATM IV from the two nearest option expirations."""
    result = {}
    try:
        expirations = ticker_obj.options
        if not expirations:
            return result

        for i, exp in enumerate(expirations[:3]):
            try:
                chain  = ticker_obj.option_chain(exp)
                calls  = chain.calls.dropna(subset=["impliedVolatility", "strike"])
                puts   = chain.puts.dropna(subset=["impliedVolatility", "strike"])

                # ATM = nearest strike to current spot
                if len(calls) == 0:
                    continue
                calls["dist"] = (calls["strike"] - S).abs()
                puts["dist"]  = (puts["strike"]  - S).abs()

                atm_call_iv = float(calls.nsmallest(1, "dist")["impliedVolatility"].iloc[0])
                atm_put_iv  = float(puts.nsmallest(1, "dist")["impliedVolatility"].iloc[0])
                atm_iv      = (atm_call_iv + atm_put_iv) / 2

                label = "atm_near" if i == 0 else f"atm_{exp}"
                if i == 0:
                    result["atm_near"] = atm_iv
                    result["exp_near"] = exp
                    result["smile_near"] = _build_smile(calls, puts)
                elif i == 1:
                    result["atm_far"]  = atm_iv
                    result["exp_far"]  = exp
                    result["smile_far"] = _build_smile(calls, puts)
            except Exception:
                continue
    except Exception:
        pass
    return result


def _build_smile(calls: pd.DataFrame, puts: pd.DataFrame) -> pd.DataFrame:
    """Merge call/put IVs into a smile DataFrame."""
    try:
        c = calls[["strike", "impliedVolatility", "volume", "openInterest"]].copy()
        p = puts[["strike", "impliedVolatility", "volume", "openInterest"]].copy()
        c.columns = ["strike", "call_iv", "call_vol", "call_oi"]
        p.columns = ["strike", "put_iv",  "put_vol",  "put_oi"]
        smile = c.merge(p, on="strike", how="outer").sort_values("strike")
        smile["mid_iv"] = smile[["call_iv", "put_iv"]].mean(axis=1)
        return smile.dropna(subset=["mid_iv"])
    except Exception:
        return pd.DataFrame()


def risk_free_rate() -> float:
    """US 3-month T-bill via ^IRX."""
    try:
        t = yf.Ticker("^IRX")
        h = t.history(period="5d")
        if len(h) > 0:
            return float(h["Close"].iloc[-1]) / 100
    except Exception:
        pass
    return 0.05

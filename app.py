from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ===== Single-file Nifty 50 Analytics & Forecast app (no src folder needed) =====
"""Data loading: live (Yahoo Finance) or the Kaggle Nifty-50 CSV."""



# Constituents change over time - edit this list whenever NSE rebalances the index.
NIFTY50 = [
    "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK", "BAJAJ-AUTO",
    "BAJFINANCE", "BAJAJFINSV", "BEL", "BHARTIARTL", "CIPLA", "COALINDIA", "DRREDDY",
    "EICHERMOT", "ETERNAL", "GRASIM", "HCLTECH", "HDFCBANK", "HDFCLIFE", "HINDALCO",
    "HINDUNILVR", "ICICIBANK", "INFY", "INDIGO", "ITC", "JIOFIN", "JSWSTEEL",
    "KOTAKBANK", "LT", "M&M", "MARUTI", "MAXHEALTH", "NESTLEIND", "NTPC", "ONGC",
    "POWERGRID", "RELIANCE", "SBILIFE", "SHRIRAMFIN", "SBIN", "SUNPHARMA", "TCS",
    "TATACONSUMER", "TATAMOTORS", "TATASTEEL", "TECHM", "TITAN", "TRENT",
    "ULTRACEMCO", "WIPRO",
]
INDEX_TICKER = "^NSEI"  # NIFTY 50 index on Yahoo Finance
DEFAULT_CSV = Path(__file__).resolve().parent / "data" / "NIFTY50_all.csv"
OHLCV = ["Open", "High", "Low", "Close", "Volume"]


def to_yahoo(sym: str) -> str:
    sym = sym.strip().upper()
    return sym if sym.startswith("^") or "." in sym else sym + ".NS"


def _tidy(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[[c for c in OHLCV if c in df.columns]].dropna(subset=["Close"]).copy()
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    return df[~df.index.duplicated(keep="last")].sort_index()


# ---------------------------------------------------------------- live data
@st.cache_data(ttl=300, show_spinner="Fetching market data...")
def fetch_live(symbol: str, period: str = "5y") -> pd.DataFrame:
    raw = yf.download(to_yahoo(symbol), period=period, interval="1d",
                      auto_adjust=True, progress=False, threads=False)
    if raw is None or raw.empty:
        raise ValueError(f"No data returned for '{symbol}'. Check the ticker.")
    return _tidy(raw)


@st.cache_data(ttl=600, show_spinner="Fetching prices for several stocks...")
def fetch_close_matrix_live(symbols: tuple, period: str = "1y") -> pd.DataFrame:
    tickers = [to_yahoo(s) for s in symbols]
    raw = yf.download(tickers, period=period, interval="1d", auto_adjust=True,
                      progress=False, threads=True)["Close"]
    if isinstance(raw, pd.Series):
        raw = raw.to_frame(symbols[0])
    raw = raw.dropna(how="all")
    raw.columns = [str(c).replace(".NS", "") for c in raw.columns]
    raw.index = pd.to_datetime(raw.index).tz_localize(None).normalize()
    return raw


# ------------------------------------------------------------- Kaggle data
@st.cache_data(show_spinner="Reading CSV...")
def load_kaggle(raw: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(raw), parse_dates=["Date"])
    missing = {"Date", "Symbol", "Close"} - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {sorted(missing)}. "
                         "Use NIFTY50_all.csv from the Kaggle dataset.")
    for c in ("Open", "High", "Low"):
        if c not in df:
            df[c] = df["Close"]
    if "Volume" not in df:
        df["Volume"] = 0
    return df.sort_values(["Symbol", "Date"]).reset_index(drop=True)


def kaggle_stock(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    return _tidy(df[df.Symbol == symbol].set_index("Date")[OHLCV])


def kaggle_close_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return df.pivot_table(index="Date", columns="Symbol", values="Close")


def kaggle_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """The Kaggle file has no index series, so build an equal-weighted Nifty proxy."""
    ret = kaggle_close_matrix(df).pct_change(fill_method=None).mean(axis=1).fillna(0)
    return ((1 + ret).cumprod() * 1000).to_frame("Close")

"""Technical indicators and risk statistics."""



def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d, c = df.copy(), df["Close"]
    for w in (20, 50, 200):
        d[f"SMA{w}"] = c.rolling(w).mean()
    d["EMA12"], d["EMA26"] = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    d["MACD"] = d.EMA12 - d.EMA26
    d["MACD_signal"] = d.MACD.ewm(span=9, adjust=False).mean()
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    d["RSI"] = 100 - 100 / (1 + gain / loss)
    mid, sd = c.rolling(20).mean(), c.rolling(20).std()
    d["BB_up"], d["BB_low"] = mid + 2 * sd, mid - 2 * sd
    tr = pd.concat([d.High - d.Low, (d.High - c.shift()).abs(), (d.Low - c.shift()).abs()], axis=1).max(axis=1)
    d["ATR"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    d["Ret"] = c.pct_change()
    return d


def drawdown(close: pd.Series) -> pd.Series:
    return close / close.cummax() - 1


def risk_stats(close: pd.Series, bench: pd.Series | None = None, rf: float = 0.065) -> dict:
    r = close.pct_change().dropna()
    cagr = (close.iloc[-1] / close.iloc[0]) ** (252 / len(r)) - 1
    vol = r.std() * np.sqrt(252)
    out = {
        "Total return": close.iloc[-1] / close.iloc[0] - 1,
        "CAGR": cagr,
        "Annualised volatility": vol,
        "Sharpe (rf 6.5%)": (cagr - rf) / vol if vol else np.nan,
        "Max drawdown": drawdown(close).min(),
        "1-day VaR (95%)": r.quantile(0.05),
        "Best day": r.max(),
        "Worst day": r.min(),
    }
    if bench is not None:
        j = pd.concat([r, bench.pct_change()], axis=1, join="inner").dropna()
        j.columns = ["s", "b"]
        if len(j) > 30:
            beta = j.cov().iloc[0, 1] / j.b.var()
            out["Beta vs Nifty"] = beta
            out["Alpha (annualised)"] = (j.s.mean() - beta * j.b.mean()) * 252
            out["Correlation vs Nifty"] = j.corr().iloc[0, 1]
    return out

"""Walk-forward price forecasting with honest baselines."""



MODELS = {
    "Ridge (linear)": lambda: make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
    "Random Forest": lambda: RandomForestRegressor(n_estimators=200, min_samples_leaf=10, n_jobs=-1, random_state=42),
    "Gradient Boosting": lambda: GradientBoostingRegressor(n_estimators=150, max_depth=3, learning_rate=0.05, random_state=42),
}


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    d = add_indicators(df)
    f = pd.DataFrame(index=d.index)
    for lag in (1, 2, 3, 5, 10, 21):
        f[f"ret_{lag}"] = d.Close.pct_change(lag)
    f["rsi"] = d.RSI
    f["macd"], f["macd_sig"] = d.MACD / d.Close, d.MACD_signal / d.Close
    f["bb_pos"] = (d.Close - d.BB_low) / (d.BB_up - d.BB_low)
    f["sma20_gap"], f["sma50_gap"] = d.Close / d.SMA20 - 1, d.Close / d.SMA50 - 1
    f["vol20"] = d.Ret.rolling(20).std()
    f["atr"] = d.ATR / d.Close
    f["vol_chg"] = np.log1p(d.Volume).diff(5)
    return f.replace([np.inf, -np.inf], np.nan)


def forecast(df: pd.DataFrame, model_name: str, horizon: int = 5, test_frac: float = 0.2) -> dict:
    X = make_features(df)
    y = np.log(df.Close.shift(-horizon) / df.Close).rename("y")
    data = X.join(y)
    live_row = X.iloc[[-1]]
    lab = data.dropna()
    if len(lab) < 250 or live_row.isna().any().any():
        raise ValueError("Need at least ~1.5 years of daily data for a forecast.")

    split = int(len(lab) * (1 - test_frac))
    train, test = lab.iloc[: split - horizon], lab.iloc[split:]  # gap = no label leakage
    model = MODELS[model_name]()
    model.fit(train[X.columns], train.y)
    pred = model.predict(test[X.columns])

    pos = df.index.get_indexer(test.index)
    close_t = df.Close.iloc[pos].to_numpy()
    actual = close_t * np.exp(test.y.to_numpy())
    predicted = close_t * np.exp(pred)
    mape = lambda a, p: float(np.mean(np.abs(a - p) / a))
    backtest = pd.DataFrame({"Actual": actual, "Predicted": predicted, "Naive (no change)": close_t},
                            index=df.index[pos + horizon])
    metrics = {
        "Model MAPE": mape(actual, predicted),
        "Naive MAPE": mape(actual, close_t),
        "Directional accuracy": float(np.mean(np.sign(pred) == np.sign(test.y.to_numpy()))),
        "Test samples": len(test),
    }
    resid = test.y.to_numpy() - pred

    final = MODELS[model_name]()
    final.fit(lab[X.columns], lab.y)
    p = float(final.predict(live_row)[0])
    last = float(df.Close.iloc[-1])
    s = float(resid.std())
    est = final[-1] if hasattr(final, "steps") else final
    imp = getattr(est, "feature_importances_", None)
    if imp is None:
        imp = np.abs(est.coef_)
    return {
        "metrics": metrics, "backtest": backtest,
        "last_close": last, "price": last * np.exp(p), "low": last * np.exp(p - 1.28 * s),
        "high": last * np.exp(p + 1.28 * s),
        "target_date": pd.bdate_range(df.index[-1], periods=horizon + 1)[-1],
        "importance": pd.Series(imp, index=X.columns).sort_values(),
    }

D = SimpleNamespace(NIFTY50=NIFTY50, INDEX_TICKER=INDEX_TICKER, DEFAULT_CSV=DEFAULT_CSV,
                    fetch_live=fetch_live, fetch_close_matrix_live=fetch_close_matrix_live,
                    load_kaggle=load_kaggle, kaggle_stock=kaggle_stock,
                    kaggle_close_matrix=kaggle_close_matrix, kaggle_proxy=kaggle_proxy)
I = SimpleNamespace(add_indicators=add_indicators, drawdown=drawdown, risk_stats=risk_stats)
M = SimpleNamespace(MODELS=MODELS, forecast=forecast)

# ===== UI =====


st.set_page_config(page_title="Nifty 50 Analytics & Forecast", page_icon="📈", layout="wide")
st.title("📈 Nifty 50 Analytics & Price Forecast")
st.caption("Educational tool, not investment advice. Live prices come from Yahoo Finance (NSE data is ~15 min delayed).")
st.markdown("Designed and developed by **Rik Karmakar**")
# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Data")
    source = st.radio("Source", ["Live (Yahoo Finance)", "Kaggle CSV"])
    live = source.startswith("Live")
    if live:
        choice = st.selectbox("Nifty 50 stock", D.NIFTY50, index=D.NIFTY50.index("RELIANCE"))
        custom = st.text_input("...or any NSE ticker (optional)", "").strip().upper()
        symbol = custom or choice
        period = st.selectbox("History", ["1y", "2y", "5y", "10y", "max"], index=2)
        if st.button("🔄 Refresh live data"):
            st.cache_data.clear()
            st.rerun()
    else:
        up = st.file_uploader("Upload NIFTY50_all.csv (Kaggle)", type="csv")
        if up is not None:
            raw = up.getvalue()
        elif D.DEFAULT_CSV.exists():
            raw = D.DEFAULT_CSV.read_bytes()
        else:
            st.info("Upload the Kaggle file, or put it in data/NIFTY50_all.csv.")
            st.stop()
        try:
            kdf = D.load_kaggle(raw)
        except Exception as e:
            st.error(str(e)); st.stop()
        symbol = st.selectbox("Stock", sorted(kdf.Symbol.unique()))
        period = "max"

try:
    if live:
        full = D.fetch_live(symbol, period)
        bench_full = D.fetch_live(D.INDEX_TICKER, period)
        bench_name = "NIFTY 50"
    else:
        full = D.kaggle_stock(kdf, symbol)
        bench_full = D.kaggle_proxy(kdf)
        bench_name = "Nifty proxy (equal-weighted)"
except Exception as e:
    st.error(f"Could not load data: {e}"); st.stop()

if len(full) < 60:
    st.warning("Not enough history for this symbol."); st.stop()

lo, hi = full.index.min().to_pydatetime(), full.index.max().to_pydatetime()
with st.sidebar:
    rng = st.slider("Date range", lo, hi, (max(lo, hi - pd.Timedelta(days=730)), hi), format="YYYY-MM-DD")
df = full.loc[rng[0]:rng[1]]
bench = bench_full.loc[rng[0]:rng[1]]
if len(df) < 30:
    st.warning("Select a wider date range (30+ trading days)."); st.stop()

# ------------------------------------------------------------------ header KPIs
last, prev = df.Close.iloc[-1], df.Close.iloc[-2]
yr = full.tail(252)
k = st.columns(5)
k[0].metric(f"{symbol} last close", f"₹{last:,.2f}", f"{last / prev - 1:+.2%}")
k[1].metric("52-week high", f"₹{yr.High.max():,.2f}")
k[2].metric("52-week low", f"₹{yr.Low.min():,.2f}")
k[3].metric("Volume", f"{df.Volume.iloc[-1]:,.0f}")
k[4].metric("Data through", str(df.index[-1].date()))

tabs = st.tabs(["📊 Technical", "🆚 vs Nifty 50", "⚠️ Risk & returns", "🧮 Multi-stock", "🔮 Forecast", "💾 Data"])

# ------------------------------------------------------------------ technical
with tabs[0]:
    c = st.columns([3, 1])
    mas = c[0].multiselect("Moving averages", ["SMA20", "SMA50", "SMA200", "EMA12", "EMA26"], ["SMA20", "SMA50"])
    bb = c[1].checkbox("Bollinger bands")
    ind = I.add_indicators(full).loc[rng[0]:rng[1]]
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.02,
                        row_heights=[0.5, 0.15, 0.17, 0.18])
    fig.add_trace(go.Candlestick(x=ind.index, open=ind.Open, high=ind.High, low=ind.Low,
                                 close=ind.Close, name="Price"), 1, 1)
    for m in mas:
        fig.add_trace(go.Scatter(x=ind.index, y=ind[m], name=m, line=dict(width=1.2)), 1, 1)
    if bb:
        for col in ("BB_up", "BB_low"):
            fig.add_trace(go.Scatter(x=ind.index, y=ind[col], name=col, line=dict(width=1, dash="dot")), 1, 1)
    fig.add_trace(go.Bar(x=ind.index, y=ind.Volume, name="Volume", marker_opacity=0.5), 2, 1)
    fig.add_trace(go.Scatter(x=ind.index, y=ind.RSI, name="RSI(14)"), 3, 1)
    for lvl in (30, 70):
        fig.add_hline(y=lvl, line_dash="dot", row=3, col=1)
    fig.add_trace(go.Scatter(x=ind.index, y=ind.MACD, name="MACD"), 4, 1)
    fig.add_trace(go.Scatter(x=ind.index, y=ind.MACD_signal, name="Signal"), 4, 1)
    fig.add_trace(go.Bar(x=ind.index, y=ind.MACD - ind.MACD_signal, name="Hist", marker_opacity=0.5), 4, 1)
    fig.update_layout(height=820, xaxis_rangeslider_visible=False, legend=dict(orientation="h", y=1.03),
                      margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, width="stretch")
    rsi = ind.RSI.iloc[-1]
    st.info(f"RSI(14) = **{rsi:.1f}** → " + ("overbought (>70)" if rsi > 70 else "oversold (<30)" if rsi < 30 else "neutral")
            + f" · MACD is **{'above' if ind.MACD.iloc[-1] > ind.MACD_signal.iloc[-1] else 'below'}** its signal line.")

# ------------------------------------------------------------------ vs Nifty
with tabs[1]:
    j = pd.concat([df.Close, bench.Close], axis=1, join="inner").dropna()
    j.columns = [symbol, bench_name]
    reb = j / j.iloc[0] * 100
    st.plotly_chart(px.line(reb, title="Rebased to 100").update_layout(height=380, legend_title=""), width="stretch")
    a, b = st.columns(2)
    rs = (reb[symbol] / reb[bench_name]).rename("Relative strength")
    a.plotly_chart(px.line(rs, title=f"{symbol} ÷ {bench_name} (rising = outperforming)").update_layout(height=320, showlegend=False), width="stretch")
    r = j.pct_change().dropna()
    win = st.slider("Rolling beta window (days)", 20, 250, 60)
    beta = (r[symbol].rolling(win).cov(r[bench_name]) / r[bench_name].rolling(win).var()).rename("Beta")
    b.plotly_chart(px.line(beta, title=f"Rolling {win}-day beta").update_layout(height=320, showlegend=False), width="stretch")
    slope, icpt = np.polyfit(r[bench_name], r[symbol], 1)
    sc = px.scatter(r, x=bench_name, y=symbol, opacity=0.5, title="Daily returns: stock vs benchmark")
    xs = np.array([r[bench_name].min(), r[bench_name].max()])
    sc.add_trace(go.Scatter(x=xs, y=slope * xs + icpt, name=f"β = {slope:.2f}", line=dict(color="orange")))
    st.plotly_chart(sc.update_layout(height=420), width="stretch")

# ------------------------------------------------------------------ risk
with tabs[2]:
    stats = I.risk_stats(df.Close, bench.Close)
    tbl = pd.DataFrame({"Metric": list(stats), "Value": [
        f"{v:.2f}" if ("Sharpe" in n or "Beta" in n or "Correlation" in n) else f"{v:.2%}" for n, v in stats.items()]})
    a, b = st.columns([1, 2])
    a.dataframe(tbl, hide_index=True, width="stretch")
    ret = df.Close.pct_change().dropna()
    b.plotly_chart(px.histogram(ret, nbins=60, title="Daily return distribution").update_layout(height=330, showlegend=False), width="stretch")
    a, b = st.columns(2)
    a.plotly_chart(px.area(I.drawdown(df.Close), title="Drawdown").update_layout(height=300, showlegend=False), width="stretch")
    b.plotly_chart(px.line(ret.rolling(30).std() * np.sqrt(252), title="30-day annualised volatility").update_layout(height=300, showlegend=False), width="stretch")
    m = df.Close.groupby(df.index.to_period("M")).last().pct_change().dropna()
    hm = pd.DataFrame({"Year": m.index.year, "Month": m.index.month, "Ret": m.values}).pivot(index="Year", columns="Month", values="Ret")
    st.plotly_chart(px.imshow(hm * 100, text_auto=".1f", color_continuous_scale="RdYlGn", color_continuous_midpoint=0,
                              title="Monthly returns (%)", aspect="auto"), width="stretch")

# ------------------------------------------------------------------ multi-stock
with tabs[3]:
    universe = D.NIFTY50 if live else sorted(kdf.Symbol.unique())
    default = [s for s in ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK"] if s in universe] or universe[:5]
    picks = st.multiselect("Compare stocks", universe, default)
    allstocks = st.checkbox("Load the whole universe for the screener (slower)")
    syms = tuple(universe) if allstocks else tuple(picks)
    if len(syms) >= 2:
        cm = D.fetch_close_matrix_live(syms, period) if live else D.kaggle_close_matrix(kdf)[list(syms)]
        cm = cm.loc[rng[0]:rng[1]].dropna(how="all")
        if allstocks:
            shown = cm[[p for p in picks if p in cm.columns]] if len(picks) >= 2 else cm.iloc[:, :5]
        else:
            shown = cm
        st.plotly_chart(px.line(shown / shown.bfill().iloc[0] * 100, title="Rebased to 100").update_layout(height=380), width="stretch")
        st.plotly_chart(px.imshow(shown.pct_change().corr(), text_auto=".2f", color_continuous_scale="RdBu_r",
                                  zmin=-1, zmax=1, title="Return correlation"), width="stretch")
        if allstocks:
            def pr(n):
                return cm.iloc[-1] / cm.iloc[-1 - n] - 1 if len(cm) > n else np.nan
            scr = pd.DataFrame({"1M": pr(21), "3M": pr(63), "6M": pr(126), "1Y": pr(252),
                                "Ann. vol": cm.pct_change().std() * np.sqrt(252)}).sort_values("3M", ascending=False)
            st.subheader("Screener")
            st.dataframe(scr.style.format("{:.1%}"), width="stretch")
    else:
        st.info("Pick at least two stocks.")

# ------------------------------------------------------------------ forecast
with tabs[4]:
    st.warning("Markets are close to a random walk. Read the baseline comparison below before trusting any number here.")
    a, b, c = st.columns([2, 2, 1])
    mname = a.selectbox("Model", list(M.MODELS))
    horizon = b.slider("Forecast horizon (trading days)", 1, 30, 5)
    run = c.button("Run forecast", type="primary")
    if run:
        try:
            with st.spinner("Training with a chronological train/test split..."):
                res = M.forecast(full, mname, horizon)
            st.session_state["fc"] = (symbol, mname, horizon, res)
        except Exception as e:
            st.error(str(e))
    fc = st.session_state.get("fc")
    if fc and fc[:3] == (symbol, mname, horizon):
        res = fc[3]
        x = st.columns(4)
        x[0].metric(f"Forecast for {res['target_date'].date()}", f"₹{res['price']:,.2f}", f"{res['price'] / res['last_close'] - 1:+.2%}")
        x[1].metric("~80% range", f"₹{res['low']:,.0f} – ₹{res['high']:,.0f}")
        mt = res["metrics"]
        x[2].metric("Model MAPE (test)", f"{mt['Model MAPE']:.2%}", f"{mt['Model MAPE'] - mt['Naive MAPE']:+.2%} vs naive", delta_color="inverse")
        x[3].metric("Direction hit-rate", f"{mt['Directional accuracy']:.1%}")
        if mt["Model MAPE"] >= mt["Naive MAPE"]:
            st.error("On held-out data this model did NOT beat the naive 'price stays the same' baseline. Treat the forecast as noise.")
        else:
            st.success("The model beat the naive baseline on held-out data (still no guarantee going forward).")
        hist = full.Close.tail(120)
        f1 = go.Figure(go.Scatter(x=hist.index, y=hist, name="Close"))
        f1.add_trace(go.Scatter(x=[hist.index[-1], res["target_date"]], y=[res["last_close"], res["price"]],
                                name="Forecast", mode="lines+markers", line=dict(dash="dash", color="orange")))
        f1.add_trace(go.Scatter(x=[res["target_date"]] * 2, y=[res["low"], res["high"]], name="80% range",
                                mode="lines", line=dict(width=6, color="rgba(255,165,0,.4)")))
        st.plotly_chart(f1.update_layout(height=360, title="Forecast"), width="stretch")
        bt = res["backtest"]
        st.plotly_chart(px.line(bt[["Actual", "Predicted"]], title=f"Backtest on unseen data ({horizon}-day-ahead price)").update_layout(height=360), width="stretch")
        st.plotly_chart(px.bar(res["importance"], orientation="h", title="Feature importance").update_layout(height=380, showlegend=False), width="stretch")

# ------------------------------------------------------------------ data
with tabs[5]:
    ind_full = I.add_indicators(df)
    st.dataframe(ind_full.tail(200).round(2), width="stretch")
    st.download_button("⬇️ Download prices + indicators (CSV)", ind_full.to_csv().encode(), f"{symbol}_analysis.csv", "text/csv")
    st.divider()
st.caption("© 2026 Rik Karmakar.")

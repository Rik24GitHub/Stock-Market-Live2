from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
import yfinance as yf
from scipy import stats as sps
from plotly.subplots import make_subplots
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score
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


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_ohlcv_many(symbols: tuple, period: str = "3y") -> dict:
    raw = yf.download([to_yahoo(s) for s in symbols], period=period, interval="1d",
                      auto_adjust=True, progress=False, threads=True, group_by="ticker")
    out = {}
    for s in symbols:
        try:
            sub = raw[to_yahoo(s)].dropna(subset=["Close"])
        except Exception:
            continue
        if len(sub) > 300:
            out[s] = _tidy(sub)
    return out


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


def spike_probs(df, horizon=5, thr=0.03, n_trees=200, test_frac=0.2):
    """P(high >= +thr) and P(low <= -thr) within the next `horizon` trading days."""
    X = make_features(df)
    hi = pd.concat([df.High.shift(-k) for k in range(1, horizon + 1)], axis=1).max(axis=1, skipna=False)
    lo = pd.concat([df.Low.shift(-k) for k in range(1, horizon + 1)], axis=1).min(axis=1, skipna=False)
    targets = {"up": (hi / df.Close - 1 >= thr).astype(float).where(hi.notna()),
               "down": (lo / df.Close - 1 <= -thr).astype(float).where(lo.notna())}
    live_row = X.iloc[[-1]]
    out = {}
    for name, y in targets.items():
        lab = X.join(y.rename("y")).dropna()
        if len(lab) < 250 or live_row.isna().any().any():
            raise ValueError("Need at least ~1.5 years of daily data.")
        split = int(len(lab) * (1 - test_frac))
        train, test = lab.iloc[: split - horizon], lab.iloc[split:]
        res = {"base_rate": float(test.y.mean()), "auc": np.nan, "top_hit": np.nan, "n_test": len(test)}
        if train.y.nunique() < 2 or lab.y.nunique() < 2:
            res["prob"] = float(lab.y.mean())
            out[name] = res
            continue
        mk = lambda: RandomForestClassifier(n_estimators=n_trees, min_samples_leaf=20, max_depth=8,
                                            n_jobs=-1, random_state=42)
        clf = mk().fit(train[X.columns], train.y)
        p = clf.predict_proba(test[X.columns])[:, 1]
        if test.y.nunique() == 2:
            res["auc"] = float(roc_auc_score(test.y, p))
        sel = p >= np.quantile(p, 0.8)
        if sel.any():
            res["top_hit"] = float(test.y.to_numpy()[sel].mean())
        final = mk().fit(lab[X.columns], lab.y)
        res["prob"] = float(final.predict_proba(live_row)[0, 1])
        out[name] = res
    return out


def rank_universe(frames, horizon=5, thr=0.03, progress=None):
    rows = []
    for i, (sym, d) in enumerate(frames.items()):
        try:
            f = forecast(d, "Ridge (linear)", horizon)
            sp = spike_probs(d, horizon, thr, 60)
            rows.append({
                "Symbol": sym, "Last close": f["last_close"],
                "Expected return": f["price"] / f["last_close"] - 1,
                "Beat baseline": f["metrics"]["Model MAPE"] < f["metrics"]["Naive MAPE"],
                "P(spike up)": sp["up"]["prob"], "P(drop)": sp["down"]["prob"],
                "RSI": add_indicators(d).RSI.iloc[-1],
                "Vol 20d": d.Close.pct_change().tail(20).std() * np.sqrt(252),
            })
        except Exception:
            pass
        if progress:
            progress((i + 1) / len(frames), sym)
    if not rows:
        return pd.DataFrame()
    r = pd.DataFrame(rows)
    r["Score"] = (r["Expected return"].rank(pct=True) + (r["P(spike up)"] - r["P(drop)"]).rank(pct=True)) / 2 * 100
    return r.sort_values("Score", ascending=False).reset_index(drop=True)

D = SimpleNamespace(NIFTY50=NIFTY50, INDEX_TICKER=INDEX_TICKER, DEFAULT_CSV=DEFAULT_CSV,
                    fetch_live=fetch_live, fetch_close_matrix_live=fetch_close_matrix_live,
                    load_kaggle=load_kaggle, kaggle_stock=kaggle_stock,
                    kaggle_close_matrix=kaggle_close_matrix, kaggle_proxy=kaggle_proxy,
                    fetch_ohlcv_many=fetch_ohlcv_many)
I = SimpleNamespace(add_indicators=add_indicators, drawdown=drawdown, risk_stats=risk_stats)
M = SimpleNamespace(MODELS=MODELS, forecast=forecast, spike_probs=spike_probs, rank_universe=rank_universe)

# ===== Visual theme =====
UP, DOWN = "#00c896", "#ff5c7a"
COLORWAY = ["#00c896", "#6c8cff", "#f5a524", "#ff5c7a", "#b388ff", "#4dd0e1", "#ffd54f"]
_grid = "rgba(128,128,128,0.18)"
pio.templates["nifty"] = go.layout.Template(layout=dict(
    colorway=COLORWAY, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#8b949e", size=13),
    xaxis=dict(gridcolor=_grid, zerolinecolor=_grid, linecolor=_grid),
    yaxis=dict(gridcolor=_grid, zerolinecolor=_grid, linecolor=_grid),
    legend=dict(orientation="h", y=1.08, x=0), margin=dict(l=10, r=10, t=60, b=10),
    title=dict(font=dict(size=16)),
))
pio.templates.default = "nifty"
RSI_STEPS = [{"range": [0, 30], "color": "rgba(0,200,150,.25)"},
             {"range": [30, 70], "color": "rgba(128,128,128,.12)"},
             {"range": [70, 100], "color": "rgba(255,92,122,.28)"}]


def show(fig, width="stretch"):
    st.plotly_chart(fig, theme=None, width=width)


def show_in(where, fig, width="stretch"):
    where.plotly_chart(fig, theme=None, width=width)


def gauge(value, title, lo=0, hi=100, steps=None, color=UP, suffix=""):
    v = float(np.nan_to_num(value))
    fig = go.Figure(go.Indicator(mode="gauge+number", value=v, number={"suffix": suffix, "valueformat": ".1f"},
                                 title={"text": title, "font": {"size": 15}},
                                 gauge={"axis": {"range": [lo, hi]}, "bar": {"color": color, "thickness": 0.3},
                                        "steps": steps or [], "borderwidth": 0}))
    fig.update_layout(height=230, margin=dict(l=25, r=25, t=55, b=5))
    return fig


# ===== UI =====


st.set_page_config(page_title="Nifty 50 Analytics & Forecast", page_icon="📈", layout="wide")
st.markdown("""<style>
.block-container{padding-top:2rem;max-width:1250px}
.hero{background:linear-gradient(120deg,#00a67e 0%,#2f6bff 55%,#7a3cff 100%);border-radius:18px;padding:26px 32px;margin-bottom:14px;color:#fff}
.hero h1{margin:0;font-size:2rem;color:#fff;padding:0}
.hero p{margin:6px 0 10px 0;opacity:.92;font-size:1.02rem}
.hero .by{display:inline-block;background:rgba(255,255,255,.18);border-radius:999px;padding:4px 14px;font-size:.88rem}
div[data-testid="stMetric"]{background:rgba(128,128,128,.08);border:1px solid rgba(128,128,128,.22);border-radius:14px;padding:14px 16px}
button[data-baseweb="tab"]{font-weight:600;font-size:.98rem}
</style>""", unsafe_allow_html=True)
st.markdown("""<div class="hero"><h1>📈 Nifty 50 Analytics &amp; Price Forecast</h1><p>Technical charts · risk analytics · ML forecasts · spike alerts · 5-day watchlist</p><span class="by">Designed and developed by <b>Rik Karmakar</b></span></div>""", unsafe_allow_html=True)
st.caption("Educational tool, not investment advice. Live prices come from Yahoo Finance (NSE data is ~15 min delayed).")

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

tabs = st.tabs(["📊 Technical", "🆚 vs Nifty 50", "⚠️ Risk & returns", "🧮 Multi-stock", "🔮 Forecast", "⚡ Spikes", "🎯 5-day picks", "💾 Data"])

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
    fig.add_trace(go.Bar(x=ind.index, y=ind.Volume, name="Volume", marker_color=np.where(ind.Close >= ind.Open, UP, DOWN), marker_opacity=0.55), 2, 1)
    fig.add_trace(go.Scatter(x=ind.index, y=ind.RSI, name="RSI(14)"), 3, 1)
    for lvl in (30, 70):
        fig.add_hline(y=lvl, line_dash="dot", row=3, col=1)
    fig.add_trace(go.Scatter(x=ind.index, y=ind.MACD, name="MACD"), 4, 1)
    fig.add_trace(go.Scatter(x=ind.index, y=ind.MACD_signal, name="Signal"), 4, 1)
    fig.add_trace(go.Bar(x=ind.index, y=ind.MACD - ind.MACD_signal, name="Hist", marker_color=np.where(ind.MACD >= ind.MACD_signal, UP, DOWN), marker_opacity=0.6), 4, 1)
    fig.update_traces(increasing_line_color=UP, decreasing_line_color=DOWN, increasing_fillcolor=UP,
                      decreasing_fillcolor=DOWN, selector=dict(type="candlestick"))
    fig.add_hrect(y0=70, y1=100, row=3, col=1, fillcolor=DOWN, opacity=0.08, line_width=0)
    fig.add_hrect(y0=0, y1=30, row=3, col=1, fillcolor=UP, opacity=0.08, line_width=0)
    fig.update_layout(height=820, xaxis_rangeslider_visible=False, legend=dict(orientation="h", y=1.03),
                      margin=dict(l=10, r=10, t=30, b=10))
    show(fig, width="stretch")
    rsi = ind.RSI.iloc[-1]
    g1, g2, g3 = st.columns(3)
    show_in(g1, gauge(rsi, "RSI (14)", 0, 100, RSI_STEPS))
    gap50 = (ind.Close.iloc[-1] / ind.SMA50.iloc[-1] - 1) * 100
    show_in(g2, gauge(gap50, "Distance from 50-day avg (%)", -15, 15, None, "#6c8cff", "%"))
    show_in(g3, gauge(ind.Ret.tail(20).std() * np.sqrt(252) * 100, "20-day volatility (ann.)", 0, 60, None, "#f5a524", "%"))
    st.info(f"RSI(14) = **{rsi:.1f}** → " + ("overbought (>70)" if rsi > 70 else "oversold (<30)" if rsi < 30 else "neutral")
            + f" · MACD is **{'above' if ind.MACD.iloc[-1] > ind.MACD_signal.iloc[-1] else 'below'}** its signal line.")

# ------------------------------------------------------------------ vs Nifty
with tabs[1]:
    j = pd.concat([df.Close, bench.Close], axis=1, join="inner").dropna()
    j.columns = [symbol, bench_name]
    reb = j / j.iloc[0] * 100
    show(px.line(reb, title="Rebased to 100").update_layout(height=380, legend_title=""), width="stretch")
    a, b = st.columns(2)
    rs = (reb[symbol] / reb[bench_name]).rename("Relative strength")
    show_in(a, px.line(rs, title=f"{symbol} ÷ {bench_name} (rising = outperforming)").update_layout(height=320, showlegend=False), width="stretch")
    r = j.pct_change().dropna()
    win = st.slider("Rolling beta window (days)", 20, 250, 60)
    beta = (r[symbol].rolling(win).cov(r[bench_name]) / r[bench_name].rolling(win).var()).rename("Beta")
    show_in(b, px.line(beta, title=f"Rolling {win}-day beta").update_layout(height=320, showlegend=False), width="stretch")
    slope, icpt = np.polyfit(r[bench_name], r[symbol], 1)
    sc = px.scatter(r, x=bench_name, y=symbol, opacity=0.5, title="Daily returns: stock vs benchmark")
    xs = np.array([r[bench_name].min(), r[bench_name].max()])
    sc.add_trace(go.Scatter(x=xs, y=slope * xs + icpt, name=f"β = {slope:.2f}", line=dict(color="orange")))
    show(sc.update_layout(height=420), width="stretch")

# ------------------------------------------------------------------ risk
with tabs[2]:
    stats = I.risk_stats(df.Close, bench.Close)
    tbl = pd.DataFrame({"Metric": list(stats), "Value": [
        f"{v:.2f}" if ("Sharpe" in n or "Beta" in n or "Correlation" in n) else f"{v:.2%}" for n, v in stats.items()]})
    items = list(zip(tbl.Metric, tbl.Value))
    for i in range(0, len(items), 4):
        for col, (n, v) in zip(st.columns(4), items[i:i + 4]):
            col.metric(n, v)
    b = st.container()
    ret = df.Close.pct_change().dropna()
    show_in(b, px.histogram(ret, nbins=60, title="Daily return distribution").update_layout(height=330, showlegend=False), width="stretch")
    a, b = st.columns(2)
    show_in(a, px.area(I.drawdown(df.Close), title="Drawdown").update_traces(line_color=DOWN, fillcolor="rgba(255,92,122,.3)").update_layout(height=300, showlegend=False), width="stretch")
    show_in(b, px.line(ret.rolling(30).std() * np.sqrt(252), title="30-day annualised volatility").update_layout(height=300, showlegend=False), width="stretch")
    m = df.Close.groupby(df.index.to_period("M")).last().pct_change().dropna()
    hm = pd.DataFrame({"Year": m.index.year, "Month": m.index.month, "Ret": m.values}).pivot(index="Year", columns="Month", values="Ret")
    show(px.imshow(hm * 100, text_auto=".1f", color_continuous_scale="RdYlGn", color_continuous_midpoint=0,
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
        show(px.line(shown / shown.bfill().iloc[0] * 100, title="Rebased to 100").update_layout(height=380), width="stretch")
        show(px.imshow(shown.pct_change().corr(), text_auto=".2f", color_continuous_scale="RdBu_r",
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
        show(f1.update_layout(height=360, title="Forecast"), width="stretch")
        bt = res["backtest"]
        show(px.line(bt[["Actual", "Predicted"]], title=f"Backtest on unseen data ({horizon}-day-ahead price)").update_layout(height=360), width="stretch")
        show(px.bar(res["importance"], orientation="h", title="Feature importance").update_layout(height=380, showlegend=False), width="stretch")

# ------------------------------------------------------------------ spikes
with tabs[5]:
    st.caption("Estimates the chance that the stock's high (or low) moves by the chosen size within the next N trading days.")
    a, b, c = st.columns([2, 2, 1])
    thr = a.slider("Spike size (%)", 1.0, 10.0, 3.0, 0.5) / 100
    win = b.slider("Window (trading days)", 1, 10, 5)
    if c.button("Run spike model", type="primary"):
        try:
            with st.spinner("Training spike classifiers..."):
                st.session_state["sp"] = (symbol, thr, win, M.spike_probs(full, win, thr))
        except Exception as e:
            st.error(str(e))
    sp = st.session_state.get("sp")
    if sp and sp[:3] == (symbol, thr, win):
        up, dn = sp[3]["up"], sp[3]["down"]
        x = st.columns(4)
        x[0].metric(f"P(rally ≥ {thr:.1%})", f"{up['prob']:.1%}", f"{up['prob'] - up['base_rate']:+.1%} vs normal")
        x[1].metric(f"P(drop ≥ {thr:.1%})", f"{dn['prob']:.1%}", f"{dn['prob'] - dn['base_rate']:+.1%} vs normal", delta_color="inverse")
        x[2].metric("Rally model AUC", "n/a" if np.isnan(up["auc"]) else f"{up['auc']:.2f}")
        x[3].metric("Drop model AUC", "n/a" if np.isnan(dn["auc"]) else f"{dn['auc']:.2f}")
        weak = [n for n, r in (("rally", up), ("drop", dn)) if np.isnan(r["auc"]) or r["auc"] < 0.55]
        if weak:
            st.warning("On held-out data the " + " and ".join(weak) + " model(s) showed little skill (AUC near 0.5 = coin flip). Treat those probabilities as roughly the stock's normal rate.")
        else:
            st.success("Both models ranked risky days better than chance on held-out data (AUC > 0.55). Still no guarantee.")
        g1, g2 = st.columns(2)
        show_in(g1, gauge(up["prob"] * 100, f"Chance of rally ≥ {thr:.1%} in {win}d", 0, 100, None, UP, "%"))
        show_in(g2, gauge(dn["prob"] * 100, f"Chance of drop ≥ {thr:.1%} in {win}d", 0, 100, None, DOWN, "%"))
        bars = pd.DataFrame({"Model now": [up["prob"], dn["prob"]], "Normal rate (test period)": [up["base_rate"], dn["base_rate"]]},
                            index=["Rally", "Drop"])
        show(px.bar(bars, barmode="group", title="Spike probability vs normal rate").update_layout(height=340, yaxis_tickformat=".0%"), width="stretch")
        st.caption(f"When the rally model was in its top 20% most confident days, a rally actually followed {up['top_hit']:.1%} of the time vs {up['base_rate']:.1%} normally." if not np.isnan(up["top_hit"]) else "")

# ------------------------------------------------------------------ picks
with tabs[6]:
    st.warning("A model-ranked watchlist for the next few trading days, NOT investment advice. Short-term moves are mostly noise; check the 'Beat baseline' column before giving any row weight.")
    if not live:
        st.info("Switch the sidebar Source to 'Live (Yahoo Finance)' to scan the Nifty 50.")
    else:
        a, b, c = st.columns([2, 2, 1])
        ph = a.slider("Horizon (trading days)", 1, 10, 5, key="ph")
        pthr = b.slider("Spike size for risk columns (%)", 1.0, 10.0, 3.0, 0.5, key="pthr") / 100
        go_scan = c.button("Scan Nifty 50", type="primary")
        if go_scan:
            frames = D.fetch_ohlcv_many(tuple(D.NIFTY50), "3y")
            if not frames:
                st.error("Could not download prices. Wait a minute and try again.")
            else:
                bar = st.progress(0.0, "Scoring stocks...")
                res = M.rank_universe(frames, ph, pthr, lambda f, s: bar.progress(f, f"Scoring {s}..."))
                bar.empty()
                st.session_state["picks"] = (ph, pthr, res)
        pk = st.session_state.get("picks")
        if pk and pk[:2] == (ph, pthr) and not pk[2].empty:
            res = pk[2].copy()
            n_ok = int(res["Beat baseline"].sum())
            st.caption(f"{n_ok} of {len(res)} stock models beat the naive baseline in backtests.")
            only_ok = st.checkbox("Only show stocks whose model beat the baseline")
            if only_ok:
                res = res[res["Beat baseline"]]
            if res.empty:
                st.info("No stock passed that filter right now. That is a normal, honest outcome for short-term forecasts.")
            else:
                disp = res.copy()
                disp["Beat baseline"] = disp["Beat baseline"].map({True: "✅", False: "❌"})
                for col in ("Expected return", "P(spike up)", "P(drop)", "Vol 20d"):
                    disp[col] = disp[col] * 100
                cfg = {
                    "Last close": st.column_config.NumberColumn(format="₹%.2f"),
                    "Expected return": st.column_config.NumberColumn(format="%.2f%%"),
                    "P(spike up)": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100),
                    "P(drop)": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100),
                    "RSI": st.column_config.NumberColumn(format="%.0f"),
                    "Vol 20d": st.column_config.NumberColumn(format="%.0f%%"),
                    "Score": st.column_config.ProgressColumn(format="%.0f", min_value=0, max_value=100),
                }
                st.subheader("Top 5 by score")
                for col, (_, r) in zip(st.columns(5), res.head(5).iterrows()):
                    col.metric(r["Symbol"], f"₹{r['Last close']:,.0f}", f"{r['Expected return']:+.2%} expected")
                    col.caption(f"Score {r['Score']:.0f} · up {r['P(spike up)']:.0%} · drop {r['P(drop)']:.0%} · {'✅' if r['Beat baseline'] else '❌'}")
                sc = px.scatter(res, x=res["Expected return"] * 100, y=(res["P(spike up)"] - res["P(drop)"]) * 100,
                                size="Score", color="Score", text="Symbol", size_max=30,
                                color_continuous_scale=[DOWN, "#f5c542", UP],
                                labels={"x": "Expected return (%)", "y": "P(spike up) minus P(drop) (pct. points)"},
                                title="Stock map: top-right = best model outlook")
                sc.add_hline(y=0, line_dash="dot", line_color="gray")
                sc.add_vline(x=0, line_dash="dot", line_color="gray")
                sc.update_traces(textposition="top center")
                show(sc.update_layout(height=500))
                st.dataframe(disp, column_config=cfg, hide_index=True, width="stretch")
                st.caption("Score (0-100) = average percentile of the model's expected return and of (P(spike up) minus P(drop)). Expected return comes from a Ridge model; spike probabilities from random-forest classifiers.")

# ------------------------------------------------------------------ data
with tabs[7]:
    ind_full = I.add_indicators(df)
    st.dataframe(ind_full.tail(200).round(2), width="stretch")
    st.download_button("⬇️ Download prices + indicators (CSV)", ind_full.to_csv().encode(), f"{symbol}_analysis.csv", "text/csv")

# ------------------------------------------------------------------ analytics dashboard
BLUE, AMBER = "#6c8cff", "#f5a524"


def _rsi(c, n=14):
    d = c.diff()
    up_, dn_ = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean(), (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up_ / dn_)


def metric_series(close, metric, wn, volume=None):
    r = close.pct_change()
    return {"Close price": close, "Daily return": r, "Cumulative return": close / close.iloc[0] - 1,
            "Rolling volatility (annualised)": r.rolling(wn).std() * np.sqrt(252),
            "Rolling return": close.pct_change(wn), "Drawdown": I.drawdown(close),
            "Rolling Sharpe": (r.rolling(wn).mean() * 252 - 0.065) / (r.rolling(wn).std() * np.sqrt(252)),
            "RSI": _rsi(close), "Volume": volume if volume is not None else close * np.nan}[metric]


def close_for(syms):
    cm = D.fetch_close_matrix_live(tuple(syms), period) if live else D.kaggle_close_matrix(kdf)[list(syms)]
    return cm.loc[rng[0]:rng[1]]


st.divider()
with st.expander("🔬 Analytics dashboard: analyse this stock in depth", expanded=False):
    A = st.tabs(["🧰 Chart builder", "📐 Statistics", "📅 Seasonality", "🎲 Monte Carlo", "🧮 Calculators"])

    with A[0]:
        a1, a2, a3 = st.columns(3)
        met = a1.selectbox("Metric", ["Close price", "Daily return", "Cumulative return", "Rolling volatility (annualised)",
                                      "Rolling return", "Drawdown", "Rolling Sharpe", "RSI", "Volume"])
        wn = a2.slider("Window (days)", 5, 250, 30, key="cb_w")
        kind = a3.radio("Style", ["Line", "Area", "Bars"], horizontal=True)
        extras = st.multiselect("Compare with other stocks", [u for u in universe if u != symbol], max_selections=4) if met != "Volume" else []
        out = metric_series(df.Close, met, wn, df.Volume).rename(symbol).to_frame()
        if extras:
            try:
                cx = close_for(extras)
                for e in cx.columns:
                    out[e] = metric_series(cx[e].dropna(), met, wn)
            except Exception as e:
                st.warning(f"Could not load comparison stocks: {e}")
        out = out.dropna(how="all")
        pct_metrics = {"Daily return", "Cumulative return", "Rolling volatility (annualised)", "Rolling return", "Drawdown"}
        fx = {"Line": px.line, "Area": px.area, "Bars": px.bar}[kind](out, title=f"{met}" + (f" ({wn}d)" if "Rolling" in met else ""))
        if met in pct_metrics:
            fx.update_layout(yaxis_tickformat=".1%")
        show(fx.update_layout(height=420, hovermode="x unified"))
        st.dataframe(out.describe().T.style.format("{:,.4f}"), width="stretch")

    with A[1]:
        r = df.Close.pct_change().dropna()
        conf = st.slider("VaR / CVaR confidence (%)", 90, 99, 95)
        var = r.quantile(1 - conf / 100)
        jb_stat, jb_p = sps.jarque_bera(r)
        left, right = st.columns([1, 1])
        summ = {"Mean daily return": f"{r.mean():.3%}", "Daily volatility": f"{r.std():.3%}", "Skewness": f"{r.skew():.2f}",
                "Excess kurtosis": f"{r.kurt():.2f}", "Up days": f"{(r > 0).mean():.1%}", f"VaR {conf}% (1-day)": f"{var:.2%}",
                f"CVaR {conf}% (1-day)": f"{r[r <= var].mean():.2%}", "Jarque-Bera p-value": f"{jb_p:.4f}"}
        left.dataframe(pd.DataFrame({"Value": summ}), width="stretch")
        left.caption("A small Jarque-Bera p-value (<0.05) means returns are NOT normally distributed (fat tails), which is typical for stocks.")
        osm, osr = sps.probplot(r, dist="norm", fit=False)
        qq = go.Figure(go.Scatter(x=osm, y=osr, mode="markers", name="Returns", marker=dict(size=4, opacity=0.6)))
        qq.add_trace(go.Scatter(x=[osm.min(), osm.max()], y=[r.mean() + r.std() * osm.min(), r.mean() + r.std() * osm.max()],
                                name="Normal", line=dict(color=AMBER)))
        show_in(right, qq.update_layout(height=340, title="Q-Q plot vs normal", xaxis_title="Theoretical", yaxis_title="Actual"))
        ac = pd.Series([r.autocorr(l) for l in range(1, 21)], index=range(1, 21))
        band = 1.96 / np.sqrt(len(r))
        fa = go.Figure(go.Bar(x=ac.index, y=ac, marker_color=np.where(ac.abs() > band, AMBER, BLUE), name="Autocorrelation"))
        fa.add_hline(y=band, line_dash="dot")
        fa.add_hline(y=-band, line_dash="dot")
        show(fa.update_layout(height=300, title="Return autocorrelation (bars beyond the dotted lines are statistically notable)", xaxis_title="Lag (days)"))

    with A[2]:
        r = df.Close.pct_change().dropna()
        s1, s2 = st.columns(2)
        dow = r.groupby(r.index.dayofweek).mean() * 100
        dow.index = [["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][i] for i in dow.index]
        show_in(s1, go.Figure(go.Bar(x=dow.index, y=dow, marker_color=np.where(dow >= 0, UP, DOWN))).update_layout(
            height=320, title="Average return by weekday (%)", showlegend=False))
        mm = df.Close.groupby(df.index.to_period("M")).last().pct_change().dropna()
        moy = mm.groupby(mm.index.month).mean() * 100
        moy.index = [["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][i - 1] for i in moy.index]
        show_in(s2, go.Figure(go.Bar(x=moy.index, y=moy, marker_color=np.where(moy >= 0, UP, DOWN))).update_layout(
            height=320, title="Average return by month (%)", showlegend=False))
        st.caption("Seasonal averages are noisy and often vanish out of sample. Use a long date range and treat them as curiosities, not signals.")

    with A[3]:
        m1, m2, m3 = st.columns(3)
        days = m1.slider("Days ahead", 5, 252, 30)
        sims = m2.select_slider("Simulations", [200, 500, 1000, 2000], 1000)
        target = m3.number_input("Target price (₹)", value=float(round(last * 1.05, 2)), step=1.0)
        how = st.radio("Return model", ["Bootstrap (resample actual returns)", "Normal (random walk with drift)"], horizontal=True)
        lr = np.log(df.Close).diff().dropna().tail(756)
        gen = np.random.default_rng(42)
        draws = gen.choice(lr.to_numpy(), size=(sims, days)) if how.startswith("Bootstrap") else gen.normal(lr.mean(), lr.std(), (sims, days))
        paths = last * np.exp(np.cumsum(draws, axis=1))
        q = np.percentile(paths, [5, 25, 50, 75, 95], axis=0)
        xs = pd.bdate_range(df.index[-1], periods=days + 1)[1:]
        fm = go.Figure()
        fm.add_trace(go.Scatter(x=xs, y=q[4], line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fm.add_trace(go.Scatter(x=xs, y=q[0], fill="tonexty", fillcolor="rgba(108,140,255,.15)", line=dict(width=0), name="5-95%"))
        fm.add_trace(go.Scatter(x=xs, y=q[3], line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fm.add_trace(go.Scatter(x=xs, y=q[1], fill="tonexty", fillcolor="rgba(108,140,255,.30)", line=dict(width=0), name="25-75%"))
        fm.add_trace(go.Scatter(x=xs, y=q[2], line=dict(color=UP, width=2.5), name="Median"))
        fm.add_hline(y=target, line_dash="dash", line_color=AMBER, annotation_text="Target")
        fm.add_trace(go.Scatter(x=df.index[-90:], y=df.Close.tail(90), name="Recent close", line=dict(color="rgba(200,200,200,.8)")))
        show(fm.update_layout(height=420, title="Simulated price paths (fan chart)"))
        hit = float((paths.max(axis=1) >= target).mean()) if target > last else float((paths.min(axis=1) <= target).mean())
        k = st.columns(4)
        k[0].metric("Median final price", f"₹{q[2][-1]:,.2f}", f"{q[2][-1] / last - 1:+.1%}")
        k[1].metric("5th to 95th pct", f"₹{q[0][-1]:,.0f} – ₹{q[4][-1]:,.0f}")
        k[2].metric("P(finish above today)", f"{(paths[:, -1] > last).mean():.0%}")
        k[3].metric("P(touch target)", f"{hit:.0%}")
        st.caption("Based only on this stock's past return distribution. It ignores news, earnings and regime changes; it shows a range of outcomes, not a forecast.")

    with A[4]:
        l, rr = st.columns(2)
        with l:
            st.subheader("What-if investment")
            amt = st.number_input("Amount invested (₹)", min_value=1000, value=100000, step=5000)
            start = st.date_input("Invested on", value=df.index[max(0, len(df) - 252)].date(),
                                  min_value=df.index[0].date(), max_value=df.index[-2].date())
            i0 = min(int(df.index.searchsorted(pd.Timestamp(start))), len(df) - 2)
            val = amt * last / df.Close.iloc[i0]
            bser = bench.Close.reindex(df.index, method="ffill")
            bval = amt * bser.iloc[-1] / bser.iloc[i0]
            yrs = max((df.index[-1] - df.index[i0]).days / 365.25, 1e-6)
            w1, w2 = st.columns(2)
            w1.metric(f"{symbol} value today", f"₹{val:,.0f}", f"{val / amt - 1:+.1%}")
            w2.metric(f"{bench_name} value today", f"₹{bval:,.0f}", f"{bval / amt - 1:+.1%}")
            st.caption(f"{symbol} annualised: {(val / amt) ** (1 / yrs) - 1:+.1%} over {yrs:.1f} years. Past performance does not predict the future.")
        with rr:
            st.subheader("Position sizing (ATR stop)")
            cap = st.number_input("Trading capital (₹)", min_value=10000, value=500000, step=10000)
            rk = st.slider("Risk per trade (% of capital)", 0.25, 5.0, 1.0, 0.25)
            kx = st.slider("Stop distance (× ATR)", 1.0, 4.0, 2.0, 0.5)
            atr = float(I.add_indicators(df).ATR.iloc[-1])
            risk_ps = kx * atr
            shares = int((cap * rk / 100) // risk_ps) if risk_ps > 0 else 0
            z = st.columns(3)
            z[0].metric("Shares", f"{shares:,}")
            z[1].metric("Stop price", f"₹{last - risk_ps:,.2f}", f"-{risk_ps / last:.1%}", delta_color="inverse")
            z[2].metric("2R target", f"₹{last + 2 * risk_ps:,.2f}", f"+{2 * risk_ps / last:.1%}")
            st.caption(f"Position value ₹{shares * last:,.0f} ({shares * last / cap:.0%} of capital). Illustrative long position, not advice.")

st.divider()
st.caption("© 2026 Rik Karmakar · Built with Streamlit · Educational use only, not investment advice.")

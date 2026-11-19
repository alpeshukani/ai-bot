"""
=====================================================================
AI MULTI-ASSET BOT - Structure + Liquidity + Risk  (PAPER ONLY)
=====================================================================
Instruments : BTCUSD, ETHUSD, SOLUSD, XAUTUSD  (Delta Exchange India)
Capital     : $100 per instrument (independent compounding)
Leverage    : 10x, margin 25% -> max notional 2.5x equity per trade
Timeframes  : entry 1H (configurable), bias 4H

STRATEGY (human-trader style rule engine)
  1. BIAS (4H)   : close vs EMA200 + EMA200 slope -> LONG / SHORT / NEUTRAL
  2. LEVELS (1H) : fractal pivots (L=3,R=3) clustered into S/R zones.
                   Tighter clusters = more touches = stronger level.
  3. LIQUIDITY   : equal highs/lows = pools. A wick through a level that
                   closes back inside = sweep (stop hunt).
  4. TRIGGER     : sweep/rejection at the level, in the direction of bias.
  5. SCORE       : bias(2) + touches(2) + trigger(3) + edge(1) + rsi(1)
                   + vol(1) + distance(1) = 11. Entry only if >= SCORE_MIN.
  6. RISK        : SL = structure + 1.0*ATR, size from 1% equity risk,
                   TP = 1.5R, ATR trail after 1.5R, time stop 48 bars,
                   max 2 trades/day, 3% daily loss cap.
                   (Breakeven-at-1R tested and REMOVED: it lowered PF.)

HONEST FILLS
  - Entry = pending order: signal bar close par banta hai, fill agle bar ke
    OPEN par + 1 tick adverse (taker fee). Live aur backtest same code.
  - Sirf CLOSED candles par signal (forming candle ignore).
  - Exit SL/trail = stop-market: taker fee + 1 tick slippage.
  - Exit TP = resting limit: maker fee, fill at TP price.
  - Gap: if bar opens beyond stop, exit at open (not at stop price).
  - SL checked with the stop that was effective BEFORE the bar closes;
    trail updated AFTER the close (no look-ahead).

MODES
  python algo_bot.py --once                # ek baar (paper) - GitHub/cron
  python algo_bot.py --backtest --days 120 # multi-asset backtest report
  python algo_bot.py --loop 60             # local continuous (paper)
  python algo_bot.py --test-telegram       # telegram check
  python algo_bot.py --backtest --days 120 --tf 15   # 15m entries, 1H bias
"""
import argparse
import bisect
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ----------------------------- config -----------------------------
BASE = "https://api.india.delta.exchange"
HDRS = {"Accept": "application/json", "User-Agent": "ai-multi-bot/1.0"}

SYMBOLS = ["BTCUSD", "ETHUSD", "SOLUSD", "XAUTUSD"]
SPEC = {
    "BTCUSD": dict(unit=0.001, tick=0.5, maker=0.0002, taker=0.0005),
    "ETHUSD": dict(unit=0.01, tick=0.05, maker=0.0002, taker=0.0005),
    "SOLUSD": dict(unit=1.0, tick=0.0001, maker=0.0002, taker=0.0005),
    "XAUTUSD": dict(unit=0.001, tick=0.01, maker=0.0001, taker=0.0001),
}

START_EQUITY = 100.0
MAX_NOTIONAL_X = 2.5          # 25% margin x 10x leverage
RISK_PCT = 1.0                # 1% equity risk per trade
RR = 1.5                      # take profit = 1.5R  (tested best)
ATR_SL = 1.0                  # SL buffer beyond structure (x ATR)
ATR_TRAIL = 1.5               # trail distance after 1.5R (x ATR)
BE_AT_R = 99.0                # breakeven move OFF (tested: hurts PF)
TRAIL_AT_R = 1.5              # start ATR trail at +1.5R
MAX_HOLD_BARS = 48
MAX_TRADES_DAY = 2
DAILY_LOSS_PCT = 3.0
SCORE_MIN = 7                 # out of 11
PIV_L, PIV_R = 3, 3
ZONE_ATR = 0.25               # cluster tolerance (x ATR)

# --- entry model ---
ENTRY_MODE = "market"         # market (next open) | limit (rest at level)
PEND_BARS = 3                 # limit order kitne bars tak valid

# --- trigger / filter knobs (tunable via AI_PARAMS env JSON) ---
TRIGGER_MODE = "strict"       # strict | sweep | either
MIN_TOUCHES = 2               # level must be tested at least this many times
MIN_WICK_ATR = 0.30           # rejection candle: wick must be >= this x ATR
CLOSE_POS_MIN = 0.55          # close must sit in top(bottom) 55% of range
SWEEP_DEPTH_ATR = 0.05        # wick must pierce level by this x ATR for a sweep
MAX_CHASE_ATR = 1.0           # skip if next open drifted more than this x ATR
BIAS_STRICT = True            # require EMA200 slope agreement
BIAS_MODE = "ema"             # ema | structure
ONE_PER_LEVEL_DAY = True      # no re-entry on same level same day
LEVEL_BAND_ATR = 1.2          # entry sirf itne ATR ke andar level se
MIN_SL_ATR = 0.35             # SL itna paas nahi hona chahiye (noise)
MAX_SL_ATR = 99.0             # SL max cap (99 = off; 3.0 par sirf 1 trade ghat-ta hai)

# runtime overrides for research: AI_PARAMS='{"RR":2,"SCORE_MIN":6}'
_PARAM_KEYS = ("RR", "ATR_SL", "ATR_TRAIL", "BE_AT_R", "TRAIL_AT_R",
               "SCORE_MIN", "MIN_TOUCHES", "MIN_WICK_ATR", "CLOSE_POS_MIN",
               "SWEEP_DEPTH_ATR", "MAX_CHASE_ATR", "MAX_TRADES_DAY",
               "MAX_HOLD_BARS", "PIV_L", "PIV_R", "ZONE_ATR", "RISK_PCT",
               "TRIGGER_MODE", "BIAS_MODE", "BIAS_STRICT", "ONE_PER_LEVEL_DAY",
               "ENTRY_MODE", "PEND_BARS", "LEVEL_BAND_ATR",
               "MIN_SL_ATR", "MAX_SL_ATR")


def apply_params(overrides):
    g = globals()
    for k, v in (overrides or {}).items():
        if k in _PARAM_KEYS:
            g[k] = v


apply_params(json.loads(os.environ.get("AI_PARAMS", "{}") or "{}"))

ENTRY_TF_MIN = 60             # override with --tf
STATE_FILE = "bot_state.json"
TRADES_FILE = "bot_trades.csv"
CONFIG_FILE = "bot_config.txt"

RES_MAP = {15: "15m", 60: "1h", 240: "4h", 1440: "1d"}


# ----------------------------- utils -----------------------------
QUIET = False


def log(msg):
    if QUIET:
        return
    print(msg, flush=True)


def now_utc_str():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def day_key(ts):
    return str(int(ts // 86400))


def telegram_creds():
    tok = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if (not tok or not chat) and os.path.exists(CONFIG_FILE):
        try:
            for line in open(CONFIG_FILE, encoding="utf-8"):
                if "=" in line:
                    k, v = line.split("=", 1)
                    k, v = k.strip().upper(), v.strip()
                    if k == "TELEGRAM_BOT_TOKEN" and not tok:
                        tok = v
                    elif k == "TELEGRAM_CHAT_ID" and not chat:
                        chat = v
        except Exception:
            pass
    return tok, chat


def send_telegram(text):
    tok, chat = telegram_creds()
    if not tok or not chat:
        return False
    try:
        r = requests.post(f"https://api.telegram.org/bot{tok}/sendMessage",
                          json={"chat_id": chat, "text": text, "parse_mode": "HTML"},
                          timeout=20)
        return r.status_code == 200
    except Exception as e:
        log(f"  ! telegram error: {e}")
        return False


# ----------------------------- data -----------------------------
CACHE_DIR = "data_cache"


def _cache_file(symbol, res, bars):
    return os.path.join(CACHE_DIR, f"{symbol}_{res}_{bars}.pkl")


def fetch_candles(symbol, tf_min, bars):
    res = RES_MAP.get(tf_min)
    if not res:
        raise ValueError(f"unsupported tf {tf_min}")
    use_cache = os.environ.get("AI_CACHE") == "1"
    path = _cache_file(symbol, res, bars)
    if use_cache and os.path.exists(path) and (time.time() - os.path.getmtime(path) < 6 * 3600):
        return pd.read_pickle(path)

    now = int(time.time())
    secs = tf_min * 60
    start = now - int(bars * secs * 1.06) - 3600
    r = requests.get(f"{BASE}/v2/history/candles",
                     params={"resolution": res, "symbol": symbol,
                             "start": start, "end": now},
                     headers=HDRS, timeout=40)
    d = r.json()
    if not d.get("success"):
        raise RuntimeError(f"{symbol} candles error: {str(d)[:160]}")
    rows = d.get("result") or []
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"{symbol}: koi candle nahi mila")
    for c in ("open", "high", "low", "close"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["time"] = df["time"].astype("int64")
    df = df.drop_duplicates("time").sort_values("time")
    df = df.tail(bars).reset_index(drop=True)
    if use_cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        df.to_pickle(path)
    return df


def fetch_mark(symbol):
    try:
        r = requests.get(f"{BASE}/v2/tickers/{symbol}", headers=HDRS, timeout=15)
        j = r.json()
        if j.get("success"):
            return float(j["result"]["mark_price"])
    except Exception:
        pass
    return None


# ----------------------------- indicators -----------------------------
def ema(arr, n):
    return pd.Series(arr).ewm(span=n, adjust=False).mean().to_numpy()


def atr_wilder(h, l, c, n=14):
    N = len(c)
    tr = np.full(N, np.nan)
    for i in range(1, N):
        pc = c[i - 1]
        tr[i] = max(h[i] - l[i], abs(h[i] - pc), abs(l[i] - pc))
    return pd.Series(tr).ewm(alpha=1.0 / n, min_periods=n, adjust=False).mean().to_numpy()


def rsi_wilder(c, n=14):
    N = len(c)
    out = np.full(N, np.nan)
    if N < n + 1:
        return out
    diff = np.diff(c)
    gain = np.where(diff > 0, diff, 0.0)
    loss = np.where(diff < 0, -diff, 0.0)
    ag, al = gain[:n].mean(), loss[:n].mean()
    out[n] = 100.0 if al == 0 else 100 - 100 / (1 + ag / al)
    for i in range(n + 1, N):
        ag = (ag * (n - 1) + gain[i - 1]) / n
        al = (al * (n - 1) + loss[i - 1]) / n
        out[i] = 100.0 if al == 0 else 100 - 100 / (1 + ag / al)
    return out


def percentile_rank(arr, i, lookback=100):
    lo = max(0, i - lookback)
    win = arr[lo:i + 1]
    win = win[~np.isnan(win)]
    if len(win) < 20 or np.isnan(arr[i]):
        return 50.0
    return float((win <= arr[i]).mean() * 100.0)


# ----------------------------- levels -----------------------------
def find_pivots(h, l, left=PIV_L, right=PIV_R):
    """Fractal pivots. Confirmed only after `right` bars."""
    piv = []
    n = len(h)
    for i in range(left, n - right):
        seg_h = h[i - left:i + right + 1]
        seg_l = l[i - left:i + right + 1]
        if h[i] == np.max(seg_h):
            piv.append((i, float(h[i]), "R", i + right))
        elif l[i] == np.min(seg_l):
            piv.append((i, float(l[i]), "S", i + right))
    return piv


def build_levels(df, atr):
    """Cluster pivots into S/R zones - CAUSAL.

    Har level ka price sirf un pivots se banta hai jo us waqt tak CONFIRM ho
    chuke the (conf <= i). Future pivots se average karna look-ahead hoga,
    isliye pts list rakhi hai aur price on-demand nikala jata hai.
    """
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    piv = find_pivots(h, l)
    if not piv:
        return []
    atr_med = float(np.nanmedian(atr)) if np.isfinite(np.nanmedian(atr)) else 0.0

    levels = []
    for idx, px, kind, conf in piv:
        a = atr[idx] if np.isfinite(atr[idx]) and atr[idx] > 0 else atr_med
        tol = ZONE_ATR * (a if a > 0 else 1.0)
        hit = None
        for lv in levels:
            if lv["kind"] != kind:
                continue
            cur = _lv_price(lv, conf - 1)
            if cur is not None and abs(cur - px) <= tol:
                hit = lv
                break
        if hit is None:
            levels.append(dict(kind=kind, pts=[(int(conf), float(px))],
                               first=int(idx), broken=None))
        else:
            hit["pts"].append((int(conf), float(px)))

    n = len(c)
    for lv in levels:
        lv["pts"].sort()
        lv["_confs"] = [p[0] for p in lv["pts"]]
        lv["_cs"] = np.cumsum([p[1] for p in lv["pts"]])

    # break detection, walk-forward: close beyond zone by > 0.6 ATR
    for lv in levels:
        for i in range(lv["_confs"][0], n):
            cur = _lv_price(lv, i)
            if cur is None:
                continue
            a = atr[i] if np.isfinite(atr[i]) and atr[i] > 0 else atr_med
            if a <= 0:
                continue
            if lv["kind"] == "S" and c[i] < cur - 0.6 * a:
                lv["broken"] = i
                break
            if lv["kind"] == "R" and c[i] > cur + 0.6 * a:
                lv["broken"] = i
                break
    return levels


def _lv_price(lv, i):
    """Level price using only pivots confirmed at or before bar i."""
    confs = lv.get("_confs")
    cs = lv.get("_cs")
    if confs is None:
        confs = [p[0] for p in lv["pts"]]
        cs = [p[1] for p in lv["pts"]]
        k = bisect.bisect_right(confs, i)
        return float(sum(cs[:k]) / k) if k > 0 else None
    k = bisect.bisect_right(confs, i)
    if k <= 0:
        return None
    return float(cs[k - 1] / k)


def _lv_touches(lv, i):
    confs = lv.get("_confs") or [p[0] for p in lv["pts"]]
    return bisect.bisect_right(confs, i)


def nearest_levels(levels, i, close, atr_i, band_atr=1.2):
    """Nearest unbroken support below and resistance above, within band."""
    sup, res = None, None
    for lv in levels:
        k = _lv_touches(lv, i)
        if k < MIN_TOUCHES:
            continue
        if lv["broken"] is not None and lv["broken"] <= i:
            continue
        px = _lv_price(lv, i)
        if px is None:
            continue
        d = close - px
        if lv["kind"] == "S" and -0.1 * atr_i <= d <= band_atr * atr_i:
            if sup is None or px > sup["px"]:
                sup = dict(px=px, kind="S", touches=k)
        if lv["kind"] == "R" and -0.1 * atr_i <= -d <= band_atr * atr_i:
            if res is None or px < res["px"]:
                res = dict(px=px, kind="R", touches=k)
    return sup, res


# ----------------------------- context -----------------------------
def build_context(symbol, entry_tf, bars_entry, bars_bias):
    df_e = fetch_candles(symbol, entry_tf, bars_entry)
    bias_tf = entry_tf * 4
    df_b = fetch_candles(symbol, bias_tf, bars_bias)

    # abhi ban rahi candle hatao - sirf CLOSED bars par signal
    now = int(time.time())
    df_e = df_e[df_e["time"] + entry_tf * 60 <= now].reset_index(drop=True)
    df_b = df_b[df_b["time"] + bias_tf * 60 <= now].reset_index(drop=True)
    if len(df_e) < 50 or len(df_b) < 210:
        raise RuntimeError(f"{symbol}: closed candles kam hain "
                           f"({len(df_e)} x {entry_tf}m, {len(df_b)} x {bias_tf}m)")

    he = df_e["high"].to_numpy(float)
    le = df_e["low"].to_numpy(float)
    ce = df_e["close"].to_numpy(float)
    oe = df_e["open"].to_numpy(float)
    te = df_e["time"].to_numpy("int64")

    hb = df_b["high"].to_numpy(float)
    lb = df_b["low"].to_numpy(float)
    cb = df_b["close"].to_numpy(float)
    tb = df_b["time"].to_numpy("int64")

    atr_e = atr_wilder(he, le, ce, 14)
    rsi_e = rsi_wilder(ce, 14)
    ema_b = ema(cb, 200)
    atr_b = atr_wilder(hb, lb, cb, 14)
    atr_b_med = float(np.nanmedian(atr_b)) if np.isfinite(np.nanmedian(atr_b)) else 1.0
    slope_bars = 10

    # bias per entry bar (uses last CLOSED bias bar)
    bidx = np.searchsorted(tb, te, side="left") - 1
    bias = np.zeros(len(te), dtype=int)
    for i in range(len(te)):
        j = bidx[i]
        if j < 200 or not np.isfinite(ema_b[j]) or not np.isfinite(atr_b[j]):
            bias[i] = 0
            continue
        j0 = max(0, j - slope_bars)
        rising = ema_b[j] > ema_b[j0]
        falling = ema_b[j] < ema_b[j0]
        if BIAS_MODE == "structure":
            up = cb[j] > ema_b[j] and cb[j] > cb[j0]
            dn = cb[j] < ema_b[j] and cb[j] < cb[j0]
            if BIAS_STRICT:
                up = up and rising
                dn = dn and falling
            bias[i] = 1 if up else (-1 if dn else 0)
        else:
            if cb[j] > ema_b[j] and (rising or not BIAS_STRICT):
                bias[i] = 1
            elif cb[j] < ema_b[j] and (falling or not BIAS_STRICT):
                bias[i] = -1
            else:
                bias[i] = 0

    levels = build_levels(df_e, atr_e)
    return dict(symbol=symbol, df_e=df_e, df_b=df_b, o=oe, h=he, l=le, c=ce,
                ot=te, atr=atr_e, rsi=rsi_e, bias=bias, levels=levels,
                ema_b=ema_b, atr_b=atr_b)


# ----------------------------- risk -----------------------------
def size_position(equity, entry, sl, unit):
    risk_usd = equity * RISK_PCT / 100.0
    sl_dist = abs(entry - sl)
    if sl_dist <= 0:
        return 0
    qty_risk = risk_usd / (sl_dist * unit)
    qty_cap = (equity * MAX_NOTIONAL_X) / (entry * unit)
    return max(0, int(min(qty_risk, qty_cap)))


def find_break(levels, i, bias_i, atr_i):
    """Bar i par jo level toota (close beyond by 0.6 ATR) us par breakout setup."""
    best = None
    for lv in levels:
        if lv.get("broken") != i:
            continue
        k = _lv_touches(lv, i)
        if k < MIN_TOUCHES:
            continue
        px = _lv_price(lv, i)
        if px is None:
            continue
        if lv["kind"] == "R" and bias_i >= 0:
            cand = dict(px=px, kind="R", touches=k)
        elif lv["kind"] == "S" and bias_i <= 0:
            cand = dict(px=px, kind="S", touches=k)
        else:
            continue
        d = 1 if lv["kind"] == "R" else -1
        if best is None or k > best[1]["touches"]:
            best = (d, cand)
    return best


def detect_trigger(d, oi, hi, li, ci, lvl_px, atr_i):
    """Bar par valid liquidity trigger hai ya nahi.

    sweep  = wick level ke paar gaya aur close wapas andar (stop hunt)
    rej    = level se strong wick rejection + close range ke sahi side
    Returns (is_sweep, is_rejection)
    """
    rng = hi - li
    if rng <= 0 or atr_i <= 0:
        return False, False
    if d == 1:
        wick = min(oi, ci) - li
        sweep = li <= lvl_px - SWEEP_DEPTH_ATR * atr_i and ci > lvl_px
        rej = (wick >= MIN_WICK_ATR * atr_i and ci > oi and ci > lvl_px
               and (ci - li) / rng >= CLOSE_POS_MIN)
        if TRIGGER_MODE == "either":
            rej = rej or (li <= lvl_px + 0.2 * atr_i and ci > lvl_px and ci > oi)
    else:
        wick = hi - max(oi, ci)
        sweep = hi >= lvl_px + SWEEP_DEPTH_ATR * atr_i and ci < lvl_px
        rej = (wick >= MIN_WICK_ATR * atr_i and ci < oi and ci < lvl_px
               and (hi - ci) / rng >= CLOSE_POS_MIN)
        if TRIGGER_MODE == "either":
            rej = rej or (hi >= lvl_px - 0.2 * atr_i and ci < lvl_px and ci < oi)

    if TRIGGER_MODE == "sweep":
        return sweep, False
    return sweep, rej


def score_setup(bias_i, lv, sweep, rejection, rsi_i, atr_pct, dist_atr,
                close_edge=0.0, breakout=False):
    """11-point confluence score. Sirf quality setups ko pass hone dete hain."""
    s = 0
    s += 2 if bias_i != 0 else 0                       # trend alignment
    t = lv["touches"]
    s += 2 if t >= 4 else (1 if t >= 3 else 0)         # level strength
    if breakout:
        s += 3                                          # S/R flip breakout
    elif sweep:
        s += 3                                          # liquidity sweep
    elif rejection:
        s += 2                                          # wick rejection
    s += 1 if close_edge >= 0.15 else 0                # momentum away from level
    if 30 <= rsi_i <= 70:
        s += 1
    if 15 <= atr_pct <= 90:
        s += 1
    if dist_atr <= 0.40:
        s += 1
    return s


# ----------------------------- engine -----------------------------
def _entry_fee(px, qty, unit, rate):
    return abs(px * qty * unit) * rate


def _close_position(st, P, exit_px, reason, unit, fee_rate, bar_ts, slip):
    d = 1 if P["side"] == "LONG" else -1
    gross = d * (exit_px - P["entry_px"]) * P["qty"] * unit
    fee = _entry_fee(exit_px, P["qty"], unit, fee_rate)
    pnl = gross - P["entry_fee"] - fee
    st["equity"] += pnl
    st["trades"] += 1
    risk = abs(P["entry_px"] - P["sl0"])
    r = (d * (exit_px - P["entry_px"]) / risk) if risk > 0 else 0.0
    ts = datetime.fromtimestamp(bar_ts, timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    log_trade(dict(time=ts, instrument=st["symbol"], event="EXIT", side=P["side"],
                   entry_px=round(P["entry_px"], 4), exit_px=round(exit_px, 4),
                   qty=P["qty"], reason=reason, pnl=round(pnl, 4),
                   equity_after=round(st["equity"], 2),
                   level=round(P["lvl"], 4), kind=P["kind"], r=round(r, 3)))
    log(f"  [{st['symbol']}] EXIT {P['side']} @ {exit_px:.4f} ({reason}) "
        f"pnl=${pnl:+.2f} R={r:+.2f} | eq ${st['equity']:.2f}")
    if st.get("notify"):
        send_telegram(
            f"🔕 <b>AI BOT — EXIT</b>\n{st['symbol']}\nSide: {P['side']}\n"
            f"Entry: {P['entry_px']:.2f}\nExit: {exit_px:.2f} ({reason})\n"
            f"P&L: <b>${pnl:+.2f}</b>  R={r:+.2f}\nEquity: ${st['equity']:.2f}")
    st["pos"] = None


def _try_fill_pending(st, ctx, i):
    """Resting limit / market-next-open order ko is bar par fill karo."""
    pend = st.get("pending")
    if not pend or st.get("pos"):
        return False
    sym = st["symbol"]
    spec = SPEC[sym]
    unit, tick = spec["unit"], spec["tick"]
    o, h, l, c = ctx["o"], ctx["h"], ctx["l"], ctx["c"]
    bar_ts = int(ctx["ot"][i])
    bars_since = (bar_ts - pend["bar_ts"]) // (ENTRY_TF_MIN * 60)
    d = pend["dir"]
    sl = pend["sl"]

    if pend.get("market"):
        # signal bar ke agle bar ke OPEN par fill (live + backtest same).
        # bars_since >= 1 zaroori hai; agar cron run miss hua to pehle available
        # bar ke open par fill karte hain (max 2 bars purana signal).
        if bars_since < 1:
            return False
        if bars_since > 2:
            log(f"  [{sym}] market signal stale ({bars_since} bars) - drop")
            st["pending"] = None
            return False
        chase = d * (o[i] - pend.get("ref", o[i]))
        if MAX_CHASE_ATR > 0 and chase > MAX_CHASE_ATR * pend["atr"]:
            log(f"  [{sym}] skip: price {chase / pend['atr']:.2f} ATR gap "
                f"(chase limit {MAX_CHASE_ATR})")
            st["pending"] = None
            return False
        fill = o[i] + d * tick
    else:
        if bars_since > PEND_BARS:
            log(f"  [{sym}] pending expire @ {pend['px']:.4f}")
            st["pending"] = None
            return False
        px = pend["px"]
        fill = None
        if d == 1:
            if o[i] <= px:
                fill = o[i]
            elif l[i] <= px:
                fill = px
        else:
            if o[i] >= px:
                fill = o[i]
            elif h[i] >= px:
                fill = px
        if fill is None:
            return False

    if d * (fill - sl) <= 0:
        st["pending"] = None
        return False
    qty = size_position(st["equity"], fill, sl, unit)
    if qty < 1:
        st["pending"] = None
        return False

    tp = fill + d * RR * abs(fill - sl)
    st["pos"] = dict(
        side="LONG" if d == 1 else "SHORT", entry_px=fill, qty=qty,
        sl=round(sl, 6), sl0=round(sl, 6), tp=round(tp, 6),
        atr=float(pend["atr"]), best=fill,
        entry_fee=_entry_fee(fill, qty, unit,
                             spec["taker"] if pend.get("market") else spec["maker"]),
        entry_bar_ts=bar_ts, lvl=float(pend.get("lvl_px") or fill),
        kind=pend.get("kind", "SUP" if d == 1 else "RES"),
        touches=pend["touches"])
    bk = day_key(bar_ts)
    st["day_counts"][bk] = st["day_counts"].get(bk, 0) + 1
    if ONE_PER_LEVEL_DAY and pend.get("key"):
        st["held"].append(pend["key"])
        if len(st["held"]) > 400:
            st["held"] = st["held"][-400:]
    st["pending"] = None
    ts = datetime.fromtimestamp(bar_ts, timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    log_trade(dict(time=ts, instrument=sym, event="ENTRY",
                   side=st["pos"]["side"], entry_px=round(fill, 4),
                   exit_px="", qty=qty, reason=f"S{pend['score']}",
                   pnl="", equity_after=round(st["equity"], 2),
                   level=round(st["pos"]["lvl"], 4), kind=st["pos"]["kind"]))
    log(f"  [{sym}] FILL {st['pos']['side']} {qty} @ {fill:.4f} "
        f"SL {sl:.4f} TP {tp:.4f} score={pend['score']}")
    if st.get("notify"):
        send_telegram(
            f"🎯 <b>AI BOT — ENTRY</b>\n{sym}\n"
            f"Side: {st['pos']['side']}\nEntry: {fill:.2f}\n"
            f"SL: {sl:.2f}  TP: {tp:.2f}\nQty: {qty}\n"
            f"Level: {st['pos']['kind']} @ {st['pos']['lvl']:.2f} "
            f"({st['pos']['touches']} touches)\n"
            f"Score: {pend['score']}/11\n"
            f"Equity: ${st['equity']:.2f}")
    return True


def process_bars(ctx, st, from_ts, verbose=True, to_ts=None):
    """Replay every closed bar with from_ts < ot < to_ts. Idempotent by last_bar."""
    sym = ctx["symbol"]
    spec = SPEC[sym]
    unit, tick = spec["unit"], spec["tick"]
    o, h, l, c = ctx["o"], ctx["h"], ctx["l"], ctx["c"]
    ot, atr, rsi, bias = ctx["ot"], ctx["atr"], ctx["rsi"], ctx["bias"]
    levels = ctx["levels"]
    n = len(ot)
    if to_ts is not None:
        n = int(np.searchsorted(ot, to_ts, side="left"))
    i = int(np.searchsorted(ot, from_ts, side="right"))
    if i >= n:
        return 0

    processed = 0
    while i < n:
        bar_ts = int(ot[i])
        if not st.get("pos") and st.get("day_start") is None:
            st["day_start"] = st["equity"]
            st["day_key"] = day_key(bar_ts)

        # ---------- 0) pending order (market-next-open ya resting limit) ----------
        _try_fill_pending(st, ctx, i)

        # ---------- 1) manage open position ----------
        P = st.get("pos")
        if P:
            d = 1 if P["side"] == "LONG" else -1
            slip = tick
            sl, tp = P["sl"], P["tp"]
            exited = False
            if d == 1:
                if o[i] <= sl:
                    _close_position(st, P, o[i] - slip, "SL-GAP", unit, spec["taker"], bar_ts, slip)
                    exited = True
                elif l[i] <= sl:
                    _close_position(st, P, sl - slip, "SL", unit, spec["taker"], bar_ts, slip)
                    exited = True
                elif h[i] >= tp:
                    _close_position(st, P, tp, "TP", unit, spec["maker"], bar_ts, slip)
                    exited = True
            else:
                if o[i] >= sl:
                    _close_position(st, P, o[i] + slip, "SL-GAP", unit, spec["taker"], bar_ts, slip)
                    exited = True
                elif h[i] >= sl:
                    _close_position(st, P, sl + slip, "SL", unit, spec["taker"], bar_ts, slip)
                    exited = True
                elif l[i] <= tp:
                    _close_position(st, P, tp, "TP", unit, spec["maker"], bar_ts, slip)
                    exited = True

            if not exited:
                held = (bar_ts - P["entry_bar_ts"]) // (ENTRY_TF_MIN * 60)
                if held >= MAX_HOLD_BARS:
                    _close_position(st, P, c[i] - d * slip, "TIME", unit, spec["taker"], bar_ts, slip)
                else:
                    # trail AFTER the close (no look-ahead)
                    R = abs(P["entry_px"] - P["sl0"])
                    if d == 1:
                        P["best"] = max(P["best"], h[i])
                        adv = P["best"] - P["entry_px"]
                        if adv >= BE_AT_R * R:
                            P["sl"] = max(P["sl"], P["entry_px"] + 0.05 * R)
                        if adv >= TRAIL_AT_R * R:
                            P["sl"] = max(P["sl"], P["best"] - ATR_TRAIL * P["atr"])
                    else:
                        P["best"] = min(P["best"], l[i])
                        adv = P["entry_px"] - P["best"]
                        if adv >= BE_AT_R * R:
                            P["sl"] = min(P["sl"], P["entry_px"] - 0.05 * R)
                        if adv >= TRAIL_AT_R * R:
                            P["sl"] = min(P["sl"], P["best"] + ATR_TRAIL * P["atr"])

        # ---------- 2) new setup (flat only) ----------
        P = st.get("pos")
        if not P and not st.get("pending") and np.isfinite(atr[i]) and atr[i] > 0:
            bk = day_key(bar_ts)
            if st.get("day_key") != bk:
                st["day_key"] = bk
                st["day_start"] = st["equity"]
            day_pnl_pct = (st["equity"] / st["day_start"] - 1.0) * 100.0
            trades_today = st["day_counts"].get(bk, 0)
            if (day_pnl_pct > -DAILY_LOSS_PCT and trades_today < MAX_TRADES_DAY
                    and bias[i] != 0):
                sup, res = nearest_levels(levels, i, c[i], atr[i], LEVEL_BAND_ATR)
                setup = None
                if TRIGGER_MODE in ("break", "both"):
                    hit = find_break(levels, i, bias[i], atr[i])
                    if hit:
                        d0, lv0 = hit
                        dist = abs(c[i] - lv0["px"]) / atr[i]
                        sc = score_setup(bias[i], lv0, False, False, rsi[i],
                                         percentile_rank(atr, i), dist,
                                         close_edge=dist, breakout=True)
                        if sc >= SCORE_MIN:
                            setup = (d0, lv0, sc)
                if setup is None and TRIGGER_MODE != "break":
                    if bias[i] == 1 and sup:
                        sweep, rej = detect_trigger(1, o[i], h[i], l[i], c[i],
                                                    sup["px"], atr[i])
                        if sweep or rej:
                            dist = (c[i] - sup["px"]) / atr[i]
                            sc = score_setup(bias[i], sup, sweep, rej, rsi[i],
                                             percentile_rank(atr, i), dist,
                                             close_edge=dist)
                            if sc >= SCORE_MIN:
                                setup = (1, sup, sc)
                    elif bias[i] == -1 and res:
                        sweep, rej = detect_trigger(-1, o[i], h[i], l[i], c[i],
                                                    res["px"], atr[i])
                        if sweep or rej:
                            dist = (res["px"] - c[i]) / atr[i]
                            sc = score_setup(bias[i], res, sweep, rej, rsi[i],
                                             percentile_rank(atr, i), dist,
                                             close_edge=dist)
                            if sc >= SCORE_MIN:
                                setup = (-1, res, sc)

                if setup:
                    d, lv, sc = setup
                    lvl_key = f"{bk}|{sym}|{round(lv['px'], 2)}"
                    lvl_px = float(lv["px"])
                    # structure-based stop (same for both entry modes)
                    if d == 1:
                        sl = min(l[i], lvl_px) - ATR_SL * atr[i]
                    else:
                        sl = max(h[i], lvl_px) + ATR_SL * atr[i]
                    if ENTRY_MODE == "limit":
                        px = lvl_px
                        ref = px
                    else:
                        # market: agle bar ke open par fill
                        px = None
                        ref = float(o[i + 1]) if i + 1 < n else float(c[i])
                    dist = abs(ref - sl)
                    blocked = ONE_PER_LEVEL_DAY and lvl_key in st["held"]
                    if not blocked and MIN_SL_ATR * atr[i] <= dist <= MAX_SL_ATR * atr[i]:
                        st["pending"] = dict(
                            dir=d, px=px, sl=round(sl, 6), atr=float(atr[i]),
                            bar_ts=bar_ts, key=lvl_key, score=sc,
                            touches=lv["touches"], lvl_px=lvl_px,
                            kind="SUP" if d == 1 else "RES",
                            ref=float(c[i]), market=(ENTRY_MODE != "limit"))
                        side_txt = "LONG" if d == 1 else "SHORT"
                        tag = "MARKET" if px is None else f"LIMIT @ {px:.4f}"
                        log(f"  [{sym}] SIGNAL {side_txt} {tag} SL {sl:.4f} "
                            f"score={sc} lvl={lv['kind']}@{lvl_px:.4f} "
                            f"x{lv['touches']}")
                        if st.get("notify"):
                            send_telegram(
                                f"🎯 <b>AI BOT — SIGNAL</b>\n{sym}\n"
                                f"Side: {side_txt}\n"
                                + (f"Entry: market (next open)\n" if px is None
                                   else f"Entry: {px:.2f} (limit)\n")
                                + f"SL: {sl:.4f}\n"
                                f"Level: {lv['kind']} @ {lvl_px:.2f} "
                                f"({lv['touches']} touches)\n"
                                f"Score: {sc}/11\n"
                                f"Equity: ${st['equity']:.2f}")

        st["last_bar"] = bar_ts
        processed += 1
        i += 1
    return processed


# ----------------------------- state / csv -----------------------------
COLLECT = None      # research mode: list collects trades instead of CSV


def fresh_state():
    return dict(
        mode="PAPER",
        instruments={s: dict(symbol=s, equity=START_EQUITY, pos=None, pending=None,
                             trades=0, day_counts={}, held=[],
                             day_key=None, day_start=None,
                             last_bar=0, last_price=None)
                     for s in SYMBOLS},
        last_run=None, last_run_ts=0, last_heartbeat=None,
        last_prices={})


def load_state():
    if not os.path.exists(STATE_FILE):
        return fresh_state()
    with open(STATE_FILE, encoding="utf-8") as f:
        st = json.load(f)
    base = fresh_state()
    for k, v in base.items():
        st.setdefault(k, v)
    for s in SYMBOLS:
        st["instruments"].setdefault(s, base["instruments"][s])
        for k, v in base["instruments"][s].items():
            st["instruments"][s].setdefault(k, v)
    return st


def save_state(st):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(st, f, indent=2)


TRADE_COLS = ["time", "instrument", "event", "side", "entry_px", "exit_px",
              "qty", "reason", "pnl", "equity_after", "level", "kind", "r"]


def log_trade(rec):
    rec = {k: rec.get(k, "") for k in TRADE_COLS}
    if COLLECT is not None:
        COLLECT.append(dict(rec))
        return
    df = pd.DataFrame([rec], columns=TRADE_COLS)
    hdr = not os.path.exists(TRADES_FILE)
    df.to_csv(TRADES_FILE, mode="a", header=hdr, index=False)


# ----------------------------- live -----------------------------
def run_once(notify=False, heartbeat=False):
    st = load_state()
    total_before = sum(v["equity"] for v in st["instruments"].values())
    log(f"=== AI MULTI BOT | PAPER | {len(SYMBOLS)} instruments | entry {ENTRY_TF_MIN}m ===")
    tok, chat = telegram_creds()
    log(f"Telegram: {'ON' if (tok and chat) else 'OFF — secrets/config nahi mile'}")

    for s in SYMBOLS:
        ist = st["instruments"][s]
        ist["notify"] = notify
        try:
            ctx = build_context(s, ENTRY_TF_MIN, bars_entry=1500, bars_bias=700)
            px = fetch_mark(s)
            ist["last_price"] = round(px, 6) if px else None
            if px is None and len(ctx["c"]):
                ist["last_price"] = float(ctx["c"][-1])
            if int(ist.get("last_bar") or 0) == 0:
                # fresh state: purani history replay mat karo, sirf latest
                # closed bar se shuru karo (equity $100 hi rahegi)
                if len(ctx["ot"]) >= 2:
                    ist["last_bar"] = int(ctx["ot"][-2])
                log(f"  [{s}] fresh state -> start from last closed bar")
            n = process_bars(ctx, ist, ist["last_bar"], verbose=True)
            log(f"  [{s}] bars={n} price={ist['last_price']} "
                f"eq=${ist['equity']:.2f} "
                f"pos={ist['pos']['side'] if ist['pos'] else 'FLAT'}")
        except Exception as e:
            log(f"  [{s}] ERROR: {e}")
        ist.pop("notify", None)

    total_after = sum(v["equity"] for v in st["instruments"].values())
    st["last_run"] = now_utc_str()
    st["last_run_ts"] = int(time.time())
    st["last_prices"] = {s: st["instruments"][s]["last_price"] for s in SYMBOLS}
    save_state(st)

    log(f"TOTAL equity: ${total_before:.2f} -> ${total_after:.2f}")
    if notify:
        lines = [f"<b>AI BOT — status</b>", f"Mode: PAPER",
                 f"Total: ${total_after:.2f}"]
        for s in SYMBOLS:
            i2 = st["instruments"][s]
            p = i2["pos"]
            lines.append(f"{s}: ${i2['equity']:.2f} "
                         f"{p['side'] + ' ' + str(p['qty']) if p else 'FLAT'}")
        send_telegram("\n".join(lines))
    if heartbeat:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if st.get("last_heartbeat") != today:
            if send_telegram(f"💓 AI BOT zinda | {today}\nTotal: ${total_after:.2f}\n"
                             + " · ".join(f"{s} ${st['instruments'][s]['equity']:.1f}"
                                          for s in SYMBOLS)):
                st["last_heartbeat"] = today
                save_state(st)
                log("  daily heartbeat bheja")
    return st


# ----------------------------- backtest -----------------------------
def stats_from_trades(trades, start_eq):
    if not trades:
        return dict(trades=0, wr=0.0, pf=0.0, net=0.0, maxdd=0.0, avg_r=0.0,
                    longs=0, shorts=0)
    pnl = np.array([t["pnl"] for t in trades], dtype=float)
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    pf = (wins.sum() / abs(losses.sum())) if losses.sum() != 0 else 999.0
    eq = start_eq + np.cumsum(pnl)
    curve = np.concatenate([[start_eq], eq])
    peak = np.maximum.accumulate(curve)
    dd = (curve - peak) / peak * 100.0
    rs = [t.get("r", 0.0) for t in trades]
    return dict(trades=len(pnl), wr=float((pnl > 0).mean() * 100),
                pf=float(min(pf, 999)), net=float(eq[-1] / start_eq * 100 - 100),
                maxdd=float(dd.min()), avg_r=float(np.mean(rs)),
                longs=int(sum(1 for t in trades if t.get("side") == "LONG")),
                shorts=int(sum(1 for t in trades if t.get("side") == "SHORT")))


def load_ctx(symbol, days):
    bars_e = min(days * (1440 // ENTRY_TF_MIN) + 400, 5000)
    bars_b = min(days * (1440 // (ENTRY_TF_MIN * 4)) + 260, 2000)
    return build_context(symbol, ENTRY_TF_MIN, bars_e, bars_b)


def run_backtest(symbols, days, ctxs=None, warmup=0, end_ts=None, quiet=True):
    """In-memory replay. Returns (per_symbol_stats, combined_stats, trades)."""
    global COLLECT, QUIET
    out, all_trades = {}, []
    prev_quiet = QUIET
    QUIET = quiet
    for s in symbols:
        try:
            ctx = (ctxs or {}).get(s) or load_ctx(s, days)
        except Exception as e:
            log(f"  [{s}] data error: {e}")
            continue
        st = dict(symbol=s, equity=START_EQUITY, pos=None, pending=None, trades=0,
                  day_counts={}, held=[], day_key=None, day_start=None,
                  last_bar=0, notify=False)
        COLLECT = []
        process_bars(ctx, st, warmup, verbose=False, to_ts=end_ts)
        rows = list(COLLECT)
        COLLECT = None
        exits = [r for r in rows if r.get("event") == "EXIT" and "pnl" in r]
        # note: end par khuli position ko mark-to-market nahi karte (sirf
        # closed trades count hote hain) - isliye live aur backtest same hain
        stt = stats_from_trades(exits, START_EQUITY)
        out[s] = stt
        all_trades.extend(exits)
    tot = stats_from_trades(all_trades, START_EQUITY * max(1, len(out)))
    QUIET = prev_quiet
    return out, tot, all_trades


def backtest(symbols, days):
    log(f"=== BACKTEST | {days} din | entry {ENTRY_TF_MIN}m + bias {ENTRY_TF_MIN*4}m ===")
    per, tot, _ = run_backtest(symbols, days)
    for s, stt in per.items():
        log(f"  {s}: trades={stt['trades']} WR={stt['wr']:.1f}% PF={stt['pf']:.2f} "
            f"net={stt['net']:+.1f}% maxDD={stt['maxdd']:.1f}% "
            f"avgR={stt['avg_r']:+.2f} L/S={stt['longs']}/{stt['shorts']}")
    log(f"\nCOMBINED: trades={tot['trades']} WR={tot['wr']:.1f}% "
        f"PF={tot['pf']:.2f} net={tot['net']:+.1f}% maxDD={tot['maxdd']:.1f}% "
        f"avgR={tot['avg_r']:+.2f}")


# ----------------------------- main -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--loop", type=int, default=0, metavar="SEC")
    ap.add_argument("--notify", action="store_true")
    ap.add_argument("--heartbeat", action="store_true")
    ap.add_argument("--test-telegram", action="store_true")
    ap.add_argument("--backtest", action="store_true")
    ap.add_argument("--days", type=int, default=120)
    ap.add_argument("--tf", type=int, default=60, choices=[15, 60, 240],
                    help="entry timeframe (bias = 4x; 240 par bias 16h unsupported hai)")
    ap.add_argument("--symbols", default=",".join(SYMBOLS))
    a = ap.parse_args()

    global ENTRY_TF_MIN
    ENTRY_TF_MIN = a.tf
    if ENTRY_TF_MIN * 4 not in RES_MAP:
        raise SystemExit(f"--tf {ENTRY_TF_MIN} ke liye bias timeframe "
                         f"{ENTRY_TF_MIN*4}m supported nahi (15 ya 60 use karo)")
    syms = [s.strip().upper() for s in a.symbols.split(",") if s.strip()]
    for s in syms:
        if s not in SPEC:
            raise SystemExit(f"unknown symbol {s}")

    if a.test_telegram:
        ok = send_telegram("✅ AI BOT test — Telegram working!")
        print("sent" if ok else "nahi gaya — token/chat check karo")
        return

    if a.backtest:
        backtest(syms, a.days)
        return

    if a.loop:
        while True:
            try:
                run_once(notify=a.notify, heartbeat=True)
            except Exception as e:
                log(f"ERROR: {e}")
            time.sleep(a.loop)
    else:
        run_once(notify=a.notify, heartbeat=a.heartbeat)


if __name__ == "__main__":
    main()

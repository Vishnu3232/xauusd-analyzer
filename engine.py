"""XAUUSD analysis engine.

A rule-based implementation of the analyze-gold / pair-analyze pipeline:
regime -> news gate -> structure -> liquidity -> entry refinement -> risk -> plan.

Data: Yahoo Finance public chart endpoint (gold futures GC=F, dollar index)
and the Forex Factory weekly calendar. Standard library only.
"""
from __future__ import annotations

import json
import threading
import time
import urllib.request
from datetime import datetime, timezone

GOLD = "GC=F"
DXY = "DX-Y.NYB"
CALENDAR_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
UA = {"User-Agent": "Mozilla/5.0"}

# timeframe -> (yahoo interval, yahoo range, seconds per bar)
FEEDS = {
    "M5": ("5m", "5d", 300),
    "M15": ("15m", "1mo", 900),
    "H1": ("60m", "3mo", 3600),
    "H4": ("60m", "3mo", 14400),
    "D1": ("1d", "1y", 86400),
    "W1": ("1wk", "5y", 604800),
}
HIGHER_TF = {"M5": "H1", "M15": "H1", "H1": "H4", "H4": "D1", "D1": "W1"}

# Mode table from the analyze-gold skill (gold-adjusted risk).
MODES = {
    "conservative": {"risk_pct": 0.5, "min_rr": 3.0, "grades": ["A"], "tfs": ["H4", "D1"]},
    "standard": {"risk_pct": 1.0, "min_rr": 2.0, "grades": ["A", "B"], "tfs": ["H1", "H4"]},
    "aggressive": {"risk_pct": 1.5, "min_rr": 1.5, "grades": ["A", "B", "C"], "tfs": ["H1"]},
    "scalp": {"risk_pct": 0.5, "min_rr": 1.5, "grades": ["A"], "tfs": ["M5", "M15"]},
    "swing": {"risk_pct": 1.0, "min_rr": 3.0, "grades": ["A", "B"], "tfs": ["H4", "D1"]},
}

USD_PER_POINT_PER_LOT = 100.0  # 1.00 move x 100 oz contract
MIN_SL_POINTS = 2.5
# The skill's volatility bands (H1 ATR 12 = top of normal, 20 = stand aside)
# were written with gold near 2345. They are kept as a share of price so they
# still mean the same thing at other price levels.
ATR_ELEVATED_PCT = 12 / 2345 * 100
ATR_HIGH_PCT = 20 / 2345 * 100

_cache: dict = {}
_lock = threading.Lock()


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def _get_json(url: str):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.load(resp)


def _cached(key: str, ttl: float, loader):
    """Return (value, is_stale). Serves the previous value if a refresh fails."""
    now = time.time()
    with _lock:
        hit = _cache.get(key)
    if hit and now - hit[0] < ttl:
        return hit[1], False
    try:
        value = loader()
    except Exception:
        if hit:
            return hit[1], True
        raise
    with _lock:
        _cache[key] = (now, value)
    return value, False


def _bucket(ts: int, tf: str) -> int:
    if tf == "W1":  # weeks start Monday; the epoch started on a Thursday
        return ts - ((ts - 4 * 86400) % 604800)
    step = FEEDS[tf][2]
    return ts - ts % step


def _load_bars(symbol: str, tf: str):
    interval, rng, _ = FEEDS[tf]
    last_error = None
    for host in ("query1", "query2"):
        url = (f"https://{host}.finance.yahoo.com/v8/finance/chart/{symbol}"
               f"?interval={interval}&range={rng}")
        try:
            result = _get_json(url)["chart"]["result"][0]
            break
        except Exception as exc:  # noqa: BLE001 - try the next host
            last_error = exc
    else:
        raise RuntimeError(f"price feed unavailable: {last_error}")

    quote = result["indicators"]["quote"][0]
    merged: dict[int, dict] = {}
    for i, ts in enumerate(result.get("timestamp") or []):
        o, h, l, c = quote["open"][i], quote["high"][i], quote["low"][i], quote["close"][i]
        if None in (o, h, l, c):
            continue
        key = _bucket(ts, tf)
        bar = merged.get(key)
        if bar is None:
            merged[key] = {"t": key, "o": o, "h": h, "l": l, "c": c}
        else:
            bar["h"], bar["l"], bar["c"] = max(bar["h"], h), min(bar["l"], l), c
    bars = [merged[k] for k in sorted(merged)]
    if len(bars) < 60:
        raise RuntimeError(f"not enough {tf} bars for {symbol} ({len(bars)})")
    meta = result.get("meta", {})
    return {
        "bars": bars,
        "price": meta.get("regularMarketPrice") or bars[-1]["c"],
        "quote_time": meta.get("regularMarketTime") or bars[-1]["t"],
        "name": meta.get("shortName") or symbol,
    }


def get_bars(symbol: str, tf: str):
    return _cached(f"bars:{symbol}:{tf}", 60, lambda: _load_bars(symbol, tf))


def get_calendar():
    def load():
        events = []
        for e in _get_json(CALENDAR_URL):
            if e.get("country") != "USD" or e.get("impact") != "High":
                continue
            when = datetime.fromisoformat(e["date"]).astimezone(timezone.utc)
            events.append({"title": e.get("title", ""), "time": int(when.timestamp()),
                           "forecast": e.get("forecast", ""), "previous": e.get("previous", "")})
        return sorted(events, key=lambda x: x["time"])
    return _cached("calendar", 3600, load)


# --------------------------------------------------------------------------
# indicators
# --------------------------------------------------------------------------
def atr(bars, n: int = 14) -> float:
    trs = [max(b["h"] - b["l"], abs(b["h"] - bars[i]["c"]), abs(b["l"] - bars[i]["c"]))
           for i, b in enumerate(bars[1:])]
    value = sum(trs[:n]) / n
    for tr in trs[n:]:
        value = (value * (n - 1) + tr) / n
    return value


def ema(values, n: int) -> float:
    k, e = 2 / (n + 1), values[0]
    for v in values[1:]:
        e = v * k + e * (1 - k)
    return e


def swings(bars, wing: int = 2):
    """Fractal swing points: a high/low that beats `wing` bars on each side."""
    out = []
    for i in range(wing, len(bars) - wing):
        window = bars[i - wing:i + wing + 1]
        if bars[i]["h"] == max(b["h"] for b in window):
            out.append({"i": i, "t": bars[i]["t"], "kind": "H", "price": bars[i]["h"]})
        if bars[i]["l"] == min(b["l"] for b in window):
            out.append({"i": i, "t": bars[i]["t"], "kind": "L", "price": bars[i]["l"]})
    return out


def structure(bars):
    """Label market structure from the last two swing highs and lows."""
    sw = swings(bars)
    highs = [s for s in sw if s["kind"] == "H"]
    lows = [s for s in sw if s["kind"] == "L"]
    if len(highs) < 2 or len(lows) < 2:
        return {"label": "mixed", "text": "not enough swing points"}
    higher_high = highs[-1]["price"] > highs[-2]["price"]
    higher_low = lows[-1]["price"] > lows[-2]["price"]
    close = bars[-1]["c"]
    if higher_high and higher_low:
        label, text = "bullish", "higher highs and higher lows"
    elif not higher_high and not higher_low:
        label, text = "bearish", "lower highs and lower lows"
    else:
        label, text = "mixed", "swing highs and lows disagree"
    if close < lows[-1]["price"]:
        if label == "bullish":
            label, text = "mixed", f"up-structure broken below {lows[-1]['price']:.2f}"
        else:
            label, text = "bearish", f"{text}; close below last swing low {lows[-1]['price']:.2f}"
    elif close > highs[-1]["price"]:
        if label == "bearish":
            label, text = "mixed", f"down-structure broken above {highs[-1]['price']:.2f}"
        else:
            label, text = "bullish", f"{text}; close above last swing high {highs[-1]['price']:.2f}"
    return {"label": label, "text": text,
            "last_high": highs[-1]["price"], "last_low": lows[-1]["price"]}


def fair_value_gaps(bars, min_size: float, lookback: int = 80):
    """Three-candle gaps whose midpoint has not been traded back into."""
    gaps = []
    n = len(bars)
    for i in range(max(1, n - lookback), n - 2):  # n-2: skip the forming bar
        before, after = bars[i - 1], bars[i + 1]
        if before["l"] - after["h"] >= min_size:
            kind, bottom, top = "bearish", after["h"], before["l"]
        elif after["l"] - before["h"] >= min_size:
            kind, bottom, top = "bullish", before["h"], after["l"]
        else:
            continue
        mid = (top + bottom) / 2
        later = bars[i + 2:]
        if kind == "bearish":
            filled = any(b["h"] >= mid for b in later)
            anchor = max(b["h"] for b in bars[i - 1:i + 2])
        else:
            filled = any(b["l"] <= mid for b in later)
            anchor = min(b["l"] for b in bars[i - 1:i + 2])
        if not filled:
            gaps.append({"kind": kind, "top": top, "bottom": bottom, "mid": mid,
                         "anchor": anchor, "t": bars[i]["t"]})
    return gaps


# --------------------------------------------------------------------------
# pipeline
# --------------------------------------------------------------------------
def _step(key, title, status, summary, details=None):
    return {"key": key, "title": title, "status": status, "summary": summary,
            "details": details or []}


def _utc(ts: int) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%a %d %b %H:%M UTC")


def _day_range(bars, day):
    rows = [b for b in bars if datetime.fromtimestamp(b["t"], timezone.utc).date() == day]
    if not rows:
        return None
    return {"high": max(b["h"] for b in rows), "low": min(b["l"] for b in rows)}


def _dedupe(levels, tolerance):
    kept = []
    for lvl in levels:
        if all(abs(lvl["price"] - k["price"]) > tolerance for k in kept):
            kept.append(lvl)
    return kept


def analyze(tf: str = "H1", mode: str = "standard"):
    tf = tf.upper()
    mode = mode.lower()
    if mode not in MODES:
        raise ValueError(f"unknown mode '{mode}'")
    if tf not in HIGHER_TF:
        raise ValueError(f"unknown timeframe '{tf}'")
    cfg = MODES[mode]
    htf_name = HIGHER_TF[tf]
    now = datetime.now(timezone.utc)
    now_ts = int(now.timestamp())

    feed, stale_tf = get_bars(GOLD, tf)
    htf_feed, _ = get_bars(GOLD, htf_name)
    h1_feed, _ = get_bars(GOLD, "H1")
    bars, htf, h1 = feed["bars"], htf_feed["bars"], h1_feed["bars"]
    price = float(feed["price"])
    quote_age_min = max(0, (now_ts - int(feed["quote_time"])) / 60)

    atr_tf = atr(bars)
    atr_h1 = atr(h1)
    atr_h1_pct = atr_h1 / price * 100
    steps, blocks, warnings = [], [], []

    # ---- 0. data feed ----------------------------------------------------
    if stale_tf or quote_age_min > 90:
        status = "warn"
        summary = (f"Last quote {quote_age_min:.0f} min old - market closed or feed delayed."
                   if quote_age_min > 90 else "Feed refresh failed - showing the last good data.")
        warnings.append("Price data is not live right now.")
    else:
        status, summary = "pass", f"Live - last quote {quote_age_min:.0f} min ago."
    steps.append(_step("data", "Data feed", status, summary,
                       [f"Source: Yahoo Finance {GOLD} ({feed['name']}), gold futures - not broker spot.",
                        "Yahoo futures quotes run about 10 minutes behind the market.",
                        f"{len(bars)} {tf} bars, {len(htf)} {htf_name} bars."]))

    if tf not in cfg["tfs"]:
        blocks.append(f"{mode} mode is only defined for {', '.join(cfg['tfs'])}.")

    # ---- 1. regime -------------------------------------------------------
    htf_closes = [b["c"] for b in htf]
    e20, e50 = ema(htf_closes, 20), ema(htf_closes, 50)
    htf_close = htf_closes[-1]
    if htf_close < e20 < e50:
        regime = "trend-down"
    elif htf_close > e20 > e50:
        regime = "trend-up"
    else:
        regime = "range"
    hour = now.hour
    if 12 <= hour < 16:
        session = "London-NY overlap"
    elif 8 <= hour < 12:
        session = "London"
    elif 16 <= hour < 21:
        session = "New York"
    else:
        session = "Asia / off-hours"
    regime_details = [
        f"{htf_name} close {htf_close:.2f} vs EMA20 {e20:.2f} / EMA50 {e50:.2f}.",
        f"H1 ATR(14) {atr_h1:.1f} pts = {atr_h1_pct:.2f}% of price "
        f"(elevated above {ATR_ELEVATED_PCT:.2f}%, stand aside above {ATR_HIGH_PCT:.2f}%).",
        f"Session: {session}.",
    ]
    regime_status = "pass"
    if atr_h1_pct > ATR_HIGH_PCT:
        regime_status = "block"
        blocks.append(f"Volatility too high: H1 ATR {atr_h1:.1f} pts.")
    elif atr_h1_pct > ATR_ELEVATED_PCT:
        regime_status = "warn"
        warnings.append(f"Volatility elevated: H1 ATR {atr_h1:.1f} pts.")
    if session == "Asia / off-hours":
        regime_status = "warn" if regime_status == "pass" else regime_status
        warnings.append("Outside the London-New York window (08:00-21:00 UTC); gold liquidity is thin.")
    if mode == "scalp" and session != "London-NY overlap":
        regime_status = "block"
        blocks.append("Scalp mode is for the London-NY overlap only (12:00-16:00 UTC).")
    steps.append(_step("regime", "Regime", regime_status,
                       f"{htf_name} {regime.replace('-', ' ')}; {session}.", regime_details))

    # ---- 2. news gate ----------------------------------------------------
    news = {"events": [], "next_event": None, "blocked": False, "available": True}
    try:
        events, _ = get_calendar()
        news["events"] = [dict(e, when=_utc(e["time"])) for e in events]
        active = [e for e in events if e["time"] - 7200 <= now_ts <= e["time"] + 3600]
        upcoming = [e for e in events if e["time"] > now_ts]
        if upcoming:
            news["next_event"] = dict(upcoming[0], when=_utc(upcoming[0]["time"]))
        if active:
            news["blocked"] = True
            blocks.append(f"News gate: {active[0]['title']} at {_utc(active[0]['time'])}.")
            steps.append(_step("news", "News gate", "block",
                               f"Blocked by {active[0]['title']} ({_utc(active[0]['time'])}).",
                               ["No plan from 2 h before to 1 h after a high-impact USD release."]))
        else:
            nxt = news["next_event"]
            steps.append(_step("news", "News gate", "pass",
                               f"Clear. Next: {nxt['title']}, {nxt['when']}." if nxt
                               else "Clear. No more high-impact USD releases this week.",
                               ["Source: Forex Factory weekly calendar, high-impact USD events only.",
                                "Unscheduled headlines and Fed speakers are not covered."]))
    except Exception:  # noqa: BLE001 - the calendar is optional
        news["available"] = False
        warnings.append("Economic calendar unavailable - check for USD news yourself.")
        steps.append(_step("news", "News gate", "warn", "Calendar unavailable - check manually."))

    # ---- 3. structure ----------------------------------------------------
    st = structure(bars)
    if regime == "trend-down":
        bias = "flat" if st["label"] == "bullish" else "short"
    elif regime == "trend-up":
        bias = "flat" if st["label"] == "bearish" else "long"
    else:
        bias = {"bearish": "short", "bullish": "long"}.get(st["label"], "flat")
    gaps = fair_value_gaps(bars, min_size=0.15 * atr_tf)
    wanted = "bearish" if bias == "short" else "bullish"
    if bias == "short":
        zone_candidates = sorted((g for g in gaps if g["kind"] == wanted and g["mid"] > price),
                                 key=lambda g: g["mid"])
    elif bias == "long":
        zone_candidates = sorted((g for g in gaps if g["kind"] == wanted and g["mid"] < price),
                                 key=lambda g: -g["mid"])
    else:
        zone_candidates = []
    zone = zone_candidates[0] if zone_candidates else None
    struct_details = [f"{tf}: {st['text']}."]
    if zone:
        struct_details.append(f"Nearest unfilled {wanted} gap: {zone['bottom']:.2f} - {zone['top']:.2f}.")
    else:
        struct_details.append("No unfilled fair value gap on the trade side.")
    if bias == "flat":
        blocks.append(f"No directional bias: {htf_name} is {regime.replace('-', ' ')} "
                      f"but {tf} structure is {st['label']}.")
        steps.append(_step("structure", "Structure", "block",
                           f"{tf} structure {st['label']} - no agreement with {htf_name}.", struct_details))
    else:
        steps.append(_step("structure", "Structure", "pass",
                           f"{tf} structure {st['label']}; bias {bias}.", struct_details))

    # ---- 4. liquidity ----------------------------------------------------
    today = datetime.fromtimestamp(bars[-1]["t"], timezone.utc).date()
    prior_days = sorted({datetime.fromtimestamp(b["t"], timezone.utc).date() for b in h1} - {today})
    today_rng = _day_range(h1, today)
    prev_rng = _day_range(h1, prior_days[-1]) if prior_days else None
    pools = []
    for name, series in ((tf, bars), (htf_name, htf)):
        for s in swings(series)[-24:]:
            pools.append({"price": s["price"], "side": "high" if s["kind"] == "H" else "low",
                          "label": f"{name} swing {'high' if s['kind'] == 'H' else 'low'}", "t": s["t"]})
    if prev_rng:
        pools.append({"price": prev_rng["high"], "side": "high", "label": "Previous day high", "t": 0})
        pools.append({"price": prev_rng["low"], "side": "low", "label": "Previous day low", "t": 0})
    if today_rng:
        pools.append({"price": today_rng["high"], "side": "high", "label": "Today high", "t": 0})
        pools.append({"price": today_rng["low"], "side": "low", "label": "Today low", "t": 0})
    tol = 0.2 * atr_tf
    above = _dedupe(sorted((p for p in pools if p["side"] == "high" and p["price"] > price),
                           key=lambda p: p["price"]), tol)[:5]
    below = _dedupe(sorted((p for p in pools if p["side"] == "low" and p["price"] < price),
                           key=lambda p: -p["price"]), tol)[:5]

    swept = []
    if today_rng:
        for s in swings(htf)[-30:]:
            if datetime.fromtimestamp(s["t"], timezone.utc).date() >= today:
                continue
            if bias == "short" and s["kind"] == "L" and today_rng["low"] < s["price"] < price:
                swept.append(s["price"])
            if bias == "long" and s["kind"] == "H" and today_rng["high"] > s["price"] > price:
                swept.append(s["price"])
    liq_details = []
    if above:
        liq_details.append("Above: " + ", ".join(f"{p['price']:.2f} ({p['label']})" for p in above[:3]) + ".")
    if below:
        liq_details.append("Below: " + ", ".join(f"{p['price']:.2f} ({p['label']})" for p in below[:3]) + ".")
    if swept:
        side = "below" if bias == "short" else "above"
        text = (f"Price ran stops {side} {', '.join(f'{p:.2f}' for p in sorted(set(swept)))} today "
                f"and came back - possible reversal.")
        warnings.append(text)
        steps.append(_step("liquidity", "Liquidity", "warn", text, liq_details))
    else:
        steps.append(_step("liquidity", "Liquidity", "pass",
                           "No stop-run against the trade direction today.", liq_details))

    # ---- DXY -------------------------------------------------------------
    dxy_ctx = {"bias": "unknown", "price": None}
    try:
        dxy_feed, _ = get_bars(DXY, "H1")
        d = dxy_feed["bars"]
        d_close = d[-1]["c"]
        d_ema = ema([b["c"] for b in d], 20)
        d_prev = d[-7]["c"]
        if d_close > d_ema and d_close > d_prev:
            dxy_ctx["bias"] = "strengthening"
        elif d_close < d_ema and d_close < d_prev:
            dxy_ctx["bias"] = "weakening"
        else:
            dxy_ctx["bias"] = "flat"
        dxy_ctx["price"] = round(d_close, 3)
        dxy_ctx["change_6h"] = round(d_close - d_prev, 3)
    except Exception:  # noqa: BLE001 - DXY is supporting context only
        pass
    dxy_supports = ((bias == "short" and dxy_ctx["bias"] == "strengthening")
                    or (bias == "long" and dxy_ctx["bias"] == "weakening"))
    dxy_opposes = ((bias == "short" and dxy_ctx["bias"] == "weakening")
                   or (bias == "long" and dxy_ctx["bias"] == "strengthening"))
    if dxy_opposes:
        warnings.append(f"Dollar index is {dxy_ctx['bias']} - that works against a gold {bias}.")

    # ---- 5. entry refinement + 6. risk ----------------------------------
    plan = None
    if bias != "flat":
        sign = -1 if bias == "short" else 1  # direction of profit
        buffer = 0.15 * atr_tf
        entry_source = None
        if zone:
            entry = zone["mid"]
            stop_anchor = zone["anchor"]
            entry_source = f"midpoint of the {tf} fair value gap"
        else:
            sw = swings(bars)
            kind = "H" if bias == "short" else "L"
            pivots = [s for s in sw if s["kind"] == kind]
            entry = None
            if pivots:
                pivot = pivots[-1]
                leg = bars[pivot["i"]:]
                extreme = min(b["l"] for b in leg) if bias == "short" else max(b["h"] for b in leg)
                mid = (pivot["price"] + extreme) / 2
                if (bias == "short" and mid > price) or (bias == "long" and mid < price):
                    entry, stop_anchor = mid, pivot["price"]
                    entry_source = f"50% retracement of the last {tf} leg (no gap available)"
        if entry is None:
            blocks.append("No clean pullback entry: price is already past the 50% retracement.")
            steps.append(_step("entry", "Entry refinement", "block",
                               "No unfilled gap and no 50% retracement left to sell/buy into."))
        else:
            structural = abs(stop_anchor - entry) + buffer
            sl_points = max(structural, atr_h1, MIN_SL_POINTS)
            stop = entry - sign * sl_points
            tp1 = entry + sign * cfg["min_rr"] * sl_points
            recent = bars[-50:]
            extreme = min(b["l"] for b in recent) if bias == "short" else max(b["h"] for b in recent)
            room_ok = tp1 >= extreme if bias == "short" else tp1 <= extreme
            side_pools = below if bias == "short" else above
            beyond = [p for p in side_pools
                      if (tp1 - p["price"]) * (1 if bias == "short" else -1) > 0.25 * atr_tf]
            if beyond:
                tp2, tp2_basis = beyond[0]["price"], beyond[0]["label"]
            else:
                tp2 = entry + sign * (cfg["min_rr"] + 1) * sl_points
                tp2_basis = "measured, no structure level there"
            in_zone = (price >= zone["bottom"] if bias == "short" else price <= zone["top"]) if zone \
                else abs(price - entry) <= 0.25 * atr_tf
            distance = abs(entry - price)

            checks = [
                {"name": f"{htf_name} trend agrees", "ok": regime != "range"},
                {"name": f"{tf} structure agrees", "ok": st["label"] == wanted},
                {"name": "Fair value gap to enter in", "ok": zone is not None},
                {"name": "Dollar index agrees", "ok": dxy_supports},
                {"name": "No stop-run reversal signal today", "ok": not swept},
                {"name": "Target inside the recent range", "ok": room_ok},
            ]
            score = sum(c["ok"] for c in checks)
            grade = "A" if score == 6 else "B" if score >= 4 else "C" if score == 3 else "D"
            past_target = price <= tp1 if bias == "short" else price >= tp1
            if past_target:
                blocks.append(f"Price is already past target 1 ({tp1:.2f}) for this entry - "
                              f"no setup until a fresh pullback forms.")
            if not room_ok:
                warnings.append(f"Target 1 needs a new {'low' if bias == 'short' else 'high'} beyond "
                                f"{extreme:.2f}.")
            if grade not in cfg["grades"]:
                blocks.append(f"Setup grade {grade} ({score}/6 checks) - {mode} mode needs "
                              f"{' or '.join(cfg['grades'])}.")
            steps.append(_step(
                "entry", "Entry refinement", "pass" if in_zone else "warn",
                (f"Price is in the entry zone - wait for a rejection candle." if in_zone
                 else f"Entry {entry:.2f} is {distance:.1f} pts away - wait for the pullback."),
                [f"Entry at the {entry_source}.",
                 "A rejection candle on a lower timeframe is required before entering."]))
            steps.append(_step(
                "risk", "Risk & sizing", "pass" if grade in cfg["grades"] else "block",
                f"Stop {sl_points:.1f} pts, target {cfg['min_rr']:.1f}R, grade {grade} ({score}/6).",
                [f"Stop = larger of structure ({structural:.1f}), 1x H1 ATR ({atr_h1:.1f}), "
                 f"{MIN_SL_POINTS} pts minimum.",
                 f"{mode} mode: {cfg['risk_pct']}% risk per trade, minimum R:R {cfg['min_rr']}."]))
            plan = {
                "bias": bias,
                "entry": round(entry, 2),
                "stop_loss": round(stop, 2),
                "take_profit_1": round(tp1, 2),
                "take_profit_2": round(tp2, 2),
                "tp2_basis": tp2_basis,
                "sl_points": round(sl_points, 2),
                "risk_reward": cfg["min_rr"],
                "risk_reward_tp2": round(abs(tp2 - entry) / sl_points, 2),
                "setup_grade": grade,
                "score": score,
                "checks": checks,
                "entry_source": entry_source,
                "in_zone": in_zone,
                "distance": round(distance, 2),
                "zone": ({"top": round(zone["top"], 2), "bottom": round(zone["bottom"], 2)}
                         if zone else None),
                "invalidation": (f"{tf} close {'above' if bias == 'short' else 'below'} "
                                 f"{stop:.2f}"),
            }

    # ---- 7. synthesis ----------------------------------------------------
    if blocks:
        verdict, headline = "NO_TRADE", "No trade"
    elif plan["in_zone"]:
        verdict = "IN_ZONE"
        headline = f"{bias.capitalize()} zone reached - wait for rejection"
    else:
        verdict = "WAIT"
        headline = f"Wait for pullback, then {bias}"
    steps.append(_step("synthesis", "Synthesis",
                       "block" if blocks else "pass",
                       headline + ("." if not blocks else f": {blocks[0]}"),
                       blocks[1:] if blocks else []))

    return {
        "symbol": "XAUUSD",
        "feed_symbol": GOLD,
        "feed_name": feed["name"],
        "timeframe": tf,
        "higher_timeframe": htf_name,
        "mode": mode,
        "generated_at": now_ts,
        "quote_time": int(feed["quote_time"]),
        "price": round(price, 2),
        "regime": regime,
        "bias": bias,
        "verdict": verdict,
        "headline": headline,
        "blocks": blocks,
        "warnings": warnings,
        "plan": plan,
        "steps": steps,
        "news": news,
        "liquidity": {
            "above": [{"price": round(p["price"], 2), "label": p["label"]} for p in above],
            "below": [{"price": round(p["price"], 2), "label": p["label"]} for p in below],
        },
        "gold_context": {
            "atr_h1": round(atr_h1, 2),
            "atr_tf": round(atr_tf, 2),
            "atr_h1_pct": round(atr_h1_pct, 3),
            "session": session,
            "dxy": dxy_ctx,
            "today": ({k: round(v, 2) for k, v in today_rng.items()} if today_rng else None),
        },
        "sizing": {"risk_pct": cfg["risk_pct"], "usd_per_point_per_lot": USD_PER_POINT_PER_LOT},
        "modes": {m: {"risk_pct": c["risk_pct"], "min_rr": c["min_rr"], "tfs": c["tfs"]}
                  for m, c in MODES.items()},
        "bars": [{"time": b["t"], "open": round(b["o"], 2), "high": round(b["h"], 2),
                  "low": round(b["l"], 2), "close": round(b["c"], 2)} for b in bars[-400:]],
    }

"""Walk-forward backtest of the engine's trade plans.

At every closed bar the pipeline is run on the data available up to that
bar only. If it produces a plan, a limit order sits at the entry for the
next bar; a filled trade is then tracked to its stop, its first target, or
a time limit.
"""
from __future__ import annotations

import bisect
import threading
import time
from datetime import datetime, timezone

import engine

# timeframe -> (history feed, bars to warm up, max bars a trade may stay open)
SUPPORTED = {
    "M15": ("M15_LONG", 400, 96),
    "H1": ("H1_LONG", 300, 48),
    "H4": ("H1_LONG", 1200, 30),
}
COST_POINTS = 0.30       # spread paid per trade
WINDOW = 320             # bars of the trade timeframe given to the pipeline
HTF_WINDOW = 160
H1_WINDOW = 200
RESULT_TTL = 3600

_jobs: dict = {}
_lock = threading.Lock()


def _resampler(tf: str):
    """Incrementally roll base bars up into `tf` bars (last one still forming)."""
    out: list = []

    def push(bar):
        key = engine._bucket(bar["t"], tf)
        if out and out[-1]["t"] == key:
            last = out[-1]
            last["h"], last["l"], last["c"] = max(last["h"], bar["h"]), min(last["l"], bar["l"]), bar["c"]
        else:
            out.append({"t": key, "o": bar["o"], "h": bar["h"], "l": bar["l"], "c": bar["c"]})
    return out, push


def _iso(ts: int) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d %H:%M")


def _stats(trades):
    n = len(trades)
    if not n:
        return {"trades": 0}
    rs = [t["r"] for t in trades]
    wins = [r for r in rs if r > 0]
    losses = [r for r in rs if r <= 0]
    equity, peak, drawdown = 0.0, 0.0, 0.0
    for r in rs:
        equity += r
        peak = max(peak, equity)
        drawdown = max(drawdown, peak - equity)
    streak = worst_streak = 0
    for r in rs:
        streak = streak + 1 if r <= 0 else 0
        worst_streak = max(worst_streak, streak)
    gross_loss = -sum(losses)
    return {
        "trades": n,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / n * 100, 1),
        "avg_r": round(sum(rs) / n, 3),
        "total_r": round(sum(rs), 2),
        "profit_factor": round(sum(wins) / gross_loss, 2) if gross_loss > 0 else None,
        "max_drawdown_r": round(drawdown, 2),
        "worst_losing_streak": worst_streak,
    }


def run(tf: str, mode: str, progress=None):
    tf, mode = engine.check_args(tf, mode)
    if tf not in SUPPORTED:
        raise ValueError(f"backtest supports {', '.join(SUPPORTED)} only - {tf} has too little history")
    if tf not in engine.MODES[mode]["tfs"]:
        raise ValueError(f"{mode} mode is not defined for {tf}")
    feed_key, warmup, max_hold = SUPPORTED[tf]
    base = engine.get_bars(engine.GOLD, feed_key)[0]["bars"]
    base_step = engine.FEEDS[feed_key][2]
    try:
        dxy_all = engine.get_bars(engine.DXY, "H1_LONG")[0]["bars"]
    except Exception:  # noqa: BLE001 - run without the dollar check
        dxy_all = []
    dxy_times = [b["t"] for b in dxy_all]

    htf_name = engine.HIGHER_TF[tf]
    tf_is_base = engine.FEEDS[tf][2] == base_step
    tf_bars, push_tf = _resampler(tf)
    htf_bars, push_htf = _resampler(htf_name)
    h1_bars, push_h1 = _resampler("H1")
    tf_step = engine.FEEDS[tf][2]

    trades, order, position = [], None, None
    signals = evaluated = 0
    total = len(base)

    for i, bar in enumerate(base):
        # --- 1. act on this bar with what was decided before it -----------
        if position:
            position["held"] += 1
            short = position["bias"] == "short"
            hit_stop = bar["h"] >= position["stop"] if short else bar["l"] <= position["stop"]
            hit_target = bar["l"] <= position["target"] if short else bar["h"] >= position["target"]
            exit_price = reason = None
            if hit_stop:  # stop is checked first: worst case when both are touched
                exit_price, reason = position["stop"], "stop"
            elif hit_target:
                exit_price, reason = position["target"], "target"
            elif position["held"] >= max_hold * (tf_step // base_step):
                exit_price, reason = bar["c"], "time"
            if reason:
                move = (position["fill"] - exit_price) if short else (exit_price - position["fill"])
                trades.append({
                    "opened": _iso(position["opened"]), "closed": _iso(bar["t"] + base_step),
                    "closed_ts": bar["t"] + base_step, "bias": position["bias"],
                    "grade": position["grade"], "entry": round(position["fill"], 2),
                    "stop": round(position["stop"], 2), "target": round(position["target"], 2),
                    "exit": round(exit_price, 2), "result": reason,
                    "r": round((move - COST_POINTS) / position["risk"], 3),
                })
                position = None
        elif order:
            short = order["bias"] == "short"
            reached = bar["h"] >= order["entry"] if short else bar["l"] <= order["entry"]
            if reached:
                fill = max(order["entry"], bar["o"]) if short else min(order["entry"], bar["o"])
                gapped_past_stop = fill >= order["stop"] if short else fill <= order["stop"]
                if not gapped_past_stop:
                    position = {"bias": order["bias"], "fill": fill, "stop": order["stop"],
                                "target": order["target"], "risk": order["risk"],
                                "grade": order["grade"], "opened": bar["t"], "held": 0}
                    stopped = bar["h"] >= position["stop"] if short else bar["l"] <= position["stop"]
                    if stopped:  # filled and stopped inside the same bar
                        trades.append({
                            "opened": _iso(bar["t"]), "closed": _iso(bar["t"] + base_step),
                            "closed_ts": bar["t"] + base_step, "bias": position["bias"],
                            "grade": position["grade"], "entry": round(fill, 2),
                            "stop": round(position["stop"], 2), "target": round(position["target"], 2),
                            "exit": round(position["stop"], 2), "result": "stop",
                            "r": round((-abs(position["stop"] - fill) - COST_POINTS) / position["risk"], 3),
                        })
                        position = None
        order = None

        # --- 2. add the bar to history ------------------------------------
        if not tf_is_base:
            push_tf(bar)
        push_htf(bar)
        push_h1(bar)
        if progress and i % 100 == 0:
            progress(i / total)

        # --- 3. decide for the next bar -----------------------------------
        if position or i < warmup or i == total - 1:
            continue
        closes_tf_bar = tf_is_base or engine._bucket(base[i + 1]["t"], tf) != engine._bucket(bar["t"], tf)
        if not closes_tf_bar:
            continue
        now_ts = bar["t"] + base_step
        window = base[max(0, i + 1 - WINDOW):i + 1] if tf_is_base else tf_bars[-WINDOW:]
        h1_window = base[max(0, i + 1 - H1_WINDOW):i + 1] if base_step == 3600 else h1_bars[-H1_WINDOW:]
        cut = bisect.bisect_right(dxy_times, now_ts - 3600)
        result = engine.evaluate(tf, mode, window, htf_bars[-HTF_WINDOW:], h1_window,
                                 dxy_all[max(0, cut - 60):cut], None, now_ts, bar["c"], now_ts,
                                 backtest=True)
        evaluated += 1
        plan = result["plan"]
        if result["verdict"] != "NO_TRADE" and plan:
            signals += 1
            order = {"bias": plan["bias"], "entry": plan["entry"], "stop": plan["stop_loss"],
                     "target": plan["take_profit_1"], "risk": plan["sl_points"],
                     "grade": plan["setup_grade"]}

    equity, curve = 0.0, []
    for t in trades:
        equity += t["r"]
        curve.append({"time": t["closed_ts"], "value": round(equity, 3)})
    first_tested = base[min(warmup, total - 1)]["t"]
    return {
        "status": "done",
        "timeframe": tf,
        "mode": mode,
        "period": {"from": _iso(first_tested), "to": _iso(base[-1]["t"]),
                   "days": round((base[-1]["t"] - first_tested) / 86400)},
        "bars_evaluated": evaluated,
        "signals": signals,
        "summary": _stats(trades),
        "by_direction": {side: _stats([t for t in trades if t["bias"] == side])
                         for side in ("long", "short")},
        "by_result": {r: sum(1 for t in trades if t["result"] == r) for r in ("target", "stop", "time")},
        "equity": curve,
        "trades": trades,
        "assumptions": [
            "A limit order sits at the plan's entry for one bar; no rejection-candle confirmation.",
            f"Full exit at target 1 ({engine.MODES[mode]['min_rr']}R) or the stop; "
            f"if one bar touches both, the stop is counted.",
            f"A trade still open after {max_hold} {tf} bars is closed at market.",
            f"{COST_POINTS:.2f} points of spread charged per trade; no slippage or swap.",
            "News gate is off - there is no historical calendar. Trades around CPI/NFP/FOMC are included.",
            "Prices are gold futures (continuous front contract), not broker spot.",
        ],
        "generated_at": int(time.time()),
    }


def start(tf: str, mode: str, peek: bool = False):
    """Return the cached result, or the state of a background run.

    Starts a run when there is none, unless `peek` is set.
    """
    tf, mode = engine.check_args(tf, mode)
    if tf not in SUPPORTED:
        raise ValueError(f"Backtest covers {', '.join(SUPPORTED)} only - {tf} has too little history.")
    if tf not in engine.MODES[mode]["tfs"]:
        raise ValueError(f"{mode} mode is not defined for {tf}.")
    key = f"{tf}:{mode}"
    with _lock:
        job = _jobs.get(key)
        fresh = job and job["state"] == "done" and time.time() - job["finished"] < RESULT_TTL
        if job and (job["state"] == "running" or fresh):
            return job
        if peek:
            return {"state": "idle", "progress": 0.0, "result": None, "error": None}
        job = {"state": "running", "progress": 0.0, "result": None, "error": None, "finished": 0}
        _jobs[key] = job

    def work():
        try:
            job["result"] = run(tf, mode, lambda p: job.__setitem__("progress", round(p, 3)))
            job["state"] = "done"
        except Exception as exc:  # noqa: BLE001 - reported to the page
            job["error"] = str(exc)
            job["state"] = "failed"
        job["finished"] = time.time()

    threading.Thread(target=work, daemon=True).start()
    return job

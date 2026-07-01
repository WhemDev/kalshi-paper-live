"""
CLOUD PAPER ENGINE - Kalshi 15m crypto direction, EXTREME-TAILS strategy.

PAPER MODE: no real orders ever. Auto paper-enters when the rule fires, at the
REAL Kalshi ask captured in the first minutes of each 15m window; resolves with
the REAL Kalshi settlement result.

Strategy (reverse-engineering result, 2026-07):
  market makers price mean-reversion at open but UNDERPRICE it at extremes.
  -> bet only when prev 15m return is in the extreme tails (trailing 10% quantiles)
     AND the ML model (logistic, mean-reversion + order-flow) confirms direction.

State is pushed to the 'state' branch via GitHub Contents API (no Pages rebuild).
Runs on GitHub Actions (or locally). ASCII-only logs.

Usage: python engine.py [run_minutes]
Env:   GITHUB_TOKEN + GITHUB_REPOSITORY enable state pushing (on runner).
"""
import base64
import csv
import io
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

RUN_MINUTES = float(sys.argv[1]) if len(sys.argv) > 1 else 350.0
CAMPAIGN_HOURS = 12.0
POLL_SECONDS = 30
CAPTURE_SEC = 150            # bet only in first 2.5 min of a window
TAIL_Q = 0.10                # extreme tails: top/bottom 10%
CONF_UP, CONF_DN = 0.52, 0.48
MAX_ENTRY = 0.53            # backtest: giris >~0.52 ise edge olur; ustunu ALMA (skip+logla)
TRAIN_DAYS = 90
ASSETS = {"BTC": ("BTCUSDT", "KXBTC15M"), "ETH": ("ETHUSDT", "KXETH15M")}
KALSHI = "https://api.elections.kalshi.com/trade-api/v2"

GH_TOKEN = os.environ.get("GITHUB_TOKEN", "")
GH_REPO = os.environ.get("GITHUB_REPOSITORY", "")
STATE_BRANCH = "state"

S = requests.Session()
S.headers.update({"User-Agent": "Mozilla/5.0"})
BINANCE_HOSTS = ["https://api.binance.com", "https://api.binance.us"]
_binance = {"host": None}


def log(msg):
    print(f"[{datetime.now(timezone.utc):%H:%M:%S}] {msg}", flush=True)


def now():
    return datetime.now(timezone.utc)


# ---------------- Binance (features/model) ----------------
def klines(symbol, limit=200, start=None, end=None):
    hosts = [_binance["host"]] if _binance["host"] else BINANCE_HOSTS
    for h in hosts:
        try:
            p = {"symbol": symbol, "interval": "15m", "limit": limit}
            if start:
                p["startTime"] = start
            if end:
                p["endTime"] = end
            r = S.get(f"{h}/api/v3/klines", params=p, timeout=12)
            if r.status_code == 200:
                d = r.json()
                if isinstance(d, list) and d:
                    _binance["host"] = h
                    return d
        except Exception:
            continue
    return []


def fetch_history(symbol, days):
    end = int(time.time() * 1000)
    cur = end - days * 86400 * 1000
    out = []
    while cur < end:
        k = klines(symbol, 1000, start=cur)
        if not k:
            break
        out += k
        cur = k[-1][6] + 1
        if len(k) < 1000:
            break
        time.sleep(0.06)
    return out


def features_from(k):
    o = np.array([float(x[1]) for x in k]); c = np.array([float(x[4]) for x in k])
    h = np.array([float(x[2]) for x in k]); l = np.array([float(x[3]) for x in k])
    vol = np.array([float(x[5]) for x in k]); tb = np.array([float(x[9]) for x in k])
    ts = np.array([int(x[0]) for x in k])
    ret = (c - o) / o
    tbr = tb / np.maximum(vol, 1e-9)
    rng = (h - l) / o
    up = (c >= o).astype(int)
    rv = np.array([ret[max(0, i - 8):i].std() if i >= 2 else 0.0 for i in range(len(ret))])

    def fv(i):
        hr = (ts[i] // 3600000) % 24
        return [ret[i], tbr[i] - 0.5, ret[i - 1], tbr[i - 1] - 0.5, ret[i] + ret[i - 1],
                rng[i], rv[i], np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24)]
    return ret, up, fv, len(k)


def train(symbol):
    k = fetch_history(symbol, TRAIN_DAYS)
    ret, up, fv, n = features_from(k)
    X = np.array([fv(i) for i in range(10, n - 1)])
    y = up[11:n]
    sc = StandardScaler().fit(X)
    m = LogisticRegression(C=0.3, max_iter=2000).fit(sc.transform(X), y)
    log(f"trained {symbol}: n={len(X)} host={_binance['host']}")
    return m, sc, list(ret[-5000:])


def latest_signal(symbol, model, sc, ret_hist):
    """From last CLOSED candle: (prob_up, prev_ret). Appends to ret_hist."""
    k = klines(symbol, 40)
    k = k[:-1]                       # drop forming candle
    ret, up, fv, n = features_from(k)
    x = np.array(fv(n - 1)).reshape(1, -1)
    prob = float(model.predict_proba(sc.transform(x))[0, 1])
    r1 = float(ret[n - 1])
    ret_hist.append(r1)
    if len(ret_hist) > 6000:
        del ret_hist[:1000]
    return prob, r1


# ---------------- Kalshi (prices + settlement) ----------------
def kalshi_get(path, params=None):
    r = S.get(KALSHI + path, params=params or {}, timeout=12)
    r.raise_for_status()
    return r.json()


def kalshi_fee(p):
    return min(0.07, max(0.0, round(0.07 * p * (1 - p), 4)))


def current_market(series):
    try:
        d = kalshi_get("/markets", {"series_ticker": series, "status": "open", "limit": 20})
    except Exception:
        return None
    best = None
    for m in d.get("markets", []):
        ct = m.get("close_time")
        if not ct:
            continue
        cdt = datetime.fromisoformat(ct.replace("Z", "+00:00"))
        if cdt <= now():
            continue
        if best is None or cdt < best[0]:
            def f(v):
                try:
                    return float(v)
                except (TypeError, ValueError):
                    return None
            best = (cdt, {"ticker": m.get("ticker"), "close_time": ct,
                          "yes_ask": f(m.get("yes_ask_dollars")),
                          "no_ask": f(m.get("no_ask_dollars"))})
    return best[1] if best else None


def kalshi_result(ticker):
    try:
        m = kalshi_get(f"/markets/{ticker}").get("market", {})
        r = m.get("result")
        if r == "yes":
            return 1
        if r == "no":
            return 0
    except Exception:
        pass
    return None


# ---------------- state publishing (Contents API, state branch) ----------------
def gh_put(path, content_bytes, message):
    if not (GH_TOKEN and GH_REPO):
        with open(os.path.basename(path), "wb") as f:
            f.write(content_bytes)
        return True
    url = f"https://api.github.com/repos/{GH_REPO}/contents/{path}"
    hdr = {"Authorization": f"token {GH_TOKEN}"}
    sha = None
    try:
        g = S.get(url, params={"ref": STATE_BRANCH}, headers=hdr, timeout=12)
        if g.status_code == 200:
            sha = g.json().get("sha")
    except Exception:
        pass
    body = {"message": message, "branch": STATE_BRANCH,
            "content": base64.b64encode(content_bytes).decode()}
    if sha:
        body["sha"] = sha
    try:
        r = S.put(url, headers=hdr, json=body, timeout=15)
        return r.status_code in (200, 201)
    except Exception:
        return False


def gh_get_json(path):
    if not (GH_TOKEN and GH_REPO):
        try:
            return json.load(open(os.path.basename(path), encoding="utf-8"))
        except Exception:
            return None
    try:
        r = S.get(f"https://raw.githubusercontent.com/{GH_REPO}/{STATE_BRANCH}/{path}",
                  headers={"Cache-Control": "no-cache"}, timeout=12)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


# ---------------- main loop ----------------
def main():
    start = now()
    prev = gh_get_json("state.json") or {}
    deadline_iso = prev.get("campaign_deadline")
    trades = prev.get("all_trades", [])
    if deadline_iso:
        deadline = datetime.fromisoformat(deadline_iso)
    else:
        deadline = start.timestamp() + CAMPAIGN_HOURS * 3600
        deadline = datetime.fromtimestamp(deadline, timezone.utc)
    if now() >= deadline:
        log(f"campaign deadline {deadline.isoformat()} passed - exiting")
        return
    run_end = min(start.timestamp() + RUN_MINUTES * 60, deadline.timestamp())
    log(f"engine start | run until {datetime.fromtimestamp(run_end, timezone.utc):%H:%M:%S} "
        f"| campaign deadline {deadline:%Y-%m-%d %H:%M:%S}")

    models = {}
    ret_hist = {}
    for a, (sym, _) in ASSETS.items():
        m, sc, rh = train(sym)
        models[a] = (m, sc)
        ret_hist[a] = rh

    pending = {k: v for k, v in (prev.get("pending") or {}).items()}
    seen = set(tuple(x) for x in (prev.get("seen") or []))
    openings = prev.get("openings") or []      # her pencerenin acilis ask kaydi (formul dogrulama)
    skips = prev.get("skips") or {"price": 0, "no_signal": 0}
    current = {}
    last_push = 0.0

    def publish(force=False):
        nonlocal last_push
        if not force and time.time() - last_push < 90:
            return
        wins = sum(1 for t in trades if t["pnl"] > 0)
        nb = len(trades)
        eq = list(np.cumsum([t["pnl"] for t in trades])) if trades else []
        maxdd = 0.0
        pk = -1e9
        for v in eq:
            pk = max(pk, v)
            maxdd = max(maxdd, pk - v)
        state = {
            "updated": now().isoformat(),
            "campaign_deadline": deadline.isoformat(),
            "mode": "PAPER (auto-entry, real Kalshi ask + real settlement)",
            "strategy": f"extreme-tails {int(TAIL_Q*100)}% + model confirm | Kalshi 15m",
            "current": current,
            "stats": {
                "bets": nb, "wins": wins,
                "winrate": round(wins / nb, 4) if nb else None,
                "total_pnl": round(sum(t["pnl"] for t in trades), 4),
                "avg_pnl": round(sum(t["pnl"] for t in trades) / nb, 4) if nb else None,
                "max_drawdown": round(maxdd, 4),
                "equity": [round(x, 4) for x in eq],
            },
            "recent_trades": trades[-30:][::-1],
            "openings": openings[-120:],
            "skips": skips,
            "pending": pending,
            "seen": [list(x) for x in list(seen)[-200:]],
            "all_trades": trades,
        }
        ok = gh_put("state.json", json.dumps(state, indent=1).encode(), "state update")
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["ts", "asset", "ticker", "side", "entry", "fee", "prob", "prev_ret",
                    "resolved_up", "won", "pnl"])
        for t in trades:
            w.writerow([t["ts"], t["asset"], t["ticker"], t["side"], t["entry"], t["fee"],
                        t["prob"], t["prev_ret"], t.get("resolved_up"), int(t["pnl"] > 0), t["pnl"]])
        gh_put("trades.csv", buf.getvalue().encode(), "trades update")
        last_push = time.time()
        log(f"state pushed (ok={ok}) bets={nb}")

    publish(force=True)
    while time.time() < run_end:
        # ---- capture ----
        for asset, (sym, series) in ASSETS.items():
            try:
                mkt = current_market(series)
                if not mkt:
                    continue
                ct = mkt["close_time"]
                cdt = datetime.fromisoformat(ct.replace("Z", "+00:00"))
                secs_in = now().timestamp() - (cdt.timestamp() - 900)
                prob, r1 = latest_signal(sym, *models[asset], ret_hist[asset])
                rh = np.array(ret_hist[asset])
                lo, hi = np.quantile(rh, [TAIL_Q, 1 - TAIL_Q])
                side = None
                if r1 > hi and prob < CONF_DN:
                    side, px = "NO", mkt["no_ask"]
                elif r1 < lo and prob > CONF_UP:
                    side, px = "YES", mkt["yes_ask"]
                in_cap = 0 <= secs_in <= CAPTURE_SEC
                sig = "WAIT"
                if side and in_cap:
                    sig = f"ENTER {side} @ {px}"
                elif side:
                    sig = "signal-but-late"
                current[asset] = {"ticker": mkt["ticker"], "window_end": ct,
                                  "secs_in": round(secs_in), "prob_up": round(prob, 4),
                                  "prev_ret": round(r1, 5),
                                  "tail_lo": round(float(lo), 5), "tail_hi": round(float(hi), 5),
                                  "yes_ask": mkt["yes_ask"], "no_ask": mkt["no_ask"],
                                  "signal": sig}
                key = (asset, ct)
                if key in seen or str(key) in pending:
                    continue
                if in_cap:
                    openings.append({"asset": asset, "window_end": ct,
                                     "yes_ask": mkt["yes_ask"], "no_ask": mkt["no_ask"],
                                     "prev_ret": round(r1, 5), "secs_in": round(secs_in)})
                    if side and px and px <= MAX_ENTRY:
                        pending[str(key)] = {"asset": asset, "ticker": mkt["ticker"],
                                             "close_iso": ct, "side": side, "entry": px,
                                             "prob": round(prob, 4), "prev_ret": round(r1, 5)}
                        log(f"PAPER-ENTER {asset} {ct} {side} @ {px} (prob={prob:.3f} r1={r1:+.5f})")
                    elif side and px:
                        skips["price"] += 1
                        seen.add(key)
                        log(f"SKIP-PRICE {asset} {ct} {side} ask={px} > {MAX_ENTRY}")
                    else:
                        skips["no_signal"] += 1
                        seen.add(key)
            except Exception as e:
                log(f"warn capture {asset}: {repr(e)[:90]}")

        # ---- resolve ----
        for skey in list(pending.keys()):
            info = pending[skey]
            cdt = datetime.fromisoformat(info["close_iso"].replace("Z", "+00:00"))
            if (now() - cdt).total_seconds() < 45:
                continue
            res = kalshi_result(info["ticker"])
            if res is None:
                if (now() - cdt).total_seconds() > 900:
                    pending.pop(skey, None)
                    seen.add((info["asset"], info["close_iso"]))
                continue
            won = (res == (1 if info["side"] == "YES" else 0))
            fee = kalshi_fee(info["entry"])
            pnl = round((1 - info["entry"] - fee) if won else (-info["entry"] - fee), 4)
            trades.append({"ts": now().isoformat(), "asset": info["asset"],
                           "ticker": info["ticker"], "side": info["side"],
                           "entry": info["entry"], "fee": round(fee, 4),
                           "prob": info["prob"], "prev_ret": info["prev_ret"],
                           "resolved_up": res, "pnl": pnl})
            pending.pop(skey, None)
            seen.add((info["asset"], info["close_iso"]))
            log(f"RESOLVE {info['asset']} {info['side']} -> {'UP' if res else 'DOWN'} "
                f"{'WIN' if won else 'LOSS'} pnl={pnl:+.3f}")
            publish(force=True)

        publish()
        time.sleep(POLL_SECONDS)

    publish(force=True)
    log("run finished")


if __name__ == "__main__":
    main()

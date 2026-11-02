#!/usr/bin/env bash
set -euo pipefail

APP_DIR="/root/coindcx-bot"
DEST="$APP_DIR/bot_v3.py"
TMP="$APP_DIR/.bot_v3.py.tmp"
mkdir -p "$APP_DIR"

# Use the existing project virtual environment. Do not silently fall back to system Python.
PY="$APP_DIR/venv/bin/python"
if [ ! -x "$PY" ]; then
  echo "ERROR: $PY not found. Existing bot environment was not changed." >&2
  exit 1
fi

"$PY" -c 'import requests, numpy, pandas, sklearn' >/dev/null 2>&1 || {
  echo "ERROR: Required packages are missing in the project venv. No files were changed." >&2
  exit 1
}

cat > "$TMP" <<'PYEOF'
import os, time, json, hmac, hashlib, sqlite3, threading, logging, math
from datetime import datetime, date
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from decimal import Decimal, ROUND_DOWN

import requests
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score

APP_DIR = "/root/coindcx-bot"
CONFIG_FILE = os.path.join(APP_DIR, "config.json")
LOG_FILE = os.path.join(APP_DIR, "bot.log")
DB_FILE = os.path.join(APP_DIR, "learning.db")
BASE_URL = "https://api.coindcx.com"
PUBLIC_URL = "https://public.coindcx.com"

# Established large-cap candidates only; active contracts and liquidity ranking are checked live.
CANDIDATE_ASSETS = ["BTC", "ETH", "SOL", "XRP", "BNB", "DOGE", "ADA", "LINK", "LTC", "BCH", "AVAX", "DOT"]
MAX_SCAN_PAIRS = 8
FEATURES = ["ret1", "ret3", "ret6", "rsi", "ema_gap", "ema_slope", "bb_pos", "atr_pct", "vol_ratio", "upper_wick", "lower_wick", "body_ratio", "macd_norm"]

os.makedirs(APP_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.StreamHandler(), logging.FileHandler(LOG_FILE)])
log = logging.getLogger("CoinDCX-ML-v3")

state = {
    "status": "STARTING", "last_scan_time": None, "last_train_time": None,
    "futures_balance_inr": 0.0, "free_balance_inr": 0.0, "locked_balance_inr": 0.0,
    "active_position": None, "active_position_details": [], "api_configured": False,
    "api_status_msg": "", "last_execution_error": None, "latest_scans": {},
    "trade_history": [], "selected_pairs": [], "validation_score": None,
    "learning_stats": {}, "training_stats": {}, "auto_trade_enabled": False,
    "validation_passed": False, "validation_min_score": 0.36,
    "training_rows": 0, "validation_rows": 0, "model_pairs": [],
    "last_training_attempt_time": None, "last_scan_attempt_time": None,
    "last_scan_status": "NOT_STARTED", "active_universe": [],
    "last_training_failure": None, "last_risk_block": None
}


def load_config():
    defaults = {
        "api_key": "", "api_secret": "", "max_trade_margin_inr": 2000,
        "leverage": 15, "auto_trade_enabled": True, "inr_per_usdt": 91.5,
        "max_daily_trades": 3, "cooldown_minutes": 30,
        "scan_interval_seconds": 60, "max_scan_pairs": 8,
        "min_validation_score": 0.36, "base_confidence": 0.72,
        "max_confidence": 0.82, "min_free_balance_inr": 500
    }
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            user = json.load(f)
        defaults.update(user if isinstance(user, dict) else {})
    except FileNotFoundError:
        pass
    except Exception as exc:
        log.error("Config read error: %s", exc)
    return defaults


class Store:
    def __init__(self, path):
        self.path = path
        self.lock = threading.Lock()
        self.init_db()

    def connect(self):
        c = sqlite3.connect(self.path, timeout=10)
        c.row_factory = sqlite3.Row
        return c

    def init_db(self):
        with self.connect() as c:
            c.execute("CREATE TABLE IF NOT EXISTS predictions (id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT NOT NULL, candle_ts INTEGER NOT NULL, created_ts INTEGER NOT NULL, side TEXT NOT NULL, entry REAL NOT NULL, confidence REAL NOT NULL, outcome TEXT, exit_price REAL, resolved_ts INTEGER, UNIQUE(symbol,candle_ts))")
            c.execute("CREATE TABLE IF NOT EXISTS trade_events (id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT, side TEXT, created_ts INTEGER, confidence REAL, order_id TEXT)")
            c.execute("""CREATE TABLE IF NOT EXISTS training_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT, created_ts INTEGER NOT NULL,
                validation_score REAL, passed INTEGER NOT NULL, reason TEXT NOT NULL,
                training_rows INTEGER NOT NULL DEFAULT 0, validation_rows INTEGER NOT NULL DEFAULT 0,
                pairs TEXT NOT NULL DEFAULT ''
            )""")
            c.execute("CREATE INDEX IF NOT EXISTS idx_training_runs_created ON training_runs(created_ts)")
            c.execute("""CREATE TABLE IF NOT EXISTS order_intents (
                id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT NOT NULL, candle_ts INTEGER NOT NULL,
                created_ts INTEGER NOT NULL, side TEXT NOT NULL, status TEXT NOT NULL,
                order_id TEXT, UNIQUE(symbol,candle_ts,side)
            )""")
            c.commit()

    def add_prediction(self, symbol, candle_ts, side, price, confidence):
        with self.lock, self.connect() as c:
            c.execute("INSERT OR IGNORE INTO predictions(symbol,candle_ts,created_ts,side,entry,confidence) VALUES(?,?,?,?,?,?)", (symbol, int(candle_ts), int(time.time()), side, float(price), float(confidence)))
            c.commit()

    def resolve_due(self, prices):
        now = int(time.time())
        with self.lock, self.connect() as c:
            rows = c.execute("SELECT id,symbol,side,entry,created_ts FROM predictions WHERE outcome IS NULL AND created_ts<=?", (now - 900,)).fetchall()
            for r in rows:
                px = prices.get(r["symbol"])
                if px is None or px <= 0 or r["entry"] <= 0:
                    continue
                change = (px - r["entry"]) / r["entry"]
                signed = change if r["side"] == "BUY" else -change
                # Ignore near-flat changes to avoid counting noise as a win/loss.
                outcome = "FLAT" if abs(signed) < 0.0005 else ("WIN" if signed > 0 else "LOSS")
                c.execute("UPDATE predictions SET outcome=?,exit_price=?,resolved_ts=? WHERE id=?", (outcome, float(px), now, int(r["id"])))
            c.commit()

    def threshold(self, symbol, cfg):
        base = float(cfg.get("base_confidence", 0.72))
        maximum = float(cfg.get("max_confidence", 0.82))
        with self.connect() as c:
            rows = c.execute("SELECT outcome FROM predictions WHERE symbol=? AND outcome IN ('WIN','LOSS') ORDER BY resolved_ts DESC LIMIT 30", (symbol,)).fetchall()
        if len(rows) < 15:
            return base, len(rows), None
        wins = sum(1 for r in rows if r["outcome"] == "WIN")
        wr = wins / len(rows)
        if wr < 0.48:
            return min(maximum, base + 0.10), len(rows), wr
        if wr < 0.55:
            return min(maximum, base + 0.05), len(rows), wr
        return base, len(rows), wr

    def stats(self):
        with self.connect() as c:
            total = c.execute("SELECT COUNT(*) n FROM predictions").fetchone()["n"]
            rows = c.execute("SELECT outcome,COUNT(*) n FROM predictions WHERE outcome IS NOT NULL GROUP BY outcome").fetchall()
            by_symbol = c.execute("SELECT symbol, COUNT(*) resolved, SUM(CASE WHEN outcome='WIN' THEN 1 ELSE 0 END) wins FROM predictions WHERE outcome IN ('WIN','LOSS') GROUP BY symbol ORDER BY resolved DESC").fetchall()
            trades = c.execute("SELECT COUNT(*) n FROM trade_events WHERE created_ts>=?", (int(time.time()) - 86400,)).fetchone()["n"]
        counts = {r["outcome"]: r["n"] for r in rows}
        decided = counts.get("WIN", 0) + counts.get("LOSS", 0)
        return {
            "tracked_signals": total, "resolved_signals": sum(counts.values()),
            "wins": counts.get("WIN", 0), "losses": counts.get("LOSS", 0),
            "flat": counts.get("FLAT", 0), "win_rate_pct": round(100 * counts.get("WIN", 0) / decided, 1) if decided else None,
            "executed_trades_24h": trades,
            "by_symbol": [{"symbol": r["symbol"], "resolved": r["resolved"], "wins": r["wins"], "win_rate_pct": round(100*r["wins"]/r["resolved"],1) if r["resolved"] else None} for r in by_symbol]
        }

    def add_training_run(self, score, passed, reason, training_rows=0, validation_rows=0, pairs=None):
        with self.lock, self.connect() as c:
            c.execute(
                "INSERT INTO training_runs(created_ts,validation_score,passed,reason,training_rows,validation_rows,pairs) VALUES(?,?,?,?,?,?,?)",
                (int(time.time()), None if score is None else float(score), 1 if passed else 0,
                 str(reason)[:300], int(training_rows or 0), int(validation_rows or 0),
                 ",".join(pairs or []))
            )
            c.commit()

    def training_stats(self):
        with self.connect() as c:
            total = c.execute("SELECT COUNT(*) n FROM training_runs").fetchone()["n"]
            passed = c.execute("SELECT COUNT(*) n FROM training_runs WHERE passed=1").fetchone()["n"]
            last = c.execute("SELECT created_ts,validation_score,passed,reason,training_rows,validation_rows,pairs FROM training_runs ORDER BY id DESC LIMIT 1").fetchone()
            recent = c.execute("SELECT created_ts,validation_score,passed,reason,training_rows,validation_rows,pairs FROM training_runs ORDER BY id DESC LIMIT 20").fetchall()
        def pack(r):
            return {
                "time": datetime.fromtimestamp(int(r["created_ts"])).strftime("%Y-%m-%d %H:%M:%S"),
                "validation_score": r["validation_score"], "passed": bool(r["passed"]),
                "reason": r["reason"], "training_rows": int(r["training_rows"]),
                "validation_rows": int(r["validation_rows"]),
                "pairs": [x for x in str(r["pairs"] or "").split(",") if x]
            }
        return {
            "attempts": int(total), "passed_attempts": int(passed),
            "failed_attempts": int(total-passed),
            "last_attempt": pack(last) if last else None,
            "recent": [pack(r) for r in recent]
        }

    def reserve_order_intent(self, symbol, candle_ts, side):
        # Persist intent before sending; ambiguous timeouts must not trigger an automatic duplicate.
        with self.lock, self.connect() as c:
            cur = c.execute(
                "INSERT OR IGNORE INTO order_intents(symbol,candle_ts,created_ts,side,status) VALUES(?,?,?,?,?)",
                (symbol, int(candle_ts), int(time.time()), side, "PENDING")
            )
            c.commit()
            return cur.rowcount == 1

    def finish_order_intent(self, symbol, candle_ts, side, status, order_id=None):
        with self.lock, self.connect() as c:
            c.execute(
                "UPDATE order_intents SET status=?,order_id=? WHERE symbol=? AND candle_ts=? AND side=?",
                (status, str(order_id or ""), symbol, int(candle_ts), side)
            )
            c.commit()

    def trade_count_today(self):
        start = int(datetime.combine(date.today(), datetime.min.time()).timestamp())
        with self.connect() as c:
            return c.execute("SELECT COUNT(*) n FROM trade_events WHERE created_ts>=?", (start,)).fetchone()["n"]

    def last_trade_ts(self):
        with self.connect() as c:
            r = c.execute("SELECT MAX(created_ts) t FROM trade_events").fetchone()
            return int(r["t"] or 0)

    def add_trade(self, symbol, side, confidence, order_id):
        with self.lock, self.connect() as c:
            c.execute("INSERT INTO trade_events(symbol,side,created_ts,confidence,order_id) VALUES(?,?,?,?,?)", (symbol, side, int(time.time()), float(confidence), str(order_id or "")))
            c.commit()


store = Store(DB_FILE)


class CoinDCX:
    def __init__(self, cfg):
        self.key = str(cfg.get("api_key", "")).strip()
        self.secret = str(cfg.get("api_secret", "")).strip()
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "CoinDCX-ML-v3/1.0"})

    def update(self, cfg):
        self.key = str(cfg.get("api_key", "")).strip()
        self.secret = str(cfg.get("api_secret", "")).strip()

    def signed(self, method, path, body=None, params=None, timeout=10):
        if not self.key or not self.secret:
            raise RuntimeError("CoinDCX API credentials missing")
        if body is None:
            body = {"timestamp": int(time.time() * 1000)}
        payload = json.dumps(body, separators=(",", ":"))
        sig = hmac.new(self.secret.encode(), payload.encode(), hashlib.sha256).hexdigest()
        headers = {"Content-Type": "application/json", "X-AUTH-APIKEY": self.key, "X-AUTH-SIGNATURE": sig}
        url = BASE_URL + path
        if method == "GET":
            resp = self.session.get(url, data=payload, params=params, headers=headers, timeout=timeout)
        else:
            resp = self.session.post(url, data=payload, headers=headers, timeout=timeout)
        if resp.status_code >= 400:
            raise RuntimeError(f"CoinDCX HTTP {resp.status_code}: {resp.text[:500]}")
        try:
            return resp.json()
        except Exception:
            raise RuntimeError(f"CoinDCX non-JSON response: {resp.text[:300]}")

    def active_instruments(self):
        url = BASE_URL + "/exchange/v1/derivatives/futures/data/active_instruments"
        r = self.session.get(url, params={"margin_currency_short_name[]": "INR"}, timeout=12)
        r.raise_for_status()
        data = r.json()
        out = set()
        if isinstance(data, list):
            for x in data:
                pair = x.get("pair") if isinstance(x, dict) else x
                if pair:
                    out.add(str(pair).upper())
        return out

    def instrument(self, pair):
        r = self.session.get(BASE_URL + "/exchange/v1/derivatives/futures/data/instrument", params={"pair": pair, "margin_currency_short_name": "INR"}, timeout=10)
        r.raise_for_status()
        d = r.json()
        return d.get("instrument", d) if isinstance(d, dict) else {}

    def wallet(self, fx):
        data = self.signed("GET", "/exchange/v1/derivatives/futures/wallets")
        total = free = locked = 0.0
        if isinstance(data, list):
            for w in data:
                cur = str(w.get("currency_short_name", "")).upper()
                bal = float(w.get("balance", 0) or 0)
                lk = float(w.get("locked_balance", 0) or 0)
                rate = 1.0 if cur == "INR" else fx
                total += (bal + lk) * rate
                free += bal * rate
                locked += lk * rate
        return {"total": round(total, 2), "free": round(free, 2), "locked": round(locked, 2)}

    def positions(self):
        found = {}
        for margin in (["INR"], ["USDT"]):
            for page in (1, 2):
                body = {"timestamp": int(time.time()*1000), "page": page, "size": 100, "margin_currency_short_name": margin}
                try:
                    data = self.signed("POST", "/exchange/v1/derivatives/futures/positions", body=body)
                    if isinstance(data, list):
                        for p in data:
                            qty = float(p.get("active_pos", 0) or 0)
                            if abs(qty) > 1e-8:
                                found[str(p.get("id"))] = p
                except Exception as exc:
                    log.warning("Positions query (%s) failed: %s", margin, exc)
        return list(found.values())

    def set_tpsl(self, pos_id, tp, sl):
        body = {"timestamp": int(time.time()*1000), "id": pos_id,
                "take_profit": {"stop_price": str(tp), "order_type": "take_profit_market"},
                "stop_loss": {"stop_price": str(sl), "order_type": "stop_market"}}
        data = self.signed("POST", "/exchange/v1/derivatives/futures/positions/create_tpsl", body=body)
        if isinstance(data, dict):
            def confirmed(item):
                if not isinstance(item, dict):
                    return False
                if item.get("success") is False:
                    return False
                return bool(item.get("success") is True or item.get("id"))
            tp_ok = confirmed(data.get("take_profit"))
            sl_ok = confirmed(data.get("stop_loss"))
            if not (tp_ok and sl_ok):
                log.error("TP/SL not fully confirmed by API: %s", data)
                return False
            return True
        return False

    def exit_position(self, pos_id):
        data = self.signed("POST", "/exchange/v1/derivatives/futures/positions/exit", body={"timestamp": int(time.time()*1000), "id": pos_id})
        return data

    def place_order(self, pair, side, qty, leverage):
        order = {"pair": pair, "side": side.lower(), "order_type": "market_order", "total_quantity": qty,
                 "leverage": int(leverage), "margin_currency_short_name": "INR"}
        body = {"timestamp": int(time.time()*1000), "order": order}
        return self.signed("POST", "/exchange/v1/derivatives/futures/orders/create", body=body)


class MLEngine:
    def __init__(self, client):
        self.client = client
        self.model = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.5, random_state=42)
        self.is_trained = False
        self.validation_score = None
        self.last_train = 0
        self.last_train_attempt = 0
        self.last_universe_refresh = 0
        self.selected_assets = []
        self.pairs = []
        self.price_cache = {}
        self.candle_cache = {}
        self.instrument_cache = {}

    @staticmethod
    def pair_for(asset):
        return f"B-{asset}_USDT"

    def refresh_universe(self, force=False):
        if self.pairs and not force and time.time() - self.last_universe_refresh < 3600:
            return self.pairs
        try:
            active = self.client.active_instruments()
            pairs = [self.pair_for(a) for a in CANDIDATE_ASSETS if self.pair_for(a) in active]
            if not pairs:
                log.warning("None of the large-cap allowlisted pairs are active for INR futures")
                self.pairs = []
            else:
                self.pairs = pairs
                log.info("Active allowlisted Futures pairs: %s", ",".join(pairs))
        except Exception as exc:
            log.error("Active instrument lookup failed; entries disabled: %s", exc)
            self.pairs = []
        self.last_universe_refresh = time.time()
        return self.pairs

    def fetch(self, pair, resolution="5", limit=700):
        now = int(time.time())
        mins = int(resolution) if str(resolution).isdigit() else 5
        frm = now - (limit * mins * 60 + 3600)
        url = PUBLIC_URL + "/market_data/candlesticks"
        try:
            r = requests.get(url, params={"pair": pair, "from": frm, "to": now, "resolution": str(resolution), "pcode": "f"}, timeout=12)
            r.raise_for_status()
            raw = r.json()
            if isinstance(raw, dict):
                if raw.get("s") not in (None, "ok"):
                    return pd.DataFrame()
                raw = raw.get("data", raw.get("candlesticks", []))
            d = pd.DataFrame(raw)
            if d.empty:
                return d
            if "time" in d.columns and "timestamp" not in d.columns:
                d = d.rename(columns={"time": "timestamp"})
            req = ["timestamp", "open", "high", "low", "close", "volume"]
            if not all(k in d.columns for k in req):
                log.warning("Unexpected candle schema %s: %s", pair, list(d.columns))
                return pd.DataFrame()
            for k in req:
                d[k] = pd.to_numeric(d[k], errors="coerce")
            d = d.dropna(subset=req).sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
            if not d.empty and d.timestamp.iloc[-1] > 1e12:
                d["timestamp"] = d["timestamp"] / 1000.0
            d = d.tail(limit).reset_index(drop=True)
            if len(d):
                self.price_cache[pair.replace("B-", "").replace("_USDT", "USDT")] = float(d.close.iloc[-1])
            self.candle_cache[(pair, str(resolution))] = d
            return d
        except Exception as exc:
            log.warning("Candle fetch failed %s/%s: %s", pair, resolution, exc)
            return pd.DataFrame()

    @staticmethod
    def features(src):
        if src is None or len(src) < 60:
            return pd.DataFrame()
        d = src.copy()
        c, h, l, o, v = d.close, d.high, d.low, d.open, d.volume
        delta = c.diff()
        gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
        d["rsi"] = 100 - 100/(1 + gain/(loss+1e-9))
        e9 = c.ewm(span=9, adjust=False).mean()
        e21 = c.ewm(span=21, adjust=False).mean()
        e50 = c.ewm(span=50, adjust=False).mean()
        d["ema_gap"] = (e9-e21)/(c+1e-9)
        d["ema_slope"] = (e21-e50)/(c+1e-9)
        d["ret1"] = c.pct_change(1)
        d["ret3"] = c.pct_change(3)
        d["ret6"] = c.pct_change(6)
        mid, sd = c.rolling(20).mean(), c.rolling(20).std()
        d["bb_pos"] = (c-(mid-2*sd))/(4*sd+1e-9)
        tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/14, adjust=False).mean()
        d["atr"] = atr
        d["atr_pct"] = atr/(c+1e-9)
        d["vol_ratio"] = v/(v.rolling(20).mean()+1e-9)
        rng = (h-l).replace(0, np.nan)
        d["upper_wick"] = (h-np.maximum(o,c))/rng
        d["lower_wick"] = (np.minimum(o,c)-l)/rng
        d["body_ratio"] = (c-o).abs()/rng
        macd = c.ewm(span=12, adjust=False).mean()-c.ewm(span=26, adjust=False).mean()
        d["macd_norm"] = macd/(c+1e-9)
        return d.replace([np.inf,-np.inf], np.nan).dropna().reset_index(drop=True)

    def _label(self, pair):
        raw = self.fetch(pair, "5", 700)
        f = self.features(raw)
        if len(f) < 180:
            return None
        future = f.close.shift(-3)/f.close-1
        band = (f.atr_pct*0.85).clip(lower=0.0018, upper=0.012)
        y = pd.Series(1, index=f.index, dtype=int)
        y[future > band] = 2
        y[future < -band] = 0
        valid = future.notna()
        x = f.loc[valid, FEATURES].copy()
        yy = y.loc[valid].copy()
        return f, x, yy

    def train(self, cfg, force=False):
        now = time.time()
        # Throttle failed attempts to avoid hammering public candle endpoints.
        if not force and self.last_train_attempt and now-self.last_train_attempt < 600:
            return
        if self.is_trained and not force and now-self.last_train < 1800:
            return
        self.last_train_attempt = now
        state["last_training_attempt_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        min_score = float(cfg.get("min_validation_score", 0.36))
        state["validation_min_score"] = min_score
        state["validation_passed"] = False
        state["training_rows"] = 0
        state["validation_rows"] = 0
        pairs = self.refresh_universe()
        state["active_universe"] = list(pairs)
        state["selected_pairs"] = list(pairs)
        if not pairs:
            self.is_trained = False
            self.validation_score = None
            state["validation_score"] = None
            state["last_training_failure"] = "No active allowlisted Futures pairs"
            store.add_training_run(None, False, state["last_training_failure"], pairs=[])
            state["training_stats"] = store.training_stats()
            log.warning("ML training skipped: no active allowlisted Futures pairs")
            return
        datasets = []
        for pair in pairs:
            result = self._label(pair)
            if result is None:
                continue
            f, x, y = result
            # 24-hour quote turnover estimate from base-asset volume * close.
            turnover = float((f.close.tail(288) * f.volume.tail(288)).sum())
            datasets.append((pair, f, x, y, turnover))
        if not datasets:
            self.is_trained = False
            self.validation_score = None
            state["validation_score"] = None
            state["last_training_failure"] = "No usable active Futures candle data"
            store.add_training_run(None, False, state["last_training_failure"], pairs=pairs)
            state["training_stats"] = store.training_stats()
            log.warning("ML training skipped: no usable active futures candle data")
            return
        datasets.sort(key=lambda z: z[4], reverse=True)
        keep = max(1, min(int(cfg.get("max_scan_pairs", MAX_SCAN_PAIRS)), MAX_SCAN_PAIRS, len(datasets)))
        selected = datasets[:keep]
        train_x, train_y, val_x, val_y = [], [], [], []
        for pair, f, x, y, turnover in selected:
            cut = int(len(x)*0.80)
            # purge 3 bars so a training label cannot look into validation period
            if cut <= 100 or len(x)-cut < 35:
                continue
            train_x.append(x.iloc[:cut-3])
            train_y.append(y.iloc[:cut-3])
            val_x.append(x.iloc[cut:])
            val_y.append(y.iloc[cut:])
        if not train_x or not val_x:
            self.is_trained = False
            self.validation_score = None
            state["validation_score"] = None
            state["last_training_failure"] = "Insufficient train/validation samples"
            store.add_training_run(None, False, state["last_training_failure"], pairs=[p for p, *_ in selected])
            state["training_stats"] = store.training_stats()
            log.warning("Walk-forward training skipped: insufficient data")
            return
        Xtr, Ytr = pd.concat(train_x), pd.concat(train_y)
        Xv, Yv = pd.concat(val_x), pd.concat(val_y)
        state["training_rows"] = int(len(Xtr))
        state["validation_rows"] = int(len(Xv))
        if Ytr.nunique() < 3 or Yv.nunique() < 2:
            self.is_trained = False
            self.validation_score = None
            state["validation_score"] = None
            state["last_training_failure"] = "Missing label classes in train or validation split"
            store.add_training_run(None, False, state["last_training_failure"], len(Xtr), len(Xv), [p for p, *_ in selected])
            state["training_stats"] = store.training_stats()
            log.warning("Walk-forward training skipped: missing label classes")
            return
        probe = HistGradientBoostingClassifier(max_iter=120, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.5, random_state=42)
        try:
            probe.fit(Xtr, Ytr)
            score = float(balanced_accuracy_score(Yv, probe.predict(Xv)))
            self.validation_score = score
            state["validation_score"] = round(score, 4)
            state["validation_min_score"] = min_score
            state["validation_passed"] = bool(np.isfinite(score) and score >= min_score)
            state["last_training_failure"] = None if state["validation_passed"] else "Validation score below configured minimum"
            if not state["validation_passed"]:
                self.is_trained = False
                self.selected_assets = []
                store.add_training_run(score, False, state["last_training_failure"], len(Xtr), len(Xv), [p for p, *_ in selected])
                state["training_stats"] = store.training_stats()
                log.warning("Walk-forward score %.3f below %.3f; live entries disabled", score, min_score)
                return
            # After measuring holdout score, train production model on all available labeled rows.
            all_x = pd.concat([x for _, _, x, _, _ in selected])
            all_y = pd.concat([y for _, _, _, y, _ in selected])
            self.model.fit(all_x, all_y)
            self.selected_assets = [p.replace("B-", "").replace("_USDT", "USDT") for p, *_ in selected]
            self.last_train = time.time()
            self.is_trained = True
            log.info("ML retrained on %d rows; walk-forward balanced accuracy %.3f; scan pairs=%s", len(all_x), score, self.selected_assets)
            state["last_train_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            state["validation_score"] = round(score, 4)
            state["validation_passed"] = True
            state["last_training_failure"] = None
            state["model_pairs"] = list(self.selected_assets)
            store.add_training_run(score, True, "Validation passed and production model trained", len(all_x), len(Xv), self.selected_assets)
            state["training_stats"] = store.training_stats()
        except Exception as exc:
            self.is_trained = False
            state["validation_passed"] = False
            state["last_training_failure"] = str(exc)[:300]
            store.add_training_run(self.validation_score, False, state["last_training_failure"], state.get("training_rows", 0), state.get("validation_rows", 0), [p for p, *_ in selected] if "selected" in locals() else pairs)
            state["training_stats"] = store.training_stats()
            log.exception("ML training failed: %s", exc)

    @staticmethod
    def trend(df):
        if df.empty or len(df) < 55:
            return 0
        c = df.close.iloc[:-1] if len(df) > 2 else df.close
        e9 = c.ewm(span=9, adjust=False).mean().iloc[-1]
        e21 = c.ewm(span=21, adjust=False).mean().iloc[-1]
        e50 = c.ewm(span=50, adjust=False).mean().iloc[-1]
        return 1 if e9 > e21 > e50 else (-1 if e9 < e21 < e50 else 0)

    def analyze(self, symbol, cfg):
        pair = "B-" + symbol.replace("USDT", "") + "_USDT"
        raw = self.fetch(pair, "5", 240)
        if raw.empty or len(raw) < 80:
            return {"symbol": symbol, "action": "HOLD", "confidence": 0, "reason": "No candles"}
        # Always make decisions on the last completed 5m candle.
        f = self.features(raw.iloc[:-1].copy())
        if f.empty:
            return {"symbol": symbol, "action": "HOLD", "confidence": 0, "reason": "Insufficient features"}
        row = f.iloc[-1]
        price, atr = float(row.close), float(row.atr)
        candle_ts = int(row.timestamp)
        candle_age = max(0, int(time.time()) - candle_ts)
        if candle_age > int(cfg.get("max_candle_age_seconds", 900)):
            return {"symbol": symbol, "action": "HOLD", "price": price, "confidence": 0,
                    "reason": f"Stale 5m candle ({candle_age}s old); entries blocked", "candle_ts": candle_ts}
        if not self.is_trained:
            return {"symbol": symbol, "action": "HOLD", "price": price, "confidence": 0, "reason": "ML unavailable/validation failed"}
        x = pd.DataFrame([{k: float(row[k]) for k in FEATURES}])
        probs = self.model.predict_proba(x)[0]
        p = {int(k): float(v) for k, v in zip(self.model.classes_, probs)}
        ps, ph, pl = p.get(0, 0), p.get(1, 0), p.get(2, 0)
        h1 = self.fetch(pair, "60", 100)
        # Futures REST candle docs list 1h but not 4h; aggregate hourly bars into 4h locally.
        h4_source = self.fetch(pair, "60", 360)
        h4 = pd.DataFrame()
        if not h4_source.empty:
            h4_source = h4_source.copy()
            h4_source["bucket"] = (h4_source["timestamp"] // 14400) * 14400
            h4 = h4_source.groupby("bucket", as_index=False).agg(
                timestamp=("timestamp", "first"), open=("open", "first"),
                high=("high", "max"), low=("low", "min"),
                close=("close", "last"), volume=("volume", "sum"))
        t1, t4 = self.trend(h1), self.trend(h4)
        threshold, n, winrate = store.threshold(symbol, cfg)
        sl_dist = max(1.5*atr, price*0.0025)
        tp_dist = max(2.3*atr, price*0.0035)
        action, conf = "HOLD", max(ps, ph, pl)
        if pl >= threshold and pl > ps and t1 == 1 and t4 >= 0:
            action, conf = "BUY", pl
            out = {"symbol": symbol, "action": action, "price": price, "confidence": round(conf*100,1), "sl": price-sl_dist, "tp": price+tp_dist, "reason": f"LONG {pl:.0%}; H1/H4 aligned; adaptive min {threshold:.0%}"}
        elif ps >= threshold and ps > pl and t1 == -1 and t4 <= 0:
            action, conf = "SELL", ps
            out = {"symbol": symbol, "action": action, "price": price, "confidence": round(conf*100,1), "sl": price+sl_dist, "tp": price-tp_dist, "reason": f"SHORT {ps:.0%}; H1/H4 aligned; adaptive min {threshold:.0%}"}
        else:
            out = {"symbol": symbol, "action": "HOLD", "price": price, "confidence": round(conf*100,1), "reason": f"No aligned edge S/H/L={ps:.2f}/{ph:.2f}/{pl:.2f}; min={threshold:.0%}"}
        out.update({"candle_ts": candle_ts, "threshold": round(threshold,3), "resolved_samples": n, "recent_win_rate": None if winrate is None else round(winrate,3)})
        if action in ("BUY", "SELL"):
            store.add_prediction(symbol, candle_ts, action, price, conf)
        return out

    def instrument_rules(self, pair):
        if pair not in self.instrument_cache:
            try:
                self.instrument_cache[pair] = self.client.instrument(pair)
            except Exception as exc:
                log.warning("Instrument details failed %s: %s", pair, exc)
                self.instrument_cache[pair] = {}
        return self.instrument_cache[pair]


class Dashboard(BaseHTTPRequestHandler):
    def log_message(self, *_):
        return

    def do_GET(self):
        if self.path.startswith("/api/status"):
            payload = dict(state)
            payload["learning_stats"] = store.stats()
            payload["training_stats"] = store.training_stats()
            raw = json.dumps(payload, default=str).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Access-Control-Allow-Origin", "*"); self.end_headers(); self.wfile.write(raw)
            return
        html = """<!doctype html><html><head><meta name='viewport' content='width=device-width,initial-scale=1'><title>CoinDCX ML Bot</title><style>body{background:#0b1020;color:#e6edf7;font:15px Arial;padding:18px}h2{color:#7dd3fc}.card{background:#151e33;border:1px solid #283653;border-radius:12px;padding:14px;margin:10px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style></head><body><h2>CoinDCX ML Futures Bot v3</h2><div class='card' id='main'>Loading…</div><div class='card'><b>Signal learning (not realized PnL)</b><pre id='learn'></pre></div><div class='card'><b>Latest scans</b><pre id='scans'></pre></div><script>async function u(){try{let d=await(await fetch('/api/status')).json();document.getElementById('main').innerHTML=`Status: ${d.status}<br>Auto-trade: ${d.auto_trade_enabled}<br>Balance: ₹${d.futures_balance_inr} (free ₹${d.free_balance_inr})<br>Position: ${d.active_position||'None'}<br>Pairs: ${(d.selected_pairs||[]).join(', ')}<br>Validation: ${d.validation_score}`;document.getElementById('learn').textContent=JSON.stringify(d.learning_stats,null,2);document.getElementById('scans').textContent=JSON.stringify(d.latest_scans,null,2)}catch(e){document.getElementById('main').textContent=e}}u();setInterval(u,10000)</script></body></html>"""
        self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.end_headers(); self.wfile.write(html.encode())


def floor_to_increment(qty, increment):
    try:
        inc = Decimal(str(increment))
        if inc <= 0:
            return float(qty)
        value = (Decimal(str(qty)) / inc).to_integral_value(rounding=ROUND_DOWN) * inc
        return float(value)
    except Exception:
        return 0.0


def manage_positions(client, ml, positions, cfg):
    details = []
    for p in positions:
        pos_id = p.get("id")
        pair = str(p.get("pair", ""))
        qty = float(p.get("active_pos", 0) or 0)
        if abs(qty) < 1e-8:
            continue
        side = "LONG" if qty > 0 else "SHORT"
        symbol = pair.replace("B-", "").replace("_USDT", "USDT")
        entry = float(p.get("avg_price", 0) or 0)
        price = ml.price_cache.get(symbol, entry)
        tp = float(p.get("take_profit_trigger") or 0)
        sl = float(p.get("stop_loss_trigger") or 0)
        lev = int(float(p.get("leverage") or cfg.get("leverage", 15)))
        # If exchange position exists without bracket protection, attach protection immediately.
        if entry > 0 and (tp <= 0 or sl <= 0):
            precision = 8 if entry < 1 else 4
            new_tp = round(entry * (1.015 if side == "LONG" else 0.985), precision)
            new_sl = round(entry * (0.990 if side == "LONG" else 1.010), precision)
            ok = client.set_tpsl(pos_id, new_tp, new_sl)
            log.warning("Existing position TP/SL attach result=%s pair=%s; verify in CoinDCX app", ok, pair)
            if ok:
                tp, sl = new_tp, new_sl
        pnl_inr = 0.0
        if price > 0 and entry > 0:
            pnl_usdt = (price-entry)*abs(qty) if side == "LONG" else (entry-price)*abs(qty)
            pnl_inr = pnl_usdt * float(cfg.get("inr_per_usdt", 91.5))
        # Software fallback only; exchange-side TP/SL remains primary.
        exit_hit = (side == "LONG" and ((tp > 0 and price >= tp) or (sl > 0 and price <= sl))) or (side == "SHORT" and ((tp > 0 and price <= tp) or (sl > 0 and price >= sl)))
        if exit_hit and price > 0:
            log.warning("Software failsafe exit condition met for %s at %.8f", pair, price)
            try:
                client.exit_position(pos_id)
            except Exception as exc:
                log.error("Failsafe exit request failed: %s", exc)
        details.append({"id": pos_id, "pair": pair, "symbol": symbol, "side": side, "quantity": abs(qty), "entry_price": entry, "current_price": price, "leverage": lev, "take_profit": tp, "stop_loss": sl, "unrealized_pnl_est_inr": round(pnl_inr,2)})
    return details


def main():
    cfg = load_config()
    client = CoinDCX(cfg)
    ml = MLEngine(client)
    server = ThreadingHTTPServer(("0.0.0.0", 8080), Dashboard)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    log.info("CoinDCX ML Bot v3 started. Port 8080. Live orders=%s", cfg.get("auto_trade_enabled", False))

    while True:
        try:
            cfg = load_config()
            client.update(cfg)
            state["auto_trade_enabled"] = bool(cfg.get("auto_trade_enabled", False))
            state["selected_pairs"] = list(ml.pairs)
            state["active_universe"] = list(ml.pairs)
            state["last_scan_attempt_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            state["last_scan_status"] = "STARTED"
            fx = float(cfg.get("inr_per_usdt", 91.5))
            if not client.key or not client.secret:
                state["status"] = "API_CREDENTIALS_MISSING"
                time.sleep(30)
                continue

            try:
                wallet = client.wallet(fx)
                state["futures_balance_inr"] = wallet["total"]
                state["free_balance_inr"] = wallet["free"]
                state["locked_balance_inr"] = wallet["locked"]
                state["api_configured"] = True
                state["api_status_msg"] = f"Total ₹{wallet['total']:,.2f}; Free ₹{wallet['free']:,.2f}; Locked ₹{wallet['locked']:,.2f}"
            except Exception as exc:
                wallet = {"total": 0, "free": 0, "locked": 0}
                state["api_configured"] = False
                state["api_status_msg"] = str(exc)
                log.error("Wallet read failed: %s", exc)

            ml.refresh_universe()
            ml.train(cfg)
            if ml.is_trained:
                symbols = list(ml.selected_assets)
                state["selected_pairs"] = list(ml.pairs)
                state["model_pairs"] = symbols
                scans = {}
                for sym in symbols:
                    scans[sym] = ml.analyze(sym, cfg)
                state["latest_scans"] = scans
                state["last_scan_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                state["last_scan_status"] = "COMPLETED"
                store.resolve_due(ml.price_cache)
            else:
                scans = {}
                state["latest_scans"] = scans
                state["selected_pairs"] = list(ml.pairs)
                state["model_pairs"] = []
                state["last_scan_status"] = "BLOCKED_ML_NOT_VALIDATED"

            positions = client.positions()
            details = manage_positions(client, ml, positions, cfg)
            state["active_position_details"] = details
            state["active_position"] = "; ".join(f"{x['pair']} {x['side']} {x['leverage']}x PnL~₹{x['unrealized_pnl_est_inr']:+.2f}" for x in details) if details else None

            if positions:
                state["status"] = "POSITION_OPEN_NO_NEW_ENTRY"
            elif not ml.is_trained:
                state["status"] = "ML_NOT_VALIDATED_NO_ENTRY"
            else:
                state["status"] = "SCANNING_MARKET"
                auto = bool(cfg.get("auto_trade_enabled", False))
                max_day = int(cfg.get("max_daily_trades", 3))
                cooldown = int(cfg.get("cooldown_minutes", 30))*60
                daily_count = store.trade_count_today()
                cooldown_ok = time.time()-store.last_trade_ts() >= cooldown
                if auto and daily_count < max_day and cooldown_ok:
                    for sym, sig in scans.items():
                        if sig.get("action") not in ("BUY", "SELL"):
                            continue
                        symbol = sym
                        pair = "B-" + symbol.replace("USDT", "") + "_USDT"
                        px = float(sig.get("price", 0) or 0)
                        candle_ts = int(sig.get("candle_ts", 0) or 0)
                        if px <= 0 or not candle_ts or time.time() - candle_ts > int(cfg.get("max_signal_age_seconds", 600)):
                            state["last_risk_block"] = f"Stale or invalid signal blocked: {symbol}"
                            log.warning(state["last_risk_block"])
                            continue
                        if sig.get("tp") is None or sig.get("sl") is None:
                            state["last_risk_block"] = f"Signal missing TP/SL; entry blocked: {symbol}"
                            log.error(state["last_risk_block"])
                            continue
                        if not store.reserve_order_intent(symbol, candle_ts, sig["action"]):
                            state["last_risk_block"] = f"Duplicate candle order intent blocked: {symbol}"
                            log.warning(state["last_risk_block"])
                            continue
                        free = float(wallet.get("free", 0))
                        min_free = float(cfg.get("min_free_balance_inr", 500))
                        margin_cap = float(cfg.get("max_trade_margin_inr", 2000))
                        margin = min(margin_cap, free*0.50)
                        if free < min_free or margin < min_free:
                            state["last_execution_error"] = f"Free balance too low: ₹{free:.2f}"
                            break
                        rules = ml.instrument_rules(pair)
                        inc = rules.get("quantity_increment", rules.get("min_quantity", 0.001))
                        min_qty = float(rules.get("min_quantity", rules.get("min_trade_size", inc or 0.001)) or 0.001)
                        lev_key = "max_leverage_long" if sig["action"] == "BUY" else "max_leverage_short"
                        max_lev = float(rules.get(lev_key, cfg.get("leverage", 15)) or cfg.get("leverage", 15))
                        lev = min(int(cfg.get("leverage", 15)), int(max_lev))
                        if lev < 1:
                            state["last_execution_error"] = f"Invalid leverage for {pair}"
                            break
                        notional_usdt = (margin*lev)/fx
                        qty = floor_to_increment(notional_usdt/px, inc)
                        if qty < min_qty:
                            state["last_execution_error"] = f"Quantity below exchange minimum for {pair}: {qty} < {min_qty}"
                            continue
                        action = sig["action"]
                        try:
                            log.warning("LIVE ORDER attempt: %s %s qty=%s leverage=%sx margin~₹%.0f confidence=%s", pair, action, qty, lev, margin, sig.get("confidence"))
                            result = client.place_order(pair, action, qty, lev)
                            # Check explicit API rejection where response includes status/success.
                            if isinstance(result, dict) and (result.get("success") is False or str(result.get("status", "")).lower() in ("rejected", "failed")):
                                raise RuntimeError(f"Order rejected: {result}")
                            order_id = ""
                            if isinstance(result, dict):
                                order_id = result.get("id") or result.get("order_id") or ""
                                nested = result.get("data")
                                if not order_id and isinstance(nested, dict):
                                    order_id = nested.get("id") or nested.get("order_id") or ""
                            if not order_id and not (isinstance(result, dict) and result.get("success") is True):
                                store.finish_order_intent(symbol, candle_ts, action, "AMBIGUOUS")
                                raise RuntimeError(f"Order response did not confirm acceptance; manual exchange check required: {str(result)[:300]}")
                            store.finish_order_intent(symbol, candle_ts, action, "ACKNOWLEDGED", order_id)
                            store.add_trade(symbol, action, float(sig.get("confidence",0)), order_id)
                            state["last_execution_error"] = None
                            state["trade_history"].insert(0, {"symbol": symbol, "side": action, "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "confidence": sig.get("confidence"), "order_id": order_id})
                            state["trade_history"] = state["trade_history"][:100]
                            log.warning("Order API response: %s", result)
                            time.sleep(3)
                            latest_positions = client.positions()
                            matched = [p for p in latest_positions if p.get("pair") == pair and abs(float(p.get("active_pos",0) or 0)) > 0]
                            if matched:
                                mp = matched[0]
                                # Use signal-derived stop levels; API response must explicitly confirm TP and SL.
                                attached = client.set_tpsl(mp.get("id"), round(float(sig["tp"]), 8), round(float(sig["sl"]), 8))
                                if not attached:
                                    state["last_execution_error"] = "ENTRY SENT BUT TP/SL NOT CONFIRMED — emergency exit requested"
                                    log.critical(state["last_execution_error"])
                                    try:
                                        exit_result = client.exit_position(mp.get("id"))
                                        log.critical("Emergency exit response after TP/SL failure: %s", str(exit_result)[:500])
                                        time.sleep(2)
                                        still_open = [p for p in client.positions() if p.get("pair") == pair and abs(float(p.get("active_pos",0) or 0)) > 0]
                                        if still_open:
                                            state["last_execution_error"] = "CRITICAL: TP/SL unconfirmed and emergency exit not verified. Check CoinDCX immediately."
                                            log.critical(state["last_execution_error"])
                                        else:
                                            state["last_execution_error"] = "TP/SL unconfirmed; emergency exit request sent and position no longer reported open. Verify exchange history."
                                    except Exception as exit_exc:
                                        state["last_execution_error"] = f"CRITICAL: TP/SL unconfirmed and emergency exit failed: {exit_exc}. Check CoinDCX immediately."
                                        log.critical(state["last_execution_error"])
                            else:
                                state["last_execution_error"] = "Order response acknowledged but open position not confirmed; automatic retry blocked. Check CoinDCX immediately."
                                log.critical(state["last_execution_error"])
                                store.finish_order_intent(symbol, candle_ts, action, "ACKNOWLEDGED_POSITION_UNCONFIRMED", order_id)
                            break
                        except Exception as exc:
                            state["last_execution_error"] = str(exc)
                            try:
                                store.finish_order_intent(symbol, candle_ts, action, "ERROR_OR_AMBIGUOUS")
                            except Exception:
                                pass
                            log.error("Order execution failed: %s", exc)
                            break
                elif auto and daily_count >= max_day:
                    state["status"] = "DAILY_TRADE_LIMIT_REACHED"
                elif auto and not cooldown_ok:
                    state["status"] = "TRADE_COOLDOWN"
                elif not auto:
                    state["status"] = "AUTO_TRADE_DISABLED"

            state["learning_stats"] = store.stats()
        except Exception as exc:
            log.exception("Main loop error: %s", exc)
            state["last_execution_error"] = str(exc)
            state["status"] = "ERROR_CHECK_LOG"
        interval = max(30, int(load_config().get("scan_interval_seconds", 60)))
        time.sleep(interval)


if __name__ == "__main__":
    main()
PYEOF

"$PY" -m py_compile "$TMP"
cp "$TMP" "$DEST"
rm -f "$TMP"
"$PY" -m py_compile "$DEST"

echo ""
echo "SUCCESS: New bot candidate installed at: $DEST"
echo "Existing learning database ($APP_DIR/learning.db) is preserved; telemetry tables are created additively."
echo "Existing bot.py and its running process were NOT replaced or restarted."
echo "This is intentional to avoid interrupting an open Futures position."
if pgrep -af '[p]ython.*coindcx-bot/bot.py' >/dev/null 2>&1; then
  echo "An existing bot.py process appears to be running. Leave it running while a position is open."
else
  echo "No bot.py process detected. Still verify CoinDCX has no open position before switching."
fi
echo ""
echo "After ALL CoinDCX positions are closed and you are ready to switch, run:"
echo "  cp '$APP_DIR/bot.py' '$APP_DIR/bot.py.bak.before_v3' 2>/dev/null || true"
echo "  cp '$DEST' '$APP_DIR/bot.py'"
echo "  nohup '$PY' '$APP_DIR/bot.py' >> '$APP_DIR/bot.log' 2>&1 </dev/null &"
echo ""
echo "Do not run the launch command while the old bot is still running. Check with:"
echo "  pgrep -af '[b]ot.py'"

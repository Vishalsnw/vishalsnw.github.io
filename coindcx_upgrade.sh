#!/usr/bin/env bash
set -euo pipefail
APP_DIR="/root/coindcx-bot"
BOT="$APP_DIR/bot.py"
mkdir -p "$APP_DIR"
if [ -f "$BOT" ]; then cp "$BOT" "$BOT.bak.$(date +%Y%m%d_%H%M%S)"; fi
cat << 'PYEOF' > "$BOT"
import os, sys, time, json, hmac, hashlib, threading, math, logging
import requests, numpy as np, pandas as pd
from http.server import HTTPServer, BaseHTTPRequestHandler
from sklearn.ensemble import HistGradientBoostingClassifier

CONFIG_FILE = "/root/coindcx-bot/config.json"
LOG_FILE = "/root/coindcx-bot/bot.log"
DEFAULT_PAIRS = ["ADAUSDT", "XRPUSDT", "DOGEUSDT", "ETHUSDT", "BTCUSDT", "SOLUSDT", "PEPEUSDT", "SUIUSDT"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(), logging.FileHandler(LOG_FILE)])
logger = logging.getLogger("StrategicAlpha")

bot_state = {
    "status": "SCANNING_MARKET",
    "last_scan_time": "",
    "futures_balance_inr": 0.0,
    "free_balance_inr": 0.0,
    "locked_balance_inr": 0.0,
    "active_position": None,
    "active_position_details": None,
    "api_configured": False,
    "api_status_msg": "",
    "last_execution_error": None,
    "latest_scans": {},
    "trade_history": []
}

def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE) as f: return json.load(f)
        except Exception: pass
    return {"api_key": "", "api_secret": "", "max_trade_margin_inr": 2000.0, "leverage": 15, "auto_trade_enabled": True}

class CoinDCXClient:
    BASE_URL = "https://api.coindcx.com"
    def __init__(self, key, secret):
        self.api_key = key.strip(); self.api_secret = secret.strip()
    def update_keys(self, k, s): self.api_key = k.strip(); self.api_secret = s.strip()
    def _sign(self, body): return hmac.new(self.api_secret.encode('utf-8'), body.encode('utf-8'), hashlib.sha256).hexdigest()
    def _headers(self, sig): return {"Content-Type": "application/json", "X-AUTH-APIKEY": self.api_key, "X-AUTH-SIGNATURE": sig, "User-Agent": "StrategicAlpha/2.0"}

    def get_wallet_info(self):
        if not self.api_key: return {"total": 0.0, "free": 0.0, "locked": 0.0}
        ts = int(time.time() * 1000)
        p = json.dumps({"timestamp": ts}, separators=(",", ":"))
        total = free = locked = 0.0
        try:
            r = requests.get(f"{self.BASE_URL}/exchange/v1/derivatives/futures/wallets", data=p, headers=self._headers(self._sign(p)), timeout=8)
            if r.status_code == 200:
                for w in r.json():
                    curr = str(w.get("currency_short_name", "")).upper()
                    bal = float(w.get("balance", 0.0) or 0.0)
                    lk = float(w.get("locked_balance", 0.0) or 0.0)
                    rate = 1.0 if curr == "INR" else float(load_config().get("inr_per_usdt", 91.5))
                    total += (bal + lk) * rate
                    free += bal * rate
                    locked += lk * rate
        except Exception: pass
        return {"total": round(total, 2), "free": round(free, 2), "locked": round(locked, 2)}

    def get_active_positions(self):
        if not self.api_key: return []
        seen = set(); positions = []
        for m in [["INR"], ["USDT"], "INR", "USDT", None]:
            for page in [1, 2]:
                ts = int(time.time() * 1000)
                d = {"timestamp": ts, "page": page, "size": 100}
                if m is not None: d["margin_currency_short_name"] = m
                p = json.dumps(d, separators=(",", ":"))
                try:
                    r = requests.post(f"{self.BASE_URL}/exchange/v1/derivatives/futures/positions", data=p, headers=self._headers(self._sign(p)), timeout=8)
                    if r.status_code == 200 and isinstance(r.json(), list):
                        for it in r.json():
                            pid = it.get("id")
                            if pid in seen: continue
                            seen.add(pid)
                            qty = abs(float(it.get("active_pos", 0.0) or 0.0))
                            lm = float(it.get("locked_margin", 0.0) or 0.0)
                            lum = float(it.get("locked_user_margin", 0.0) or 0.0)
                            if qty > 0.000001 or lm > 0.01 or lum > 0.01:
                                positions.append(it)
                except Exception: pass
        return positions

    def set_position_tpsl(self, pos_id, tp, sl):
        if not self.api_key or not pos_id: return False
        ts = str(int(round(time.time() * 1000)))
        body = json.dumps({
            "timestamp": ts,
            "id": pos_id,
            "take_profit": {"stop_price": str(tp), "order_type": "take_profit_market"},
            "stop_loss": {"stop_price": str(sl), "order_type": "stop_market"}
        }, separators=(",", ":"))
        try:
            r = requests.post(f"{self.BASE_URL}/exchange/v1/derivatives/futures/positions/create_tpsl", data=body, headers=self._headers(self._sign(body)), timeout=8)
            logger.info(f"CoinDCX create_tpsl for {pos_id}: {r.status_code} - {r.text}")
            return r.status_code in (200, 201)
        except Exception as e:
            logger.error(f"create_tpsl error: {e}")
            return False

    def exit_position(self, pos_id):
        if not self.api_key or not pos_id: return False
        ts = int(round(time.time() * 1000))
        body = json.dumps({"timestamp": ts, "id": pos_id}, separators=(",", ":"))
        try:
            r = requests.post(f"{self.BASE_URL}/exchange/v1/derivatives/futures/positions/exit", data=body, headers=self._headers(self._sign(body)), timeout=8)
            logger.info(f"CoinDCX position exit {pos_id}: {r.status_code} - {r.text}")
            return r.status_code in (200, 201)
        except Exception as e:
            logger.error(f"exit error: {e}")
            return False

    def format_pair(self, s):
        c = s.upper().replace("B-", "").replace("_USDT", "")
        if c.endswith("USDT"): c = c[:-4]
        if c in ("PEPE", "1000PEPE"): return "B-1000PEPE_USDT"
        if c in ("SHIB", "1000SHIB"): return "B-1000SHIB_USDT"
        if c in ("FLOKI", "1000FLOKI"): return "B-1000FLOKI_USDT"
        return f"B-{c}_USDT"

    def place_order(self, symbol, side, qty, leverage=15):
        pair = self.format_pair(symbol)
        errs = []
        for margin_curr in ["INR"]:
            ts = int(time.time() * 1000)
            p = json.dumps({"timestamp": ts, "order": {"pair": pair, "side": side.lower(), "order_type": "market_order", "total_quantity": qty, "leverage": leverage, "margin_currency_short_name": margin_curr}}, separators=(",", ":"))
            try:
                r = requests.post(f"{self.BASE_URL}/exchange/v1/derivatives/futures/orders/create", data=p, headers=self._headers(self._sign(p)), timeout=10)
                if r.status_code in (200, 201):
                    logger.info(f"Order SUCCESS on {pair} ({side}): {r.text}")
                    return r.json()
                else: errs.append(f"{margin_curr}: {r.text}")
            except Exception as e: errs.append(str(e))
        raise Exception(" | ".join(errs))

class MLEngine:
    """CoinDCX Futures candle model: SHORT / HOLD / LONG, with time-ordered validation."""
    FEATURES = ["ret1", "ret3", "ret6", "rsi", "ema_gap", "ema_slope", "bb_pos",
                "atr_pct", "vol_ratio", "upper_wick", "lower_wick", "body_ratio", "macd_norm"]

    def __init__(self):
        self.model = HistGradientBoostingClassifier(max_iter=160, learning_rate=0.045,
                                                     max_leaf_nodes=15, l2_regularization=1.5,
                                                     random_state=42)
        self.is_trained = False
        self.validation_score = 0.0
        self.cached_prices = {}
        self.last_train = 0
        self.session = requests.Session()

    def fetch_candles(self, symbol, interval="5", limit=400):
        # CoinDCX public futures candles; pair format matches futures instruments.
        resolution = {"5m": "5", "15m": "15", "1h": "60", "4h": "240"}.get(str(interval), str(interval))
        pair = CoinDCXClient.format_pair(self, symbol)
        now = int(time.time())
        frm = now - max(100, int(limit) * int(resolution) * 60 + 3600)
        url = "https://public.coindcx.com/market_data/candlesticks"
        try:
            r = self.session.get(url, params={"pair": pair, "from": frm, "to": now,
                                               "resolution": resolution, "pcode": "f"}, timeout=10)
            r.raise_for_status()
            raw = r.json()
            if isinstance(raw, dict):
                raw = raw.get("data", raw.get("candlesticks", []))
            df = pd.DataFrame(raw)
            if df.empty: return pd.DataFrame()
            aliases = {"time": "timestamp", "open": "open", "high": "high", "low": "low",
                       "close": "close", "volume": "volume"}
            df = df.rename(columns=aliases)
            required = ["timestamp", "open", "high", "low", "close", "volume"]
            if not all(c in df.columns for c in required):
                logger.warning("CoinDCX candle schema unexpected for %s: %s", symbol, list(df.columns))
                return pd.DataFrame()
            df = df[required].copy()
            for c in required: df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.dropna().sort_values("timestamp").drop_duplicates("timestamp")
            # API timestamps may be seconds or milliseconds.
            if not df.empty and df.timestamp.iloc[-1] > 1e12: df["timestamp"] /= 1000.0
            df = df.tail(limit).reset_index(drop=True)
            if len(df): self.cached_prices[symbol] = float(df.close.iloc[-1])
            return df
        except Exception as e:
            logger.warning("Futures candle fetch failed %s %s: %s", symbol, resolution, e)
            return pd.DataFrame()

    def compute_features(self, source):
        if source is None or len(source) < 60: return pd.DataFrame()
        df = source.copy()
        c, h, l, o, v = df.close, df.high, df.low, df.open, df.volume
        delta = c.diff()
        gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
        df["rsi"] = 100 - 100 / (1 + gain / (loss + 1e-9))
        ema9 = c.ewm(span=9, adjust=False).mean()
        ema21 = c.ewm(span=21, adjust=False).mean()
        ema50 = c.ewm(span=50, adjust=False).mean()
        df["ema_gap"] = (ema9 - ema21) / (c + 1e-9)
        df["ema_slope"] = (ema21 - ema50) / (c + 1e-9)
        df["ret1"] = c.pct_change(1)
        df["ret3"] = c.pct_change(3)
        df["ret6"] = c.pct_change(6)
        mid = c.rolling(20).mean(); sd = c.rolling(20).std()
        df["bb_pos"] = (c - (mid - 2*sd)) / (4*sd + 1e-9)
        tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/14, adjust=False).mean()
        df["atr"] = atr
        df["atr_pct"] = atr / (c + 1e-9)
        df["vol_ratio"] = v / (v.rolling(20).mean() + 1e-9)
        rng = (h-l).replace(0, np.nan)
        df["upper_wick"] = (h - np.maximum(o,c)) / rng
        df["lower_wick"] = (np.minimum(o,c) - l) / rng
        df["body_ratio"] = (c-o).abs() / rng
        macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
        df["macd_norm"] = macd / (c + 1e-9)
        return df.replace([np.inf,-np.inf], np.nan).dropna().reset_index(drop=True)

    def _dataset_for(self, symbol):
        d = self.fetch_candles(symbol, "5", 700)
        f = self.compute_features(d)
        if len(f) < 180: return None, None
        # 15-minute forward return; volatility-scaled neutral band avoids forcing weak moves.
        future = f.close.shift(-3) / f.close - 1.0
        band = (f.atr_pct * 0.85).clip(lower=0.0018, upper=0.012)
        y = pd.Series(1, index=f.index, dtype=int)  # HOLD
        y[future > band] = 2   # LONG
        y[future < -band] = 0  # SHORT
        valid = future.notna()
        return f.loc[valid, self.FEATURES], y.loc[valid]

    def train_baseline(self):
        if self.is_trained and time.time() - self.last_train < 1800: return
        xs, ys = [], []
        for symbol in ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT"]:
            x, y = self._dataset_for(symbol)
            if x is not None and len(x):
                xs.append(x); ys.append(y)
        if not xs:
            self.is_trained = False
            logger.warning("ML training skipped: no usable CoinDCX futures candles")
            return
        X = pd.concat(xs, ignore_index=True)
        Y = pd.concat(ys, ignore_index=True)
        if Y.nunique() < 3 or len(X) < 500:
            self.is_trained = False
            logger.warning("ML training skipped: insufficient samples/classes")
            return
        # Chronological holdout; do not report a score from a random/leaky split.
        cut = int(len(X) * 0.80)
        probe = HistGradientBoostingClassifier(max_iter=120, learning_rate=0.05,
                                                max_leaf_nodes=15, l2_regularization=1.5,
                                                random_state=42)
        try:
            probe.fit(X.iloc[:cut], Y.iloc[:cut])
            pred = probe.predict(X.iloc[cut:])
            from sklearn.metrics import balanced_accuracy_score
            score = float(balanced_accuracy_score(Y.iloc[cut:], pred))
            if not np.isfinite(score) or score < 0.36:
                self.is_trained = False
                self.validation_score = score
                logger.warning("ML holdout too weak (balanced accuracy %.3f); live entries disabled", score)
                return
            self.model.fit(X, Y)
            self.is_trained = True
            self.validation_score = score
            self.last_train = time.time()
            logger.info("CoinDCX Futures ML trained: %d rows, holdout balanced accuracy %.3f", len(X), score)
        except Exception as e:
            self.is_trained = False
            logger.exception("ML training failed: %s", e)

    def analyze_symbol(self, s):
        if not self.is_trained or time.time() - self.last_train >= 1800: self.train_baseline()
        df = self.fetch_candles(s, "5", 240)
        if df.empty or len(df) < 80:
            return {"symbol": s, "action": "HOLD", "price": self.cached_prices.get(s,0), "confidence": 0.0, "reason": "No CoinDCX futures candles"}
        # Exclude potentially incomplete live candle for features and decision.
        f = self.compute_features(df.iloc[:-1].copy())
        if f.empty:
            return {"symbol": s, "action": "HOLD", "price": float(df.close.iloc[-1]), "confidence": 0.0, "reason": "Insufficient features"}
        curr = f.iloc[-1]
        price, atr = float(curr.close), float(curr.atr)
        self.cached_prices[s] = price
        if not self.is_trained:
            return {"symbol": s, "action": "HOLD", "price": price, "confidence": 0.0, "reason": "ML validation/training unavailable; no live entry"}
        x = pd.DataFrame([{k: float(curr[k]) for k in self.FEATURES}])
        probs = self.model.predict_proba(x)[0]
        pmap = {int(c): float(v) for c,v in zip(self.model.classes_, probs)}
        p_short, p_hold, p_long = pmap.get(0,0), pmap.get(1,0), pmap.get(2,0)
        # Higher-timeframe alignment from same CoinDCX Futures feed.
        h1 = self.fetch_candles(s, "60", 100)
        h4 = self.fetch_candles(s, "240", 80)
        def trend(d):
            if d.empty or len(d) < 55: return 0
            c = d.close.iloc[:-1] if len(d)>2 else d.close
            e9 = c.ewm(span=9, adjust=False).mean().iloc[-1]
            e21 = c.ewm(span=21, adjust=False).mean().iloc[-1]
            e50 = c.ewm(span=50, adjust=False).mean().iloc[-1]
            return 1 if e9>e21>e50 else (-1 if e9<e21<e50 else 0)
        t1, t4 = trend(h1), trend(h4)
        sl_dist, tp_dist = max(1.5*atr, price*0.002), max(2.3*atr, price*0.003)
        if p_long >= 0.68 and p_long > p_short and t1 == 1 and t4 >= 0:
            return {"symbol":s,"action":"BUY","price":price,"confidence":round(p_long*100,1),"sl":round(price-sl_dist,8),"tp":round(price+tp_dist,8),"reason":f"Futures ML LONG {p_long:.0%}; H1/H4 aligned; val={self.validation_score:.2f}"}
        if p_short >= 0.68 and p_short > p_long and t1 == -1 and t4 <= 0:
            return {"symbol":s,"action":"SELL","price":price,"confidence":round(p_short*100,1),"sl":round(price+sl_dist,8),"tp":round(price-tp_dist,8),"reason":f"Futures ML SHORT {p_short:.0%}; H1/H4 aligned; val={self.validation_score:.2f}"}
        return {"symbol":s,"action":"HOLD","price":price,"confidence":round(max(p_short,p_hold,p_long)*100,1),"reason":f"No aligned edge S/H/L={p_short:.2f}/{p_hold:.2f}/{p_long:.2f}"}

class Handler(BaseHTTPRequestHandler):
    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*"); self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS"); self.send_header("Access-Control-Allow-Headers", "*")
    def do_OPTIONS(self): self.send_response(200); self._cors(); self.end_headers()
    def do_GET(self):
        if self.path == "/api/status":
            self.send_response(200); self.send_header("Content-Type", "application/json"); self._cors(); self.end_headers()
            self.wfile.write(json.dumps(bot_state).encode("utf-8"))
            return
        self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self._cors(); self.end_headers()
        self.wfile.write(f"<html><body style='background:#0B0E14;color:#fff;font-family:sans-serif;padding:20px;'><h2>⚡ Strategic Alpha Cloud Bot</h2><p>Balance: ₹{bot_state['futures_balance_inr']:,.2f} (Free: ₹{bot_state['free_balance_inr']:,.2f})</p><p>Status: {bot_state['status']}</p><p>Active Position: {bot_state['active_position'] or 'None'}</p></body></html>".encode("utf-8"))

def main():
    threading.Thread(target=lambda: HTTPServer(("0.0.0.0", 8080), Handler).serve_forever(), daemon=True).start()
    conf = load_config()
    client = CoinDCXClient(conf.get("api_key", ""), conf.get("api_secret", ""))
    ml = MLEngine()
    PREC = {"BTCUSDT": 3, "ETHUSDT": 3, "SOLUSDT": 2, "BNBUSDT": 2, "XRPUSDT": 1, "DOGEUSDT": 0, "ADAUSDT": 1, "PEPEUSDT": 0, "SUIUSDT": 1}

    logger.info("Strategic Alpha Futures Autopilot initialized. Live orders require valid config.json credentials.")

    while True:
        try:
            conf = load_config()
            client.update_keys(conf.get("api_key", ""), conf.get("api_secret", ""))
            
            # 1. Update wallet balances
            winfo = client.get_wallet_info()
            bot_state["futures_balance_inr"] = winfo["total"]
            bot_state["free_balance_inr"] = winfo["free"]
            bot_state["locked_balance_inr"] = winfo["locked"]
            bot_state["api_configured"] = winfo["total"] > 0
            bot_state["api_status_msg"] = f"Total: ₹{winfo['total']:,.2f} (Free: ₹{winfo['free']:,.2f} | Locked: ₹{winfo['locked']:,.2f})" if winfo["locked"] > 0 else f"Available: ₹{winfo['free']:,.2f}"

            # 2. ALWAYS Scan all pairs to keep radar alive & get live prices
            for sym in DEFAULT_PAIRS:
                sig = ml.analyze_symbol(sym)
                bot_state["latest_scans"][sym] = sig
            bot_state["last_scan_time"] = time.strftime("%Y-%m-%d %H:%M:%S")

            # 3. Check active positions
            positions = client.get_active_positions()
            if positions:
                p = positions[0]
                pos_id = p.get("id")
                pair = p.get("pair", "")
                raw_qty = float(p.get("active_pos", 0.0) or 0.0)
                side = "LONG" if raw_qty >= 0 else "SHORT"
                lev = int(p.get("leverage", 15))
                entry_price = float(p.get("avg_price", p.get("entry_price", 0.0)) or 0.0)
                tp_curr = float(p.get("take_profit_trigger", 0.0) or 0.0)
                sl_curr = float(p.get("stop_loss_trigger", 0.0) or 0.0)

                # Get clean symbol for price lookup
                clean_sym = pair.replace("B-", "").replace("_USDT", "USDT")
                curr_price = ml.cached_prices.get(clean_sym, entry_price)

                # Calculate real-time live PnL in INR
                pnl_inr = 0.0
                pnl_pct = 0.0
                if curr_price > 0 and entry_price > 0:
                    if side == "LONG":
                        pnl_usdt = (curr_price - entry_price) * abs(raw_qty)
                    else:
                        pnl_usdt = (entry_price - curr_price) * abs(raw_qty)
                    pnl_inr = pnl_usdt * float(conf.get("inr_per_usdt", 91.5))
                    margin_usdt = (abs(raw_qty) * entry_price) / max(1, lev)
                    pnl_pct = (pnl_usdt / margin_usdt) * 100 if margin_usdt > 0 else 0.0

                price_prec = 4 if entry_price < 1.0 else 2
                # Ensure TP & SL are attached to CoinDCX
                if tp_curr == 0.0 or sl_curr == 0.0:
                    calc_tp = round(entry_price * 1.015, price_prec) if side == "LONG" else round(entry_price * 0.985, price_prec)
                    calc_sl = round(entry_price * 0.990, price_prec) if side == "LONG" else round(entry_price * 1.010, price_prec)
                    logger.info(f"⚡ Attaching TP: ${calc_tp} | SL: ${calc_sl} for {pair} to CoinDCX...")
                    if client.set_position_tpsl(pos_id, calc_tp, calc_sl):
                        tp_curr, sl_curr = calc_tp, calc_sl

                # Failsafe exit check in software
                if tp_curr > 0 and sl_curr > 0 and curr_price > 0:
                    should_exit = False
                    exit_reason = ""
                    if side == "SHORT":
                        if curr_price <= tp_curr:
                            should_exit = True; exit_reason = f"Take Profit Hit (${curr_price} <= ${tp_curr})"
                        elif curr_price >= sl_curr:
                            should_exit = True; exit_reason = f"Stop Loss Hit (${curr_price} >= ${sl_curr})"
                    elif side == "LONG":
                        if curr_price >= tp_curr:
                            should_exit = True; exit_reason = f"Take Profit Hit (${curr_price} >= ${tp_curr})"
                        elif curr_price <= sl_curr:
                            should_exit = True; exit_reason = f"Stop Loss Hit (${curr_price} <= ${sl_curr})"
                    
                    if should_exit:
                        logger.info(f"🎯 Failsafe Exit Triggered: {exit_reason}! Exiting position {pair}...")
                        client.exit_position(pos_id)
                        time.sleep(3)
                        continue

                bot_state["active_position"] = f"{pair} ({side} {lev}x | Entry: ${entry_price:,.4f} | TP: ${tp_curr:,.4f} | SL: ${sl_curr:,.4f} | PnL: ₹{pnl_inr:+,.2f})"
                bot_state["active_position_details"] = {
                    "id": pos_id,
                    "pair": pair,
                    "side": side,
                    "quantity": abs(raw_qty),
                    "leverage": lev,
                    "entry_price": entry_price,
                    "current_price": curr_price,
                    "take_profit": tp_curr,
                    "stop_loss": sl_curr,
                    "unrealized_pnl": round(pnl_inr, 2)
                }
                bot_state["status"] = "TRADE_IN_PROGRESS"
                bot_state["last_execution_error"] = None
            else:
                bot_state["active_position"] = None
                bot_state["active_position_details"] = None
                bot_state["status"] = "SCANNING_MARKET"

                # Check for high confidence trades if no active position
                for sym, sig in bot_state["latest_scans"].items():
                    act = sig.get("action"); conf_val = sig.get("confidence", 0)
                    if act in ("BUY", "SELL") and conf_val >= 72:
                        free_avail = winfo["free"]
                        if free_avail >= 500.0:
                            # ₹2,000 is the maximum allocated margin, not the maximum loss.
                            safe_margin = min(2000.0, free_avail * 0.70)
                            lev = 15
                            pr = float(sig.get("price", 0) or 0)
                            fx = float(conf.get("inr_per_usdt", 91.5))
                            notional_usdt = (safe_margin * lev) / fx
                            prec = PREC.get(sym.upper(), 3)
                            raw_qty = notional_usdt / pr if pr > 0 else 0
                            factor = 10 ** prec
                            qty = math.floor(raw_qty * factor) / factor
                            if qty > 0 and safe_margin >= 500.0:
                                try:
                                    logger.info(f"Auto-Executing: {act} {qty} {sym} (Margin: ~₹{safe_margin:.0f})...")
                                    client.place_order(sym, act, qty, leverage=lev)
                                    bot_state["last_execution_error"] = None
                                    bot_state["trade_history"].insert(0, {"symbol": sym, "side": act, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "confidence": conf_val}); bot_state["trade_history"] = bot_state["trade_history"][:100]
                                    time.sleep(2)
                                    pos_list = client.get_active_positions()
                                    if pos_list:
                                        matched = [q for q in pos_list if str(q.get("pair", "")).replace("B-", "").replace("_USDT", "USDT") == sym]
                                        if matched:
                                            client.set_position_tpsl(matched[0].get("id"), sig.get("tp", 0.0), sig.get("sl", 0.0))
                                        else:
                                            logger.error("Order acknowledged but matching open position not confirmed; TP/SL not attached")
                                    else:
                                        logger.error("Order acknowledged but no open position confirmed; TP/SL not attached")
                                    break
                                except Exception as e:
                                    bot_state["last_execution_error"] = str(e)

        except Exception as e:
            logger.error(f"Autopilot loop notice: {e}")
        time.sleep(5)

if __name__ == "__main__":
    main()

PYEOF
if ! python3 -m py_compile "$BOT"; then
  echo "Syntax check failed. Existing backup retained; bot was not restarted." >&2
  exit 1
fi
BOT_PIDS=$(pgrep -f "^python3 $BOT$" || true)
if [ -n "$BOT_PIDS" ]; then kill $BOT_PIDS || true; fi
sleep 2
nohup python3 "$BOT" >> "$APP_DIR/bot.log" 2>&1 </dev/null &
sleep 3
echo "Bot process:"
pgrep -af "^python3 $BOT$" || true
echo "Recent log:"
tail -n 25 "$APP_DIR/bot.log" || true

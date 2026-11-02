#!/usr/bin/env python3
"""
CoinDCX ML V4.1 research experiment — READ ONLY.

- Public market-data GET requests only.
- Does not import bot_v3, read API credentials, open SQLite, write config/model
  files, call private trading endpoints, or place orders.
- Compares label ATR multipliers and three class-weight strengths.
- Uses three chronological, non-overlapping forward test windows with expanding
  training history. Hyperparameters are not selected on the test windows.
- Includes a simple 15-minute close-to-close directional simulation with an
  explicitly configurable assumed round-trip cost. It is NOT an execution
  simulator and does not model the production bot's TP/SL or fills.

Run:
  /root/coindcx-bot/venv/bin/python ml_v41_experiment.py
"""
import time
import traceback
import requests
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, precision_recall_fscore_support,
)
from sklearn.utils.class_weight import compute_sample_weight

PUBLIC_URL = "https://public.coindcx.com"
ACTIVE_URL = "https://api.coindcx.com/exchange/v1/derivatives/futures/data/active_instruments"
ASSETS = ["BTC", "ETH", "SOL", "XRP", "BNB", "DOGE",
          "ADA", "LINK", "LTC", "BCH", "AVAX", "DOT"]
FEATURES = ["ret1", "ret3", "ret6", "rsi", "ema_gap", "ema_slope",
            "bb_pos", "atr_pct", "vol_ratio", "upper_wick",
            "lower_wick", "body_ratio", "macd_norm"]
LABELS = [0, 1, 2]
LABEL_NAMES = ["SELL", "HOLD", "BUY"]

# Label experiment. Existing V3 band is factor 1.00 of its original ATR rule.
LABEL_FACTORS = [0.65, 0.85, 1.00, 1.05]
# Alpha 0 = unweighted; 0.5 = mild square-root balanced weights;
# 1.0 = fully balanced sample weights.
WEIGHT_STRENGTHS = [0.0, 0.5, 1.0]

# Three non-overlapping chronological test windows; train expands each fold.
# Fractions are based on globally sorted candle timestamps.
FOLDS = [
    (0.00, 0.50, 0.50, 0.65),
    (0.00, 0.65, 0.65, 0.80),
    (0.00, 0.80, 0.80, 0.95),
]
ROUND_TRIP_COST_PCT = 0.12  # Assumption only; replace with verified all-in cost.
MIN_TRAIN_ROWS = 300
MIN_TEST_ROWS = 60
RANDOM_STATE = 42
TIMEOUT = 15


def fetch_pairs(session):
    r = session.get(ACTIVE_URL,
                    params={"margin_currency_short_name[]": "INR"},
                    timeout=TIMEOUT)
    r.raise_for_status()
    raw = r.json()
    active = set()
    if isinstance(raw, list):
        for item in raw:
            pair = item.get("pair") if isinstance(item, dict) else item
            if pair:
                active.add(str(pair).upper())
    return [f"B-{asset}_USDT" for asset in ASSETS
            if f"B-{asset}_USDT" in active]


def fetch_candles(session, pair, limit=700):
    now = int(time.time())
    start = now - (limit * 5 * 60 + 3600)
    r = session.get(
        PUBLIC_URL + "/market_data/candlesticks",
        params={"pair": pair, "from": start, "to": now,
                "resolution": "5", "pcode": "f"},
        timeout=TIMEOUT,
    )
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
    required = ["timestamp", "open", "high", "low", "close", "volume"]
    if not all(c in d.columns for c in required):
        raise ValueError(f"{pair}: unexpected candle schema {list(d.columns)}")
    for c in required:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = (d.dropna(subset=required).sort_values("timestamp")
         .drop_duplicates("timestamp").reset_index(drop=True))
    if not d.empty and d.timestamp.iloc[-1] > 1e12:
        d["timestamp"] /= 1000.0
    return d.tail(limit).reset_index(drop=True)


def make_features(src):
    if src is None or len(src) < 100:
        return pd.DataFrame()
    d = src.copy()
    c, h, lo, o, v = d.close, d.high, d.low, d.open, d.volume
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    d["rsi"] = 100 - 100 / (1 + gain / (loss + 1e-9))
    e9 = c.ewm(span=9, adjust=False).mean()
    e21 = c.ewm(span=21, adjust=False).mean()
    e50 = c.ewm(span=50, adjust=False).mean()
    d["ema_gap"] = (e9 - e21) / (c + 1e-9)
    d["ema_slope"] = (e21 - e50) / (c + 1e-9)
    d["ret1"] = c.pct_change(1)
    d["ret3"] = c.pct_change(3)
    d["ret6"] = c.pct_change(6)
    mid, sd = c.rolling(20).mean(), c.rolling(20).std()
    d["bb_pos"] = (c - (mid - 2 * sd)) / (4 * sd + 1e-9)
    tr = pd.concat([h - lo, (h - c.shift()).abs(),
                    (lo - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False).mean()
    d["atr"] = atr
    d["atr_pct"] = atr / (c + 1e-9)
    d["vol_ratio"] = v / (v.rolling(20).mean() + 1e-9)
    rng = (h - lo).replace(0, np.nan)
    d["upper_wick"] = (h - np.maximum(o, c)) / rng
    d["lower_wick"] = (np.minimum(o, c) - lo) / rng
    d["body_ratio"] = (c - o).abs() / rng
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    d["macd_norm"] = macd / (c + 1e-9)
    return d.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)


def make_panel(pair_frames, factor):
    rows = []
    for pair, f in pair_frames.items():
        future = f.close.shift(-3) / f.close - 1.0
        # V3 base band is ATR*0.85, clipped to [0.18%, 1.2%].
        band = (f.atr_pct * 0.85 * factor).clip(lower=0.0018, upper=0.012)
        y = pd.Series(1, index=f.index, dtype=int)
        y[future > band] = 2
        y[future < -band] = 0
        valid = future.notna()
        block = f.loc[valid, ["timestamp", "close"] + FEATURES].copy()
        block["symbol"] = pair
        block["future_return"] = future.loc[valid].values
        block["label"] = y.loc[valid].values
        rows.append(block)
    if not rows:
        return pd.DataFrame()
    panel = pd.concat(rows, ignore_index=True)
    panel = panel.replace([np.inf, -np.inf], np.nan).dropna(
        subset=FEATURES + ["timestamp", "close", "future_return", "label"])
    panel["label"] = panel["label"].astype(int)
    return panel.sort_values(["timestamp", "symbol"]).reset_index(drop=True)


def fold_masks(panel, train_end, test_start, test_end):
    # A 3-candle embargo around split boundaries avoids label horizons crossing
    # from the training period into the following evaluation period.
    times = np.sort(panel.timestamp.unique())
    n = len(times)
    i_train = max(1, int(n * train_end))
    i_test = max(i_train + 1, int(n * test_start))
    i_end = min(n, int(n * test_end))
    if i_end <= i_test:
        return None
    train_cut = times[i_train]
    test_cut = times[i_test]
    end_cut = times[i_end - 1]
    embargo = 3 * 5 * 60
    train = panel.timestamp < (train_cut - embargo)
    test = (panel.timestamp >= (test_cut + embargo)) & (panel.timestamp <= end_cut)
    return train, test, (train_cut, test_cut, end_cut)


def weights_for(y, strength):
    if strength == 0:
        return None
    full = compute_sample_weight(class_weight="balanced", y=y)
    return np.power(full, strength)


def fit_model(train, strength):
    if train.label.nunique() < 3:
        return None
    model = HistGradientBoostingClassifier(
        max_iter=120, learning_rate=0.05, max_leaf_nodes=15,
        l2_regularization=1.5, random_state=RANDOM_STATE,
    )
    model.fit(train[FEATURES], train.label,
              sample_weight=weights_for(train.label, strength))
    return model


def score_fold(train, test, strength):
    model = fit_model(train, strength)
    if model is None or test.label.nunique() < 2:
        return None
    pred = model.predict(test[FEATURES])
    actual = test.label.to_numpy()
    cm = confusion_matrix(actual, pred, labels=LABELS)
    precision, recall, f1, support = precision_recall_fscore_support(
        actual, pred, labels=LABELS, zero_division=0)
    # Simple directional 15m close-to-close P&L proxy; only directional
    # predictions trade. Cost is an explicit assumption, not a fee quote.
    pred = np.asarray(pred)
    returns = test.future_return.to_numpy()
    signed = np.where(pred == 2, returns,
             np.where(pred == 0, -returns, 0.0))
    traded = pred != 1
    net_pct = signed[traded] * 100.0 - ROUND_TRIP_COST_PCT
    return {
        "balanced_accuracy": balanced_accuracy_score(actual, pred),
        "accuracy": accuracy_score(actual, pred),
        "confusion": cm,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
        "trades": int(traded.sum()),
        "trade_rate_pct": float(traded.mean() * 100.0),
        "net_mean_pct": float(net_pct.mean()) if len(net_pct) else None,
        "net_sum_pct": float(net_pct.sum()) if len(net_pct) else None,
        "net_win_rate_pct": float((net_pct > 0).mean() * 100.0) if len(net_pct) else None,
        "n_test": len(test),
    }


def print_report(factor, strength, fold_no, result):
    method = "UNWEIGHTED" if strength == 0 else f"WEIGHT_ALPHA_{strength:.1f}"
    print(f"\n  Fold {fold_no} | label_factor={factor:.2f} | {method}")
    print(f"  rows={result['n_test']} trades={result['trades']} "
          f"trade_rate={result['trade_rate_pct']:.1f}%")
    print(f"  accuracy={result['accuracy']:.4f} "
          f"balanced_accuracy={result['balanced_accuracy']:.4f}")
    print("  Confusion matrix (actual rows, predicted columns; SELL/HOLD/BUY):")
    print(result["confusion"])
    for i, name in enumerate(LABEL_NAMES):
        print(f"  {name}: precision={result['precision'][i]:.3f} "
              f"recall={result['recall'][i]:.3f} f1={result['f1'][i]:.3f} "
              f"support={result['support'][i]}")
    print(f"  Directional proxy after assumed {ROUND_TRIP_COST_PCT:.3f}% "
          f"round-trip cost: mean={result['net_mean_pct']}, "
          f"sum={result['net_sum_pct']}, win_rate={result['net_win_rate_pct']}")


def main():
    print("=" * 76)
    print("COINDCX ML V4.1 — MULTI-WINDOW CONTROLLED RESEARCH")
    print("READ-ONLY PUBLIC DATA. NO CONFIG/DB/PRODUCTION MODEL/ORDERS.")
    print(f"Assumed round-trip cost: {ROUND_TRIP_COST_PCT:.3f}% (not a verified fee)")
    print("=" * 76)
    session = requests.Session()
    session.headers.update({"User-Agent": "CoinDCX-ML-V4.1-Research/1.0"})
    pairs = fetch_pairs(session)
    if not pairs:
        raise RuntimeError("No allowlisted INR futures contracts are active.")
    print("Active pairs:", ", ".join(pairs))

    pair_frames = {}
    for pair in pairs:
        try:
            raw = fetch_candles(session, pair)
            f = make_features(raw)
            if len(f) >= 180:
                pair_frames[pair] = f
                print(f"{pair}: {len(f)} usable feature rows")
            else:
                print(f"{pair}: skipped; only {len(f)} usable rows")
        except Exception as exc:
            print(f"{pair}: fetch/feature error: {exc}")
    if not pair_frames:
        raise RuntimeError("No pairs have enough usable candle history.")

    results = []
    for factor in LABEL_FACTORS:
        panel = make_panel(pair_frames, factor)
        print(f"\n{'=' * 76}\nLABEL FACTOR {factor:.2f}")
        counts = panel.label.value_counts().reindex(LABELS, fill_value=0)
        total = int(counts.sum())
        print("All-label distribution:",
              " | ".join(f"{name}={int(counts[label])} ({100*counts[label]/total:.1f}%)"
                         for label, name in zip(LABELS, LABEL_NAMES)))
        for fold_no, (train_end, test_start, _, test_end) in enumerate(FOLDS, 1):
            masks = fold_masks(panel, train_end, test_start, test_end)
            if masks is None:
                print(f"Fold {fold_no}: no timestamp window")
                continue
            train_mask, test_mask, bounds = masks
            train = panel.loc[train_mask].copy()
            test = panel.loc[test_mask].copy()
            print(f"\nFold {fold_no}: train_rows={len(train)} test_rows={len(test)} "
                  f"train_end={bounds[0]:.0f} test_start={bounds[1]:.0f} "
                  f"test_end={bounds[2]:.0f}")
            if len(train) < MIN_TRAIN_ROWS or len(test) < MIN_TEST_ROWS:
                print("  Skipped: insufficient rows")
                continue
            for strength in WEIGHT_STRENGTHS:
                try:
                    result = score_fold(train, test, strength)
                    if result is None:
                        print(f"  alpha={strength}: skipped; missing classes")
                        continue
                    print_report(factor, strength, fold_no, result)
                    results.append({
                        "factor": factor, "alpha": strength,
                        "fold": fold_no, **result
                    })
                except Exception as exc:
                    print(f"  factor={factor} alpha={strength} failed: {exc}")

    if not results:
        raise RuntimeError("No valid model/fold results. Check data and class coverage.")

    summary = pd.DataFrame(results)
    print("\n" + "=" * 76 + "\nAGGREGATE ACROSS CHRONOLOGICAL TEST WINDOWS")
    grouped = summary.groupby(["factor", "alpha"]).agg(
        folds=("fold", "nunique"),
        mean_balanced_accuracy=("balanced_accuracy", "mean"),
        mean_accuracy=("accuracy", "mean"),
        total_test_rows=("n_test", "sum"),
        total_trades=("trades", "sum"),
        mean_trade_rate_pct=("trade_rate_pct", "mean"),
        mean_net_return_pct=("net_mean_pct", "mean"),
        mean_net_win_rate_pct=("net_win_rate_pct", "mean"),
    ).reset_index().sort_values(
        ["mean_net_return_pct", "mean_balanced_accuracy"], ascending=False
    )
    print(grouped.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nInterpretation guardrails:")
    print("- No candidate is approved for production by this script.")
    print("- Compare consistency across all 3 folds; one strong fold is insufficient.")
    print("- Net return is a simplified close-to-close proxy, not realistic fills.")
    print("- It omits actual TP/SL execution, funding, market impact, latency and overlap.")
    print("- Cost assumption must be replaced with verified account/market costs.")
    print("- Repeat on later unseen data before considering any deployment.")
    print("\nCompleted. Production V3/config/SQLite and live positions were not touched.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\nEXPERIMENT ERROR:")
        traceback.print_exc()
        raise

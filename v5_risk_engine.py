"""Smart ML V5 risk gate.

Pure, fail-closed pre-trade controls. This module does not call an exchange,
load credentials, place orders, or modify the existing V3 bot.

The caller must supply verified exchange account state and contract metadata.
Unknown/unverified state always blocks a new entry.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import floor, isfinite
from typing import Optional


@dataclass(frozen=True)
class RiskConfig:
    max_margin_inr: float = 2000.0
    risk_per_trade_inr: float = 20.0
    max_daily_loss_inr: float = 60.0
    max_daily_trades: int = 2
    max_open_positions: int = 1
    cooldown_seconds: int = 3600
    max_leverage: float = 1.0
    min_confidence: float = 0.75
    min_reward_risk: float = 1.5
    max_market_data_age_seconds: int = 90


@dataclass(frozen=True)
class AccountSnapshot:
    free_balance_inr: float
    realized_pnl_today_inr: Optional[float]
    daily_trade_count: int
    open_position_count: int
    last_trade_epoch: Optional[int]
    positions_verified: bool
    realized_pnl_verified: bool
    kill_switch_enabled: bool = False


@dataclass(frozen=True)
class ContractSpec:
    # INR value of one price-unit move for one contract.
    inr_per_price_unit_per_contract: float
    quantity_step: float
    minimum_quantity: float


@dataclass(frozen=True)
class RiskDecision:
    allowed: bool
    reason: str
    quantity: float = 0.0
    margin_required_inr: float = 0.0
    risk_at_stop_inr: float = 0.0


def _finite_positive(value: float) -> bool:
    return isfinite(value) and value > 0


def evaluate_entry(
    *,
    side: str,
    entry_price: float,
    stop_price: float,
    target_price: float,
    confidence: float,
    market_data_age_seconds: int,
    account: AccountSnapshot,
    contract: ContractSpec,
    config: RiskConfig = RiskConfig(),
    now_epoch: Optional[int] = None,
) -> RiskDecision:
    """Return a sized, fail-closed entry decision; never submits an order."""
    import time

    def block(reason: str) -> RiskDecision:
        return RiskDecision(False, reason)

    side = side.upper().strip()
    if side not in {"BUY", "SELL"}:
        return block("Only BUY or SELL directional signals are eligible.")
    if account.kill_switch_enabled:
        return block("Kill switch is enabled.")
    if not account.positions_verified:
        return block("Exchange positions are not verified; fail-closed.")
    if not account.realized_pnl_verified or account.realized_pnl_today_inr is None:
        return block("Exchange realized daily P&L is not verified; fail-closed.")
    if not isfinite(account.realized_pnl_today_inr):
        return block("Realized daily P&L is invalid.")
    if account.realized_pnl_today_inr <= -abs(config.max_daily_loss_inr):
        return block("Daily realized-loss limit reached.")
    if account.daily_trade_count >= config.max_daily_trades:
        return block("Daily trade limit reached.")
    if account.open_position_count >= config.max_open_positions:
        return block("Maximum concurrent position limit reached.")
    if account.free_balance_inr <= 0:
        return block("No verified free INR balance.")
    if account.free_balance_inr < min(config.max_margin_inr, config.risk_per_trade_inr):
        return block("Free balance is below the minimum risk budget.")
    if confidence < config.min_confidence or confidence > 1 or not isfinite(confidence):
        return block("ML confidence is below threshold or invalid.")
    if market_data_age_seconds < 0 or market_data_age_seconds > config.max_market_data_age_seconds:
        return block("Market data is stale or timestamp is invalid.")
    if account.last_trade_epoch is not None:
        now = int(time.time()) if now_epoch is None else int(now_epoch)
        if now - account.last_trade_epoch < config.cooldown_seconds:
            return block("Trade cooldown is active.")

    if not all(_finite_positive(x) for x in (
        entry_price, contract.inr_per_price_unit_per_contract,
        contract.quantity_step, contract.minimum_quantity,
    )):
        return block("Entry price or contract specification is invalid.")
    if not all(isfinite(x) for x in (stop_price, target_price)):
        return block("Stop or target price is invalid.")

    if side == "BUY":
        if not (stop_price < entry_price < target_price):
            return block("BUY requires stop < entry < target.")
    else:
        if not (target_price < entry_price < stop_price):
            return block("SELL requires target < entry < stop.")

    stop_distance = abs(entry_price - stop_price)
    reward_distance = abs(target_price - entry_price)
    if stop_distance <= 0 or reward_distance / stop_distance < config.min_reward_risk:
        return block("Reward-to-risk ratio is below the configured minimum.")

    # V5 starts at 1x maximum leverage. Any increase needs a separate review.
    if not _finite_positive(config.max_leverage) or config.max_leverage > 1.0:
        return block("Leverage above the initial 1x V5 cap is not enabled.")
    if config.max_margin_inr <= 0 or config.risk_per_trade_inr <= 0:
        return block("Risk configuration is invalid.")

    risk_budget = min(config.risk_per_trade_inr, config.max_margin_inr * 0.01)
    raw_qty_by_risk = risk_budget / (
        stop_distance * contract.inr_per_price_unit_per_contract
    )
    max_margin = min(config.max_margin_inr, account.free_balance_inr)
    raw_qty_by_margin = (
        max_margin * config.max_leverage
        / (entry_price * contract.inr_per_price_unit_per_contract)
    )
    raw_qty = min(raw_qty_by_risk, raw_qty_by_margin)
    qty = floor(raw_qty / contract.quantity_step) * contract.quantity_step
    if qty < contract.minimum_quantity:
        return block("Safe position size is below exchange minimum quantity.")

    margin_required = (
        qty * entry_price * contract.inr_per_price_unit_per_contract
        / config.max_leverage
    )
    risk_at_stop = (
        qty * stop_distance * contract.inr_per_price_unit_per_contract
    )
    if margin_required > max_margin + 1e-8:
        return block("Calculated margin exceeds the configured cap.")
    if risk_at_stop > risk_budget + 1e-8:
        return block("Calculated stop risk exceeds the risk budget.")

    return RiskDecision(
        True,
        "Risk checks passed. Caller must still verify order and TP/SL acknowledgements.",
        quantity=round(qty, 12),
        margin_required_inr=round(margin_required, 2),
        risk_at_stop_inr=round(risk_at_stop, 2),
    )

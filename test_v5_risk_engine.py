import unittest

from v5_risk_engine import (
    AccountSnapshot, ContractSpec, RiskConfig, evaluate_entry,
)


class RiskEngineTests(unittest.TestCase):
    def setUp(self):
        self.account = AccountSnapshot(
            free_balance_inr=2000,
            realized_pnl_today_inr=0,
            daily_trade_count=0,
            open_position_count=0,
            last_trade_epoch=None,
            positions_verified=True,
            realized_pnl_verified=True,
        )
        self.contract = ContractSpec(
            inr_per_price_unit_per_contract=1,
            quantity_step=0.01,
            minimum_quantity=0.01,
        )
        self.kw = dict(
            side="BUY", entry_price=100, stop_price=99,
            target_price=102, confidence=0.80,
            market_data_age_seconds=5, account=self.account,
            contract=self.contract, now_epoch=10000,
        )

    def test_valid_signal_is_sized_within_risk_budget(self):
        result = evaluate_entry(**self.kw)
        self.assertTrue(result.allowed, result.reason)
        self.assertLessEqual(result.risk_at_stop_inr, 20.0)
        self.assertLessEqual(result.margin_required_inr, 2000.0)

    def test_unverified_realized_pnl_blocks(self):
        account = AccountSnapshot(**{
            **self.account.__dict__, "realized_pnl_verified": False,
            "realized_pnl_today_inr": None,
        })
        result = evaluate_entry(**{**self.kw, "account": account})
        self.assertFalse(result.allowed)
        self.assertIn("P&L", result.reason)

    def test_unverified_positions_block(self):
        account = AccountSnapshot(**{**self.account.__dict__, "positions_verified": False})
        self.assertFalse(evaluate_entry(**{**self.kw, "account": account}).allowed)

    def test_daily_loss_limit_blocks(self):
        account = AccountSnapshot(**{**self.account.__dict__, "realized_pnl_today_inr": -60})
        self.assertFalse(evaluate_entry(**{**self.kw, "account": account}).allowed)

    def test_stale_data_blocks(self):
        self.assertFalse(evaluate_entry(**{**self.kw, "market_data_age_seconds": 91}).allowed)

    def test_reward_risk_blocks(self):
        self.assertFalse(evaluate_entry(**{**self.kw, "target_price": 100.5}).allowed)

    def test_leverage_above_one_is_blocked(self):
        cfg = RiskConfig(max_leverage=2)
        self.assertFalse(evaluate_entry(**{**self.kw, "config": cfg}).allowed)

    def test_kill_switch_blocks(self):
        account = AccountSnapshot(**{**self.account.__dict__, "kill_switch_enabled": True})
        self.assertFalse(evaluate_entry(**{**self.kw, "account": account}).allowed)


if __name__ == "__main__":
    unittest.main()

import unittest

from v5_risk_engine import AccountSnapshot, ContractSpec, RiskConfig, evaluate_entry


class RiskEngineTests(unittest.TestCase):
    def setUp(self):
        self.account = AccountSnapshot(
            free_balance_inr=5000,
            daily_trade_count=0,
            open_position_count=0,
            last_trade_epoch=None,
            positions_verified=True,
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

    def test_valid_signal_is_sized_within_margin_cap(self):
        result = evaluate_entry(**self.kw)
        self.assertTrue(result.allowed, result.reason)
        self.assertLessEqual(result.margin_required_inr, 2000.0)
        self.assertGreater(result.risk_at_stop_inr, 0)

    def test_unverified_positions_block(self):
        account = AccountSnapshot(**{**self.account.__dict__, "positions_verified": False})
        self.assertFalse(evaluate_entry(**{**self.kw, "account": account}).allowed)

    def test_stale_data_blocks(self):
        self.assertFalse(evaluate_entry(**{**self.kw, "market_data_age_seconds": 91}).allowed)

    def test_reward_risk_blocks(self):
        self.assertFalse(evaluate_entry(**{**self.kw, "target_price": 100.5}).allowed)

    def test_15x_leverage_is_allowed(self):
        cfg = RiskConfig(max_leverage=15)
        self.assertTrue(evaluate_entry(**{**self.kw, "config": cfg}).allowed)

    def test_leverage_above_15x_blocks(self):
        cfg = RiskConfig(max_leverage=16)
        self.assertFalse(evaluate_entry(**{**self.kw, "config": cfg}).allowed)

    def test_kill_switch_blocks(self):
        account = AccountSnapshot(**{**self.account.__dict__, "kill_switch_enabled": True})
        self.assertFalse(evaluate_entry(**{**self.kw, "account": account}).allowed)

    def test_daily_trade_count_limit_blocks(self):
        account = AccountSnapshot(**{**self.account.__dict__, "daily_trade_count": 2})
        self.assertFalse(evaluate_entry(**{**self.kw, "account": account}).allowed)


if __name__ == "__main__":
    unittest.main()

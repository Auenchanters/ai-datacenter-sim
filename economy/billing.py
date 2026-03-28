from typing import List, Dict, Any
from engine.hardware import ServerRack, CoolingUnit


class BillingEngine:
    """
    Manages all financial calculations for the simulation.

    Responsibilities:
        - Calculates electricity cost per tick based on total power draw.
        - Applies dynamic electricity pricing events (e.g. peak-hour surges).
        - Tracks cumulative financial statistics for leaderboard scoring.
        - Validates that the AI has sufficient balance before hardware purchases.
    """

    BASE_COST_PER_KWH: float = 0.12
    PEAK_MULTIPLIER: float = 2.0
    OFF_PEAK_MULTIPLIER: float = 0.6

    def __init__(
        self,
        starting_budget: float = 500_000.0,
        cost_per_kwh: float = 0.12,
        tick_duration_hours: float = 1.0,
    ):
        self.bank_balance: float = starting_budget
        self.cost_per_kwh: float = cost_per_kwh
        self.tick_duration_hours: float = tick_duration_hours

        self.total_revenue: float = 0.0
        self.total_electricity_cost: float = 0.0
        self.total_hardware_spent: float = 0.0
        self.transaction_log: List[Dict[str, Any]] = []

        self._price_event_active: bool = False
        self._price_event_multiplier: float = 1.0
        self._price_event_ticks_remaining: int = 0

    @property
    def effective_cost_per_kwh(self) -> float:
        return round(self.cost_per_kwh * self._price_event_multiplier, 6)

    @property
    def net_profit(self) -> float:
        return round(
            self.total_revenue - self.total_electricity_cost - self.total_hardware_spent, 4
        )

    def can_afford(self, amount: float) -> bool:
        return self.bank_balance >= amount

    def deduct_hardware_purchase(self, amount: float, hardware_id: str, tick: int) -> bool:
        """
        Deducts the cost of a hardware purchase from the bank balance.
        Returns True if successful, False if insufficient funds.
        """
        if not self.can_afford(amount):
            return False
        self.bank_balance -= amount
        self.bank_balance = round(self.bank_balance, 4)
        self.total_hardware_spent += amount
        self.transaction_log.append({
            "tick": tick,
            "type": "HARDWARE_PURCHASE",
            "item": hardware_id,
            "amount": -amount,
            "balance_after": self.bank_balance,
        })
        return True

    def credit_revenue(self, amount: float, job_id: str, tick: int):
        """
        Credits contract revenue to the bank balance.
        """
        self.bank_balance += amount
        self.bank_balance = round(self.bank_balance, 4)
        self.total_revenue += amount
        self.transaction_log.append({
            "tick": tick,
            "type": "CONTRACT_REVENUE",
            "item": job_id,
            "amount": amount,
            "balance_after": self.bank_balance,
        })

    def charge_electricity(
        self,
        servers: List[ServerRack],
        coolers: List[CoolingUnit],
        tick: int,
    ) -> Dict[str, float]:
        """
        Calculates and deducts the electricity bill for the current tick.
        Returns a breakdown of IT power, cooling power, total cost, and effective rate.
        """
        it_power_kw = sum(s.current_power_kw for s in servers)
        cooling_power_kw = sum(c.current_power_kw for c in coolers)
        total_power_kw = it_power_kw + cooling_power_kw

        cost = total_power_kw * self.tick_duration_hours * self.effective_cost_per_kwh
        cost = round(cost, 4)

        self.bank_balance -= cost
        self.bank_balance = round(self.bank_balance, 4)
        self.total_electricity_cost += cost

        self.transaction_log.append({
            "tick": tick,
            "type": "ELECTRICITY_BILL",
            "item": "power_consumption",
            "amount": -cost,
            "balance_after": self.bank_balance,
        })

        pue = round(total_power_kw / it_power_kw, 4) if it_power_kw > 0 else 0.0

        return {
            "it_power_kw": round(it_power_kw, 3),
            "cooling_power_kw": round(cooling_power_kw, 3),
            "total_power_kw": round(total_power_kw, 3),
            "pue": pue,
            "effective_rate_per_kwh": self.effective_cost_per_kwh,
            "electricity_cost": cost,
        }

    def trigger_price_event(self, multiplier: float, duration_ticks: int):
        """
        Activates a dynamic electricity pricing event.
        Used by the benchmark event system to simulate peak-hour surges
        or off-peak discount windows that test AI adaptability.
        multiplier > 1.0 = price spike. multiplier < 1.0 = discount window.
        """
        self._price_event_active = True
        self._price_event_multiplier = multiplier
        self._price_event_ticks_remaining = duration_ticks

    def tick_price_event(self):
        """
        Called every tick to decrement and expire active pricing events.
        """
        if self._price_event_active:
            self._price_event_ticks_remaining -= 1
            if self._price_event_ticks_remaining <= 0:
                self._price_event_active = False
                self._price_event_multiplier = 1.0

    def get_financial_summary(self) -> Dict[str, Any]:
        """
        Returns a complete financial summary for the leaderboard and state payload.
        """
        return {
            "bank_balance": round(self.bank_balance, 2),
            "total_revenue": round(self.total_revenue, 2),
            "total_electricity_cost": round(self.total_electricity_cost, 2),
            "total_hardware_spent": round(self.total_hardware_spent, 2),
            "net_profit": self.net_profit,
            "effective_cost_per_kwh": self.effective_cost_per_kwh,
            "price_event_active": self._price_event_active,
            "price_event_multiplier": self._price_event_multiplier,
            "price_event_ticks_remaining": self._price_event_ticks_remaining,
        }

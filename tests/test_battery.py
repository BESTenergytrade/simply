import numpy as np
import pandas as pd
import pytest

from simply.actor import Actor
from simply.battery import Battery
from simply.config import Config
from simply.scenario import Scenario


class TestBattery:
    Config("")
    scenario = Scenario(None, None, np.ones(20))
    environment = scenario.environment

    def test_battery_creation(self):
        battery = Battery(capacity=1)
        # check if battery has all the standard attributes. These were defined at some point. If
        # they are changed, default battery creation has to be adjusted everywhere
        assert battery.capacity == 1
        assert battery.soc == 0.5
        assert battery.max_c_rate == 1

    def test_charging(self):
        battery = Battery(capacity=1, max_c_rate=1, soc_initial=0.5)
        assert battery.energy() == pytest.approx(0.5)
        battery.charge(0.5)
        assert battery.energy() == pytest.approx(1)

        with pytest.raises(AssertionError):
            battery.charge(0.5)

        battery.charge(-1)
        with pytest.raises(AssertionError):
            battery.charge(-0.001)

    def test_double_charge_error(self):
        pv = [0] * 24
        rand_gen = np.random.default_rng(seed=42)
        load = rand_gen.random(24)
        schedule = load.copy() * (-1)
        price = rand_gen.random(24) * 0.3

        df = pd.DataFrame(data=zip(load, pv, schedule, price),
                          columns=["load", "pv", "schedule", "price"])
        a = Actor(actor_id="1", df=df, environment=self.environment, battery=Battery(capacity=10))

        # Get energy for the first time step. The actor charges the battery with the amount in the
        # schedule. If the schedule is negative battery gets discharged.
        a.update_battery()
        # Increase time step and get energy again --> No error
        self.environment.time_step += 1
        a.update_battery()
        # If the time step is not increased an error should be thrown
        with pytest.raises(AssertionError):
            a.update_battery()

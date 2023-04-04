import pytest
from simply.battery import Battery

class TestBattery:

    def test_battery_creation(self):
        battery = Battery()
        # check if battery has all the standard attributes. These were defined at some point. If
        # they are changed, default battery creation has to be adjusted everywhere
        assert battery.capacity == 13.5
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


class Battery:
    """Class which implements all raw functionality around the battery"""

    def __init__(self, max_c_rate=1, soc_initial=0.5, capacity=13.5):
        # float between 0 and 1
        self.soc = soc_initial
        # battery capacity
        self.capacity = capacity
        # maximum charge rate per time step
        self.max_c_rate = max_c_rate

    def energy(self):
        """Returns the value of energy inside the battery"""
        return self.soc * self.capacity

    def charge(self, energy):
        """Charge the battery with an amount of energy.

        A negative energy value can be used for discharging. The SOC of the battery can not surpass
        0 or 1.
        :param energy: Amount of energy
        :type energy: float
        """
        soc_after_charge = self.soc + energy/self.capacity
        assert 1 >= soc_after_charge >= 0
        self.soc = soc_after_charge

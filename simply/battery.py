import simply.config as cfg


class Battery:
    """Class which implements all raw functionality around the battery """

    def __init__(self, capacity, max_c_rate=1, soc_initial=0.5, check_boundaries=True):
        # float between 0 and 1
        self.soc = soc_initial
        # battery capacity
        self.capacity = capacity
        # maximum charge rate per time step
        self.max_c_rate = max_c_rate
        self.check_boundaries = check_boundaries
        self.soc_initial = soc_initial

    def reset(self):
        self.soc = self.soc_initial

    def energy(self):
        """Returns the value of energy inside the battery"""
        return self.soc * self.capacity

    def charge(self, energy, constrain=False):
        """Charge the battery with an amount of energy.

        A negative energy value can be used for discharging. The SOC of the battery can not surpass
        0 or 1.

        Returns the difference volume of energy that cannot be charged, the energy that is
        possible to be charged and the SOC after the charging process.

        :param energy: Amount of energy
        :type energy: float
        :param constrain: Constrain charging energy to SOC boundaries
        :type constrain: bool
        """
        soc_after_charge = self.soc + energy/self.capacity
        diff = 0
        energy_chargable = energy
        if constrain:
            if soc_after_charge < 0 - cfg.config.EPS:
                # set charged energy to negative (i.e. discharging) stored energy volume
                energy_chargable = -self.soc * self.capacity
                diff = energy - energy_chargable
                if cfg.config.debug:
                    print(f"Cannot be discharged: {diff}")
                soc_after_charge = 0
            elif soc_after_charge > 1 + cfg.config.EPS:
                # set charged energy to energy volume that fills battery
                energy_chargable = (1 - self.soc) * self.capacity
                diff = energy - energy_chargable
                if cfg.config.debug:
                    print(f"Cannot be charged: {diff}")
                soc_after_charge = 1
        elif self.check_boundaries:
            assert 1+cfg.config.EPS >= soc_after_charge >= 0-cfg.config.EPS, \
                f"Battery is out of soc bounds with soc of: {soc_after_charge}."
        self.soc = soc_after_charge

        return diff, energy_chargable, self.soc

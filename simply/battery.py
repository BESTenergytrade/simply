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

        if cfg.config.debug:
            print(["Battery:", diff, energy_chargable, self.soc])
        return diff, energy_chargable, self.soc


class VariableBattery(Battery):
    """
    Class which implements all raw functionality around the battery
    with variable capacity.
    """

    def __init__(self, capacity, max_c_rate=1, soc_initial=0.5, check_boundaries=True,
                 available=False, min_soc=0.0):
        #
        self.available = available
        self.min_soc = min_soc
        super().__init__(capacity, max_c_rate=max_c_rate, soc_initial=soc_initial,
                         check_boundaries=check_boundaries)

    def energy(self):
        """Returns the value of energy inside the battery if available, otherwise 0 (non-usable)"""
        return self.soc * self.capacity * self.available

    def set_available(self, available=True):
        """Marks battery as available if True, so capacity can be used.
         If False as unavailable, so usable capacity effectively reduces to 0."""
        self.available = available

    def consume(self, demand, constrain=False):
        """Battery cannot be charged when not available but consumption (as discharging) can be
         applied."""
        super().charge(-demand, constrain=constrain)

    def charge(self, energy, constrain=False):
        """Charge the variable battery with an amount of energy if available.

        A negative energy value can be used for discharging. The SOC of the battery can not surpass
        0 or 1. Battery cannot be charged when not available.

        Returns the difference volume of energy that cannot be charged, the energy that is
        possible to be charged and the SOC after the charging process.

        :param energy: Amount of energy
        :type energy: float
        :param constrain: Constrain charging energy to SOC boundaries
        :type constrain: bool
        """
        if self.available:
            return super().charge(energy, constrain=constrain)
        else:
            return energy, 0, self.soc


class EnergyStorageSystem(Battery):
    def __init__(self, capacity, max_c_rate=1, soc_initial=0.5, check_boundaries=True,
                 available=False):
        #
        self.battery = None
        self.var_battery = None
        super().__init__(capacity, max_c_rate=max_c_rate, soc_initial=soc_initial,
                         check_boundaries=check_boundaries)

    def energy(self):
        """Returns the value of energy inside the battery if available, otherwise 0 (non-usable)"""
        return (
            self.battery.energy()
            + 0 if self.var_battery is None else self.var_battery.energy()
        )

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
        if self.available:
            return super().charge(energy, constrain=constrain)
        else:
            return energy, 0, self.soc

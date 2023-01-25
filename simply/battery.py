class Battery:
    def __init__(self, max_c_rate, soc_initial=0.5, capacity=13.5, min_charge_level=0.5):
        # float between 0 and 1
        self.soc = soc_initial
        # battery capacity
        self.capacity = capacity
        # minimum charge level
        self.min_charge_level = min_charge_level
        # maximum charge rate per timestep
        self.max_c_rate = max_c_rate

    def energy(self):
        return self.soc * self.capacity

    def get_energy(self, energy):
        self.soc += energy/self.capacity

    # def update_charge_level(self, energy):
    #
    #
    # def use(self, data):
    #     available_energy = (self.soc - self.min_charge_level) * self.capacity
    #     data['schedule'] += available_energy
    #     self.update_charge_level(available_energy)
    #
    # def charge(self, data):
    #     energy_deficit = (self.min_charge_level - self.soc) * self.capacity
    #
    #     available_surplus = data['schedule']
    #     if available_surplus > energy_deficit:
    #         data['schedule'] -= available_surplus
    #
    #
    # def basic_strategy(self, data):
    #     # Check if batter
    #     if self.soc >= self.min_charge_level:
    #         self.use(data)
    #     else:
    #         self.charge()


class Battery:
    def __init__(self, charge_level=0, capacity=13.5, min_charge_level=0.5):
        # float between 0 and 1
        self.charge_level = charge_level
        # battery capacity
        self.capacity = capacity
        # minimum charge level
        self.min_charge_level = min_charge_level

    def update_charge_level(self, energy):


    def use(self, data):
        available_energy = (self.charge_level - self.min_charge_level) * self.capacity
        data['schedule'] += available_energy
        self.update_charge_level(available_energy)

    def charge(self, data):
        energy_deficit = (self.min_charge_level - self.charge_level) * self.capacity

        available_surplus = data['schedule']
        if available_surplus > energy_deficit:
            data['schedule'] -= available_surplus


    def basic_strategy(self, data):
        # Check if batter
        if self.charge_level >= self.min_charge_level:
            self.use(data)
        else:
            self.charge()

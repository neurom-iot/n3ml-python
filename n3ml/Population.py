class Population:
    def __init__(self, num_neurons):
        self.num_neurons = num_neurons


class LIFPopulation(Population):
    def __init__(self, num_neurons):
        super().__init__(num_neurons)


class SRMPopulation(Population):
    def __init__(self, num_neurons):
        super().__init__(num_neurons)


class Processing(Population):
    def __init__(self, num_neurons):
        super().__init__(num_neurons)

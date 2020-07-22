__add__ = ['IF', 'LIF']


class Neuron:
    def __init__(self):
        raise NotImplementedError


class IF(Neuron):
    def __init__(self,
                 resting_potentials: float = 0.0,
                 threshold: float = 1.0) -> None:
        self.resting_potentials = resting_potentials
        self.threshold = threshold


class LIF(Neuron):
    def __init__(self,
                 resting_potentials: float = 0.0,
                 threshold: float = 1.0,
                 time_constant: float = 1.0) -> None:
        self.resting_potentials = resting_potentials
        self.threshold = threshold
        self.time_constant = time_constant

import numpy as np

from n3ml.neurons import IF, LIF, build_IF
from n3ml.operators import Init, Add, Div, Mul, SimNeurons


class Population:
    def __init__(self, num_neurons=1, neuron_type=LIF, rest_potentials=0,
                 threshold=1, time_constant=1, time_step=0.001):
        self.num_neurons = num_neurons
        self.neuron_type = neuron_type
        self.rest_potentials = rest_potentials
        self.time_constant = time_constant
        self.time_step = time_step
        self.threshold = threshold

    def build(self, model) -> None:
        # Define the signals of a population
        model.signal[self] = dict()
        model.signal[self]['input'] = np.zeros(self.num_neurons, dtype=np.float)
        model.signal[self]['output'] = np.zeros(self.num_neurons, dtype=np.float)
        model.signal[self]['voltage'] = np.zeros(self.num_neurons, dtype=np.float)

        # Initialize the signal of the voltage
        model.add_op(Init(model[self]['voltage']))

        # Simulate the spiking neurons of the population
        model.add_op(SimNeurons(self, model.signal[self]['input'],
                                model.signal[self]['output'], model.signal[self]['voltage']))

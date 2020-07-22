import n3ml.frontend.neuron

__add__ = ['Population']


class Population:
    def __init__(self,
                 num_neurons: int,
                 neuron_type: n3ml.frontend.neuron.Neuron = n3ml.frontend.neuron.LIF) -> None:
        self.num_neurons = num_neurons
        self.neuron_type = neuron_type

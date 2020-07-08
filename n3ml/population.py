import numpy as np

from n3ml.neurons import IF, build_IF


class Population:
    def __init__(self, num_neurons, neuron_type=IF):
        self.num_neurons = num_neurons
        self.neuron_type = neuron_type

        pass


def build_population(population, cg):
    neuron = population.neuron_type

    inp_nn = []
    out_nn = []

    for i in range(population.num_neurons):
        nn = neuron()
        build_IF(nn, cg)

        inp_nn.append(cg.tensors[nn]['in'])
        out_nn.append(cg.tensors[nn]['out'])

    cg.tensors[population] = {}

    cg.tensors[population]['in'] = np.asarray(inp_nn)
    cg.tensors[population]['out'] = np.asarray(out_nn)

import numpy as np

from n3ml.neurons import IF, build_IF

class Population:
    def __init__(self, num_neurons, neuron_type=IF):
        self.num_neurons = num_neurons
        self.neuron_type = neuron_type

    def __call__(self):
        pass

def build_population(instance, cg):
    neuron = instance.neuron_type

    inp_nn = []
    out_nn = []

    for i in range(instance.num_neurons):
        nn = neuron()
        build_IF(nn, cg)
        inp_nn.append(cg.tensors[nn]['in'])
        out_nn.append(cg.tensors[nn]['out'])

    cg.tensors[instance] = {}

    cg.tensors[instance]['in'] = np.asarray(inp_nn, dtype=np.float)
    cg.tensors[instance]['out'] = np.asarray(out_nn, dtype=np.float)
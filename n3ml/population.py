import numpy as np

from n3ml.neurons import IF, LIF, build_IF
from n3ml.operators import Init, Add, Div, Mul


class Population:
    def __init__(self, num_neurons=1, neuron_type=LIF, resting=0, tc=0):
        self.num_neurons = num_neurons
        self.neuron_type = neuron_type
        self.resting = resting
        self.tc = tc

        pass

    def build(self, model) -> None:
        model.set_var(self, 'input', np.zeros(self.num_neurons, dtype=np.float))
        model.set_var(self, 'output', np.zeros(self.num_neurons, dtype=np.float))
        model.set_var(self, 'voltage', np.zeros(self.num_neurons, dtype=np.float))

        model.add_op(Init(model.get_var(self, 'voltage'), self.resting))

        model.set_var(self, 'intermediary', np.zeros(self.num_neurons, dtype=np.float))
        model.add_op(Add(model.get_var(self, 'intermediary'),
                         model.get_var(self, 'voltage'), model.get_var(self, 'input')))

        model.set_var(self, 'tc', np.array([self.tc], dtype=np.float))
        model.set_var(self, 'constant', np.zeros(1, dtype=np.float))
        model.add_op(Div(model.get_var(self, 'constant'),
                         model.get_var('dt'), model.get_var(self, 'tc')))

        model.set_var(self, 'intermediary2', np.zeros(self.num_neurons, dtype=np.float))
        model.add_op(Mul(model.get_var(self, 'intermediary2'),
                         model.get_var(self, 'intermediary'), model.get_var(self, 'constant')))

        model.set_var(self, 'intermediary3', np.zeros(self.num_neurons, dtype=np.float))
        model.add_op(Add(model.get_var(self, 'voltage'),
                         model.get_var(self, 'voltage'), model.get_var(self, 'intermediary3')))

        pass

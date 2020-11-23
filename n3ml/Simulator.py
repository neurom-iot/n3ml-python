from n3ml.Model import Model
from n3ml.Builder import Builder


def transpose(x, index):
    y = [None for _ in range(len(x))]
    for i, v in enumerate(index):
        y[i] = x[v]
    return y


class Simulator:
    def __init__(self,
                 network,
                 model=None,
                 time_step=0.001):
        self.network = network
        self.time_step = time_step

        if model is None:
            self.model = Model()
            Builder.build(self.model, network)

    def run(self, simulation_time=2):
        import time
        import numpy as np
        import matplotlib.pyplot as plt

        num_steps = int(simulation_time / self.time_step)

        # ops for spikeprop
        #_ops = self.model.operator[:13]
        #ops = transpose(_ops, [4, 5, 7, 2, 3, 9, 10, 6, 11, 12, 8, 0, 1])

        # ops for stdp
        _ops = self.model.operator[:5]
        ops = transpose(_ops, [2, 3, 4, 0, 1])

        for step in range(num_steps):
            self._run_step(ops)
            #for op in self.model.operator:
                #start_time = time.time()
                #op()
                #print("Operator: {} - {}s seconds---".format(op, time.time() - start_time))
            #print("time: {} ms - period: {} ms".format(
                #self.model.signal['current_time'], self.model.signal['current_period']))
            #plt.imshow(self.model.signal[self.network.source[0]]['image'])
            #plt.show()

    def _run_step(self, ops):
        for op in ops:
            op()

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

        # ops for spikeprop in mnists
        _ops = self.model.operator[:24]
        ops = transpose(_ops, [3, 5, 7, 9, 12, 2, 4, 10, 11, 6, 13, 14, 8, 15, 16, 17, 18, 19, 20, 21, 22, 23, 0, 1])

        # ops for spikeprop in iris
        # _ops = self.model.operator[:26]
        # ops = transpose(_ops, [5, 7, 9, 11, 14, 2, 3, 17, 18, 4, 6, 12, 13, 8, 15, 16, 10, 19, 20, 21, 22, 23, 24, 25, 0, 1])

        logger = {
            'image': [],
            'pred': []
        }

        print("time: 0 - period: 0")

        for step in range(num_steps):
            self._run_step(ops)

            print(self.model.signal[self.network.source[0]]['data_index'])

            # print(self.model.signal[self.network.connection[0]]['synaptic_weight'])
            print(self.model.signal[self.network.connection[1]]['synaptic_weight'])

            print(self.model.signal[self.network.source[0]]['data'])
            print(self.model.signal[self.network.source[0]]['spike_time'])
            print(self.model.signal[self.network.population[0]]['spike_time'])
            print(self.model.signal[self.network.population[1]]['spike_time'])
            print(self.model.signal[self.network.learning]['target'])
            print("error: {}".format(self.model.signal[self.network.learning]['error']))

            print("time: {} - period: {}".format(self.model.signal['current_time'], self.model.signal['current_period']))

            # if self.model.signal['current_period'] != 0 and self.model.signal['current_period'] % 10 == 0:
            #     # Record an image and spikes after each period
            #     logger['image'].append(
            #         np.array(self.model.signal[self.network.source[0]]['image']))
            #     logger['pred'].append(
            #         np.array(self.model.signal[self.network.population[1]]['spike_time']))

        print("# images: {}".format(len(logger['image'])))

        # Visualize recorded data
        sampling_period = 10
        num_neurons = 10
        num_periods = len(logger['image'])
        spikes = np.zeros(shape=(sampling_period * num_periods, num_neurons))

        # Generate spikes from spike times
        for p in range(num_periods):
            for i, v in enumerate(logger['pred'][p]):
                if -1 < v < 10:
                    spikes[int(v)+p*10, i] = 1

        print(spikes)

        from n3ml.Visualizer import plot_result
        plot_result(logger['image'], spikes)

    def _run_step(self, ops):
        for op in ops:
            op()

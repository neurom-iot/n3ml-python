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
        _ops = self.model.operator[:20]
        ops = transpose(_ops, [3, 5, 7, 9, 12, 2, 4, 10, 11, 6, 13, 14, 8, 15, 16, 17, 18, 19, 0, 1])

        # ops for stdp
        #_ops = self.model.operator[:5]
        #ops = transpose(_ops, [2, 3, 4, 0, 1])

        # self._run_step(ops)
        # self._run_step(ops)
        # self._run_step(ops)
        # self._run_step(ops)
        # self._run_step(ops)
        # self._run_step(ops)
        # self._run_step(ops)
        # self._run_step(ops)
        # self._run_step(ops)
        # self._run_step(ops)
        # self._run_step(ops)
        # self._run_step(ops)
        # self._run_step(ops)

        logger = {
            'image': [],
            'pred': []
        }

        print("time: 0 - period: 0")

        for step in range(num_steps):
            self._run_step(ops)

            print(self.model.signal[self.network.connection[0]]['spike_response'])
            print(self.model.signal[self.network.population[0]]['membrane_potential'])
            print(self.model.signal[self.network.population[0]]['spike_time'])
            print(self.model.signal[self.network.population[1]]['spike_time'])
            print(self.model.signal[self.network.learning]['label'])
            print(self.model.signal[self.network.learning]['target'])
            print(self.model.signal[self.network.learning]['prediction'])
            print(self.model.signal[self.network.learning]['error'])

            print("time: {} - period: {}".format(self.model.signal['current_time'], self.model.signal['current_period']))

            if self.model.signal['current_period'] != 0 and self.model.signal['current_period'] % 10 == 0:
                # Record an image and spikes after each period
                logger['image'].append(
                    np.array(self.model.signal[self.network.source[0]]['image']))
                logger['pred'].append(
                    np.array(self.model.signal[self.network.population[1]]['spike_time']))

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

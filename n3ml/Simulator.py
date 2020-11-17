from n3ml.Model import Model
from n3ml.Builder import Builder


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
        num_steps = int(simulation_time / self.time_step)
        for step in range(num_steps):
            for op in self.model.operator:
                start_time = time.time()
                op()
                print("Operator: {} - {}s seconds---".format(op, time.time() - start_time))


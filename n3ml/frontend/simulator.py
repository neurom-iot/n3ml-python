from n3ml.model import Model


class Simulator:
    def __init__(self, network, model=None, time_step=0.001):
        self.network = network
        self.model = Model()
        self.time_step = time_step

    def run(self, time):
        iters = int(time / self.time_step)

        for i in range(iters):
            self.model.run()

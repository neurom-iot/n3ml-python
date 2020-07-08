from n3ml.network import build_network
from n3ml.utility import sort


class Simulator:
    def __init__(self, network, dt):
        self.network = network
        self.dt = dt

        self.cg = build_network(network)

        sort(self.cg)

    def run(self, t=1):
        total_iters = int(t / self.dt)

        for iter in range(total_iters):
            self.cg.run_step()

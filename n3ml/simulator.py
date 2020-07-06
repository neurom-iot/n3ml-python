class Simulator:
    def __init__(self, network, dt):
        self.network = network
        self.dt = dt

        cg = build_network(network)

        pass

    def run(self, t=1):
        total_iters = int(t / self.dt)

        for iter in range(total_iters):
            print("iter: {} - ".format(iter))
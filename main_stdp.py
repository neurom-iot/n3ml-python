from n3ml.Network import Network
from n3ml.Source import MNISTSource
from n3ml.Population import Processing
from n3ml.Connection import Connection
from n3ml.Simulator import Simulator
from n3ml.Learning import STDP

if __name__ == '__main__':
    net = Network(code='multiple', learning=STDP())

    src = MNISTSource(code='poisson')

    pop = Processing(num_neurons=400)

    conn = Connection(src, pop)

    net.add(src)
    net.add(pop)
    net.add(conn)

    sim = Simulator(network=net)

    sim.run(simulation_time=0.01)

from n3ml.Network import Network
from n3ml.Source import IRISSource
from n3ml.Population import SRMPopulation
from n3ml.Connection import Connection
from n3ml.Simulator import Simulator
from n3ml.Learning import SpikeProp

if __name__ == '__main__':
    net = Network(code='single', learning=SpikeProp())

    src = IRISSource(code='population', num_neurons=12, sampling_period=10)

    pop_1 = SRMPopulation(num_neurons=10)
    pop_2 = SRMPopulation(num_neurons=3)

    conn_1 = Connection(pre=src, post=pop_1)
    conn_2 = Connection(pre=pop_1, post=pop_2)

    net.add(src)
    net.add(conn_1)
    net.add(pop_1)
    net.add(conn_2)
    net.add(pop_2)

    sim = Simulator(network=net)

    sim.run(simulation_time=10)

import n3ml.Network as Network
import n3ml.Source as Source
import n3ml.Population as Population
import n3ml.Connection as Connection
import n3ml.Simulator as Simulator
import n3ml.Learning as Learning

if __name__ == '__main__':
    net = Network.Network(code='single', learning=Learning.SpikeProp())

    src = Source.MNISTSource(code='population', num_neurons=20)

    pop_1 = Population.SRMPopulation(num_neurons=100)
    pop_2 = Population.SRMPopulation(num_neurons=10)

    conn_1 = Connection.Connection(pre=src, post=pop_1)
    conn_2 = Connection.Connection(pre=pop_1, post=pop_2)

    net.add(src)
    net.add(conn_1)
    net.add(pop_1)
    net.add(conn_2)
    net.add(pop_2)

    sim = Simulator.Simulator(network=net)

    sim.run(simulation_time=0.01)

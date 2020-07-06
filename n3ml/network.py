from n3ml.neurons.neurons import Neuron
from n3ml.connection import Connection

class Network:
    def __init__(self):
        self.neurons = []
        self.connections = []

        pass

    def add(self, instance):
        if isinstance(instance, Neuron) or isinstance(instance, Population):
            self.neurons.append(instance)
        elif isinstance(instance, Connection):
            self.connections.append(instance)

        pass

class ComputationalGraph:
    def __init__(self):
        pass

def build_network(instance):
    pass
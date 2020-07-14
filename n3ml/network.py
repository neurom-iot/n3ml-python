from n3ml.neurons import Neuron, build_IF
from n3ml.connection import Connection, build_connection
from n3ml.population import Population, build_population
from n3ml.operators import Operator

__all__ = ['Network']


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
        self.tensors = {}
        self.operators = []

        pass

    def add_op(self, op):
        if isinstance(op, Operator):
            self.operators.append(op)

        pass

    def run_step(self):
        for op in self.operators:
            op.run_step()

        pass


def build_network(network):
    cg = ComputationalGraph()

    for instance in network.neurons:
        if isinstance(instance, Neuron):
            build_IF(instance, cg)
        elif isinstance(instance, Population):
            build_population(instance, cg)

    for conn in network.connections:
        build_connection(conn, cg)

    return cg
from n3ml.frontend.source import Source
from n3ml.frontend.population import Population
from n3ml.frontend.connection import Connection

from n3ml.model import Model

__add__ = ['Network']


class Network:
    def __init__(self):
        self.network = []

        self.source = []
        self.population = []
        self.connection = []

    def add(self, obj):
        if isinstance(obj, Network):
            self.network.append(obj)
        elif isinstance(obj, Source):
            self.source.append(obj)
        elif isinstance(obj, Population):
            self.population.append(obj)
        elif isinstance(obj, Connection):
            self.connection.append(obj)

    @classmethod
    def build(cls, network=None):
        """
        :param network: Network
        :return: Tuple[Dict, List]
        """
        if network is None or not isinstance(network, cls):
            raise TypeError

        model = Model()

        for obj in network.network + network.source + network.population + network.connection:
            if isinstance(obj, Network):
                raise NotImplementedError
            elif isinstance(obj, Source):
                obj.build(obj)
            elif isinstance(obj, Population):
                pass
            elif isinstance(obj, Connection):
                pass
            else:
                raise TypeError

        return model

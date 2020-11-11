__add__ = ['Network']


class Network:
    def __init__(self, code, learning, threshold=1.0):
        self.code = code
        self.learning = learning
        self.threshold = threshold
        self.sub_network = []
        self.source = []
        self.population = []
        self.connection = []
        self.component = []

    def add(self, obj):
        from n3ml.Source import Source
        from n3ml.Connection import Connection
        from n3ml.Population import Population

        self.component.append(obj)

        if isinstance(obj, Network):
            self.sub_network.append(obj)
        elif isinstance(obj, Source):
            self.source.append(obj)
        elif isinstance(obj, Population):
            self.population.append(obj)
        elif isinstance(obj, Connection):
            self.connection.append(obj)

from n3ml.Network import Network
from n3ml.Source import MNISTSource
from n3ml.Population import Processing

if __name__ == '__main__':
    net = Network(code='multiple')

    src = MNISTSource(code='poisson')

    pop = Processing(num_neurons=400)

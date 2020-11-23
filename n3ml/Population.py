class Population:
    def __init__(self, num_neurons):
        self.num_neurons = num_neurons


class LIFPopulation(Population):
    def __init__(self, num_neurons):
        super().__init__(num_neurons)


class SRMPopulation(Population):
    def __init__(self, num_neurons):
        super().__init__(num_neurons)


class Processing(Population):
    def __init__(self, num_neurons):
        super().__init__(num_neurons)


def heaviside(x):
    return (x + abs(x)) / (2 * x)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    x = [_ for _ in range(-5, 0, 1)]
    y = [_ for _ in range(1, 6, 1)]

    z = x + y

    print(z)

    zz = [heaviside(_) for _ in z]

    print(zz)

    plt.plot(z, zz)
    plt.show()

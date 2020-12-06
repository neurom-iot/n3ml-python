class Source:
    def __init__(self, dataset, code):
        self.dataset = dataset
        self.code = code


class IRISSource(Source):
    def __init__(self,
                 code,
                 num_neurons,
                 sampling_period,
                 beta=1.0):
        super().__init__('iris', code)
        self.num_neurons = num_neurons
        self.sampling_period = sampling_period
        self.beta = beta
        self.dataset = None
        self.target = None
        self.min_vals = None
        self.max_vals = None

        self._load_iris()
        self._get_statistics()

        import numpy as np
        self.indexes = np.arange(self.dataset.shape[0])
        np.random.shuffle(self.indexes)

    def _load_iris(self):
        from sklearn.datasets import load_iris
        iris = load_iris()
        self.dataset = iris['data']
        self.target = iris['target']

    def _get_statistics(self):
        import numpy as np
        self.min_vals = np.amin(self.dataset, axis=0)
        self.max_vals = np.amax(self.dataset, axis=0)


class MNISTSource(Source):
    def __init__(self,
                 code,
                 num_neurons=None,
                 sampling_period=10,
                 beta=1.0):
        super().__init__('mnist', code)
        if code == 'poisson':
            self.num_neurons = 28 * 28
        elif code == 'population':
            self.num_neurons = num_neurons
        self.sampling_period = sampling_period
        self.beta = beta
        self.min_value = 0.0
        self.max_value = 1.0
        self.num_images = 60000
        self.rows = 28
        self.cols = 28
        self.images = None
        self.labels = None

        self._load_mnist()

    def _load_mnist(self):
        import gzip
        import numpy as np

        with gzip.open('data/train-images-idx3-ubyte.gz') as bytestream:
            bytestream.read(16)
            raw_data = np.frombuffer(bytestream.read(), dtype='>u1')
        self.images = raw_data.reshape(self.num_images, self.rows, self.cols) / 255.0

        with gzip.open('data/train-labels-idx1-ubyte.gz') as bytestream:
            bytestream.read(8)
            raw_data = np.frombuffer(bytestream.read(), dtype='>u1')
        self.labels = raw_data.reshape(self.num_images)


if __name__ == '__main__':
    iris = IRISSource(code='population', num_neurons=12, sampling_period=10)

    print()

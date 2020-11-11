class Source:
    def __init__(self, dataset, code):
        self.dataset = dataset
        self.code = code


class MNISTSource(Source):
    def __init__(self, code, num_neurons, sampling_period=10, beta=1.0):
        super().__init__('mnist', code)
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

class Signal:
    def __init__(self, data):
        self.data = data


if __name__ == '__main__':
    import numpy as np

    sig = Signal(data=np.asarray([0, 1, 2, 3]))

    print(sig.data)

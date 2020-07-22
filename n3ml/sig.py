import numpy


class Signal(numpy.ndarray):
    def __new__(cls, *args, **kwargs):
        return super(Signal, cls).__new__(cls, *args, **kwargs)

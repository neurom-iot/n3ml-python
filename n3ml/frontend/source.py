from n3ml.sig import Signal
from n3ml.ops import *

__add__ = ['Source']


class Source:
    def __init__(self, output_size=2, firing_rate=10, images=None, input_size=None):
        self.output_size = output_size
        self.firing_rate = firing_rate

    @classmethod
    def build(cls, source=None):
        if source is None or not isinstance(source, cls):
            raise TypeError

        signal = dict()
        operator = list()

        signal[source]['out'] = Signal(source.output_size)

        operator.append(Sample(signal[source]['out'], source.firing_rate))

        return tuple(signal, operator)

import numpy as np

import n3ml.model
import n3ml.operators

from n3ml.signal import Signal

__add__ = ['Source']


class Source:
    def __init__(self, **kwargs):
        if kwargs.__contains__('distribution'):
            self.output_size = kwargs['output_size']
            self.distribution = kwargs['distribution']

    def build(self,
              model: n3ml.model.Model) -> None:
        model.signal[self]['output'] = Signal(self.output_size, np.float)

        model.add_op(n3ml.operators.Sample(self, model.signal[self]['output']))

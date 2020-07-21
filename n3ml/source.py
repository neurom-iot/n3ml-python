import numpy as np

import n3ml.model
import n3ml.operators


class Source:
    def __init__(self, **kwargs):
        if kwargs.__contains__('distribution'):
            self.output_size = kwargs['output_size']
            self.distribution = kwargs['distribution']

            return

        raise NotImplementedError

    def build(self,
              model: n3ml.model.Model) -> None:
        model.signal[self]['output'] = np.zeros(self.output_size, np.float)

        model.add_op(n3ml.operators.Sample(self, model.signal[self]['output']))

        raise NotImplementedError


if __name__ == '__main__':
    src = Source(distribution=np.random.uniform)
    src.build()

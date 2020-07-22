import numpy

import n3ml.op


class Signal(numpy.ndarray):
    pass


class Model:
    def __init__(self):
        self.signal = dict()
        self.operator = list()

    def add_op(self,
               op: n3ml.op.Operator) -> None:
        if isinstance(op, n3ml.op.Operator):
            self.operator.append(op)

    def run(self) -> None:
        for op in self.operator:
            op.make_step()

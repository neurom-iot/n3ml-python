import n3ml.operators


class Model:
    def __init__(self):
        self.signal = dict()
        self.operator = list()

    def add_op(self,
               op: n3ml.operators.Operator) -> None:
        if isinstance(op, n3ml.operators.Operator):
            self.operator.append(op)

    def make_step(self) -> None:
        for op in self.operator:
            op.make_step()

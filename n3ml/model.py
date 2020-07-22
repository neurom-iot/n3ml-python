import n3ml.ops


class Model:
    def __init__(self):
        self.signal = dict()
        self.operator = list()

    def add_op(self,
               op: n3ml.ops.Operator) -> None:
        if isinstance(op, n3ml.ops.Operator):
            self.operator.append(op)

    def run(self):
        for op in self.operator:
            op.make_step()

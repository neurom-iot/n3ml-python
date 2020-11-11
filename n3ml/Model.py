class Model:
    def __init__(self):
        self.signal = {}
        self.operator = []
        self.nan = -1

    def add_op(self, op):
        self.operator.append(op)

    def run(self):
        for op in self.operator:
            op.make_step()

class Model:
    def __init__(self, *args, **kwargs):
        self.signal = {}
        self.operator = []
        self.nan = -1
        #self.num_classes = kwargs['num_classes']

    def add_op(self, op):
        self.operator.append(op)

    def run(self):
        for op in self.operator:
            op.make_step()

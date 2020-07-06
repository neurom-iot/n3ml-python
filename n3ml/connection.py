import numpy as np

class Connection:
    def __init__(self, pre, post, initializer=np.random.uniform):
        self.pre = pre
        self.post = post
        self.initializer = initializer

    def __call__(self, inp, out, w):
        out = np.matmul(inp, w.transpose())

def build_connection(instance, cg):
    inp = cg.tensors[instance.pre]['out']
    out = cg.tensors[instance.post]['in']

    size_inp = inp.shape[0]
    size_out = out.shape[0]

    cg.tensors[instance] = {}

    cg.tensors[instance]['weights'] = np.zeros((size_out, size_inp), dtype=np.float)
    w = cg.tensors[instance]['weights']
    w = instance.initializer((size_out, size_inp))

    def simulate_connection():
        instance(inp, out, w)

    return simulate_connection
import numpy as np

from n3ml.operators import Init, Matmul


class Connection:
    def __init__(self, pre, post, initializer=np.random.uniform):
        self.pre = pre
        self.post = post
        self.initializer = initializer


def build_connection(conn, cg):
    inp = cg.tensors[conn.pre]['out']
    out = cg.tensors[conn.post]['in']

    inp_size = inp.shape[0]
    out_size = out.shape[0]

    cg.tensors[conn]['weights'] = np.zeros((out_size, inp_size), dtype=np.float)

    cg.add_op(Init(cg.tensors[conn]['weights'], conn.initializer))

    cg.add_op(Matmul(cg.tensors[conn]['out'], cg.tensors[conn]['in'], cg.tensors[conn]['weights']))

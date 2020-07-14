import math
import numpy as np
from n3ml.network import Network


class Neuron(object):
    def __init__(self):
        pass

    def build(self):
        NotImplementedError


class IF(Neuron):
    def __init__(self, v0=0, vth=1, dt=0.001):
        self.v0 = v0
        self.vth = vth
        self.dt = dt

    def __call__(self, inp, out, **kwargs):
        v = kwargs['v']
        vth = kwargs['vth']
        dt = kwargs['dt']

        v = v + inp * dt
        spikes = math.floor(v / vth)
        out = spikes / dt
        v = v - spikes


class LIF(Neuron):
    def __init__(self):
        pass

    def build(self, model: Network) -> None:
        pass


def build_IF(instance, cg):
    cg.tensors[instance] = {}

    cg.tensors[instance]['in'] = np.asarray([0], dtype=np.float)
    cg.tensors[instance]['out'] = np.asarray([0], dtype=np.float)
    cg.tensors[instance]['v'] = np.asarray([instance.v0], dtype=np.float)

    inp = cg.tensors[instance]['in']
    out = cg.tensors[instance]['out']
    v = cg.tensors[instance]['v']

    print(inp.shape)

    def simulate_neuron():
        instance(inp, out, v=v, vth=instance.vth, dt=instance.dt)

    return simulate_neuron
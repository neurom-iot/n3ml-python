import math
import numpy as np

class Neuron(object):
    def __init__(self):
        pass

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

def build_IF(instance, cg):
    cg.tensors[instance]['in'] = np.asarray(0, dtype=np.float)
    cg.tensors[instance]['out'] = np.asarray(0, dtype=np.float)
    cg.tensors[instance]['v'] = np.asarray(instance.v0, dtype=np.float)

    inp = cg.tensors[instance]['in']
    out = cg.tensors[instance]['out']
    v = cg.tensors[instance]['v']

    def simulate_neuron():
        instance(inp, out, v=v, vth=instance.vth, dt=instance.dt)

    return simulate_neuron
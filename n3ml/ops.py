from typing import *

import n3ml.sig


class Operator(object):
    def __init__(self):
        raise NotImplementedError

    def make_step(self):
        raise NotImplementedError


class Init(Operator):
    # TODO: 시그널을 초기화하는 방법 구현 계획
    #       1. 특정 값이나 값 리스트가 주어졌을 때 입력된 시그널을 초기화
    #       2. Initializer가 주어졌을 때 입력된 시그널을 초기화
    def __init__(self,
                 signal: n3ml.sig.Signal,
                 value: Union[int, List[int]]) -> None:
        self.signal = signal
        self.value = value

    def make_step(self):
        pass


class Add(Operator):
    def __init__(self):
        pass

    def make_step(self):
        pass


class Mul(Operator):
    def __init__(self):
        pass

    def make_step(self):
        pass


class MatMul(Operator):
    def __init__(self):
        pass

    def make_step(self):
        pass


class SimNeurons(Operator):
    def __init__(self):
        pass

    def make_step(self):
        pass


class Sample(Operator):
    def __init__(self, state, firing_rate):
        self.state = state
        self.firing_rate = firing_rate

    def make_step(self):
        pass

class Operator:
    def __init__(self):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class MatMul(Operator):
    def __init__(self, weight_matrix, inp_vector, out_vector):
        self.weight_matrix = weight_matrix
        self.inp_vector = inp_vector
        self.out_vector = out_vector

    def __call__(self, *args, **kwargs):
        import numpy as np
        self.out_vector = np.matmul(self.weight_matrix, self.inp_vector)


class UpdateTime(Operator):
    def __init__(self, current_time):
        self.current_time = current_time

    def __call__(self, *args, **kwargs):
        self.current_time += 1


class UpdatePeriod(Operator):
    def __init__(self,
                 current_period,
                 sampling_period):
        self.current_period = current_period
        self.sampling_period = sampling_period

    def __call__(self, *args, **kwargs):
        if self.current_period < self.sampling_period:
            self.current_period += 1
        else:
            self.current_period = 0


class SampleImage(Operator):
    def __init__(self,
                 image,
                 data_index,
                 num_images,
                 images):
        pass

    def __call__(self, *args, **kwargs):
        pass


class SpikeTime(Operator):
    def __init__(self,
                 membrane_potential,
                 spike_time,
                 threshold,
                 current_time):
        self.membrane_potential = membrane_potential
        self.spike_time = spike_time
        self.threshold = threshold
        self.current_time = current_time

    def __call__(self, *args, **kwargs):
        self.spike_time[self.membrane_potential > self.threshold] = self.current_time
        self.membrane_potential[self.membrane_potential > self.threshold] = -987654321


class UpdatePeriodAndImage(Operator):
    def __init__(self,
                 sampling_period,
                 current_period,
                 image,
                 data_index,
                 num_images,
                 images):
        self.sampling_period = sampling_period
        self.current_period = current_period
        self.image = image
        self.data_index = data_index
        self.num_images = num_images
        self.images = images

    def __call__(self, *args, **kwargs):
        import numpy as np
        if self.current_period < self.sampling_period:
            self.current_period += 1
        else:
            self.data_index = np.random.randint(0, self.num_images)
            self.image = self.images[self.data_index]
            self.current_period = 0
            import matplotlib.pyplot as plt


class PopulationEncode(Operator):
    def __init__(self,
                 image,
                 spike_time,
                 num_neurons,
                 sampling_period,
                 beta,
                 min_value,
                 max_value):
        self.image = image
        self.rows, self.cols = image.shape
        self.spike_time = spike_time
        self.num_neurons = num_neurons
        self.sampling_period = sampling_period
        self.beta = beta
        self.min_value = min_value
        self.max_value = max_value
        from scipy.stats import norm
        self.receptive_field = [norm.pdf for _ in range(self.num_neurons)]
        import numpy as np
        self.mean = [self.min_value+(2*i-3)/2*(self.max_value-self.min_value)/(self.num_neurons-2) for i in np.arange(1, self.num_neurons+1)]
        self.std = 1/self.beta*(self.max_value-self.min_value)/(self.num_neurons-2)
        self.max_pdf = norm.pdf(0, scale=self.std)

    def __call__(self, *args, **kwargs):
        flatten_image = self.image.flatten()
        for i in range(self.rows * self.cols):
            for j in range(self.num_neurons):
                self.spike_time[i, j] = self._transform_spike_time(
                    self.receptive_field[j](flatten_image[i], self.mean[j], self.std))

    def _transform_spike_time(self, response):
        max_spike_time = self.sampling_period
        max_response = self.max_pdf
        spike_time = response * max_spike_time / max_response
        spike_time = spike_time - max_spike_time
        spike_time = spike_time * -1.0
        spike_time = round(spike_time)
        return spike_time


class SpikeResponse(Operator):
    """Compute spike responses

    Compute spike responses using current time in a period and
    firing times from presynaptic neurons.

    """
    def __init__(self,
                 current_period,
                 spike_time,
                 spike_response,
                 tau=1.0):
        # inputs
        self.current_period = current_period
        self.spike_time = spike_time
        # output
        self.spike_response = spike_response
        # hyperparams
        self.tau = tau

    def __call__(self, *args, **kwargs):
        if len(self.spike_time.shape) > 1:
            spike_time = self.spike_time.flatten()
        else:
            spike_time = self.spike_time

        t = self.current_period - spike_time
        x = t / self.tau
        import numpy as np
        y = np.exp(1 - x)
        y = x * y
        self.spike_response = y


class RMSE(Operator):
    def __init__(self,
                 prediction,
                 target,
                 error,
                 nan=-1):
        # inputs
        self.prediction = prediction
        self.target = target
        # output
        self.error = error
        # hyperparameter
        self.nan = nan

    def __call__(self, *args, **kwargs):
        y_ = self.prediction[self.prediction > self.nan]
        y = self.target[self.prediction > self.nan]
        error = y_ - y
        error = error ** 2
        import numpy as np
        error = np.sum(error)
        error = error / 2.0
        self.error = error


class UpdatePeriodAndLabel(Operator):
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        pass


if __name__ == '__main__':
    import numpy as np
    from n3ml.Signal import Signal

    mp = Signal(data=np.asarray([0.5, 0.8, 1.2, 1.5]))
    st = Signal(data=np.asarray([0, 0, 0, 0]))
    th = Signal(data=np.asarray([1]))
    pt = Signal(data=np.asarray([4]))

    print(st.data)
    print(mp.data)

    op = SpikeTime(mp.data, st.data, th.data, pt.data)
    op()

    print(st.data)
    print(mp.data)

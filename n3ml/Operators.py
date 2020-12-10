import numpy as np
from scipy.stats import norm

class Operator:
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class InitSpikeTime(Operator):
    def __init__(self,
                 spike_time,
                 current_period,
                 value):
        # Signals
        self.spike_time = spike_time
        self.current_period = current_period
        #
        self.value = value

    def __call__(self, *args, **kwargs):
        if self.current_period == 0:
            self.spike_time.fill(self.value)


class InitWeight(Operator):
    def __init__(self,
                 weight,
                 current_time,
                 value=None,
                 random_process=None):
        super().__init__()
        # signals
        self.weight = weight
        self.current_time = current_time
        #
        self.value = value
        self.random_process = random_process

    def __call__(self, *args, **kwargs):
        if self.current_time == 0:
            if self.value is not None:
                self.weight *= self.value
            elif self.random_process is not None:
                self.weight[:] = self.random_process(0, 0.4, self.weight.shape)


class MatMul(Operator):
    def __init__(self, weight_matrix, inp_vector, out_vector):
        self.weight_matrix = weight_matrix
        self.inp_vector = inp_vector
        self.out_vector = out_vector

    def __call__(self, *args, **kwargs):
        import numpy as np
        self.out_vector[:] = np.matmul(self.weight_matrix, self.inp_vector)


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
            self.current_period.fill(0)


class UpdateLabel(Operator):
    def __init__(self,
                 label,
                 index,
                 labels,
                 current_period,
                 sampling_period,
                 indexes=None):
        # Signals
        self.label = label
        self.index = index
        self.current_period = current_period
        #
        self.sampling_period = sampling_period
        self.labels = labels
        self.indexes = indexes

    def __call__(self, *args, **kwargs):
        if self.current_period == self.sampling_period:
            if self.indexes is None:
                self.label[0] = self.labels[self.index]
            else:
                self.label[0] = self.labels[self.indexes[self.index]]


class UpdateTarget(Operator):
    def __init__(self,
                 target,
                 label,
                 current_period,
                 sampling_period):
        # signals
        self.target = target
        self.label = label
        self.current_period = current_period
        #
        self.sampling_period = sampling_period

    def __call__(self, *args, **kwargs):
        if self.current_period == self.sampling_period:
            self.target.fill(self.sampling_period)
            self.target[self.label] = 0.0


class UpdateWeight(Operator):
    def __init__(self,
                 weight,
                 gradient,
                 current_period,
                 sampling_period):
        # signals
        self.weight = weight
        self.gradient = gradient
        self.current_period = current_period
        #
        self.sampling_period = sampling_period

    def __call__(self, *args, **kwargs):
        if self.current_period == self.sampling_period:
            self.weight += self.gradient


class ShuffleIRISDataset(Operator):
    def __init__(self,
                 data_index,
                 current_period,
                 indexes,
                 dataset):
        # signal
        self.data_index = data_index
        self.current_period = current_period
        #
        self.indexes = indexes
        self.dataset = dataset
        self.num_data = self.dataset.shape[0]

    def __call__(self, *args, **kwargs):
        if self.current_period == 0:
            if self.data_index >= self.num_data:
                np.random.shuffle(self.indexes)


class SampleIRISData(Operator):
    def __init__(self,
                 data,
                 data_index,
                 current_period,
                 indexes,
                 dataset):
        # signals
        self.data = data
        self.data_index = data_index
        self.current_period = current_period
        #
        self.indexes = indexes
        self.dataset = dataset

    def __call__(self, *args, **kwargs):
        if self.current_period == 0:
            self.data[:] = self.dataset[self.indexes[self.data_index]]


class UpdateIndex(Operator):
    def __init__(self,
                 index,
                 num_data,
                 current_period,
                 sampling_period):
        #
        self.index = index
        #
        self.num_data = num_data
        self.current_period = current_period
        self.sampling_period = sampling_period

    def __call__(self, *args, **kwargs):
        if self.current_period == self.sampling_period:
            self.index += 1
            if self.index >= self.num_data:
                self.index.fill(0)


class SampleImage(Operator):
    def __init__(self,
                 image,
                 image_index,
                 current_period,
                 sampling_period,
                 num_images,
                 images):
        # signals
        self.image = image
        self.image_index = image_index
        self.current_period = current_period
        #
        self.sampling_period = sampling_period
        self.num_images = num_images
        self.images = images

    def __call__(self, *args, **kwargs):
        if self.current_period == 0:
            import numpy as np
            self.image_index.fill(np.random.randint(0, self.num_images))
            self.image[:] = self.images[self.image_index]


class SpikeTime(Operator):
    def __init__(self,
                 membrane_potential,
                 spike_time,
                 threshold,
                 current_period):
        self.membrane_potential = membrane_potential
        self.spike_time = spike_time
        self.threshold = threshold
        self.current_period = current_period

    def __call__(self, *args, **kwargs):
        # spike_time 중에서 not-to-fire 상태인 변수에 대해서만 아래 것을 계산하면 된다.
        self.spike_time[(self.spike_time < 0) & (self.membrane_potential > self.threshold)] = self.current_period
        self.membrane_potential[self.membrane_potential > self.threshold] = 0


class IRISPopulationEncoder(Operator):
    def __init__(self,
                 spike_time,
                 data,
                 current_period,
                 sampling_period,
                 num_neurons,
                 beta,
                 min_vals,
                 max_vals):
        # signals
        self.spike_time = spike_time
        self.data = data
        self.current_period = current_period
        #
        self.num_neurons = num_neurons
        self.sampling_period = sampling_period
        self.beta = beta
        self.min_vals = min_vals
        self.max_vals = max_vals

        self.means = np.zeros(shape=(self.data.shape[0], self.num_neurons))
        self.stds = np.zeros(shape=(self.data.shape[0]))

        # Compute means and stds for normal distributions
        for i in range(self.data.shape[0]):
            for j in range(self.num_neurons):
                x = j + 1
                self.means[i, j] = self.min_vals[i]+(2*x-3)*(self.max_vals[i]-self.min_vals[i])/(2*(self.num_neurons-2))
            self.stds[i] = (max_vals[i]-min_vals[i])/(self.beta*(self.num_neurons-2))

        self.max_pdfs = [norm.pdf(0, scale=self.stds[i]) for i in range(self.stds.shape[0])]

    def __call__(self, *args, **kwargs):
        if self.current_period == 0:
            for i in range(self.means.shape[0]):
                for j in range(self.means.shape[1]):
                    self.spike_time[i*self.means.shape[1] + j] = self._transform_spike_time(
                        norm.pdf(self.data[i], self.means[i, j], self.stds[i]), self.max_pdfs[i])

            self.spike_time[-1] = 0
            self.spike_time[-2] = 0

    def _transform_spike_time(self, response, max_pdf):
        max_spike_time = self.sampling_period
        max_response = max_pdf
        spike_time = response * max_spike_time / max_response
        spike_time = spike_time - max_spike_time
        spike_time = spike_time * -1.0
        spike_time = round(spike_time)
        return spike_time


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


class UpdateFiringRate(Operator):
    def __init__(self,
                 firing_rate,
                 image,
                 current_period):
        super().__init__()
        # signals
        self.firing_rate = firing_rate
        self.image = image
        self.current_period = current_period

    def __call__(self, *args, **kwargs):
        if self.current_period == 0:
            flattened_image = self.image.flatten()
            self.firing_rate[:] = 255.0 * flattened_image / 4.0


class PoissonSpikeGeneration(Operator):
    def __init__(self,
                 image,
                 spike,
                 firing_rate,
                 time_step=0.001):
        # signals
        self.image = image
        self.spike = spike
        self.firing_rate = firing_rate
        #
        self.time_step = time_step

    def __call__(self, *args, **kwargs):
        import numpy as np

        self.spike.fill(0)
        random_number = np.random.uniform(0, 1, self.spike.shape)
        prob_single_spike = self.firing_rate * self.time_step
        self.spike[random_number < prob_single_spike] = 1


class SpikeResponse(Operator):
    """Compute spike responses

    Compute spike responses using current time in a period and
    firing times from presynaptic neurons.

    """
    def __init__(self,
                 current_period,
                 spike_time,
                 spike_response,
                 tau=3.0):
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
        # TODO: 여기에 spike_time이 not-to-spike인 -1일
        #       때에 대한 경우 처리 방법 필요
        t = self.current_period - spike_time
        x = t / self.tau
        import numpy as np
        y = np.exp(1 - x)
        y = x * y
        y[(spike_time < 0) | (y < 0)] = 0
        self.spike_response[:] = y


class RMSE(Operator):
    def __init__(self,
                 prediction,
                 target,
                 error,
                 current_period,
                 sampling_period,
                 nan=-1):
        # inputs
        self.prediction = prediction
        self.target = target
        self.current_period = current_period
        # output
        self.error = error
        self.sampling_period = sampling_period
        # hyperparameter
        self.nan = nan

    def __call__(self, *args, **kwargs):
        if self.current_period == self.sampling_period:
            y_ = self.prediction[self.prediction > self.nan]
            y = self.target[self.prediction > self.nan]
            error = y_ - y
            error = error ** 2
            import numpy as np
            error = np.sum(error)
            error = error / 2.0
            self.error[0] = error


class ComputeOutputUpstreamGradient(Operator):
    def __init__(self,
                 gradient,
                 prediction,
                 target,
                 weights,
                 pre_spike_time,
                 current_period,
                 sampling_period,
                 tau=3.0):
        # signals
        self.gradient = gradient
        self.prediction = prediction
        self.target = target
        self.weights = weights
        self.pre_spike_time = pre_spike_time
        self.current_period = current_period
        #
        self.sampling_period = sampling_period
        self.tau = tau

    def __call__(self, *args, **kwargs):
        import numpy as np
        if self.current_period == self.sampling_period:
            numerator = np.zeros(shape=(3))
            numerator[self.prediction > 0] = self.target[self.prediction > 0] - self.prediction[self.prediction > 0]

            derivative_spike_response = np.zeros(shape=(3, 10))

            for i in range(derivative_spike_response.shape[0]):
                for j in range(derivative_spike_response.shape[1]):
                    t = self.prediction[i] - self.pre_spike_time[j]
                    if self.prediction[i] > 0 and self.pre_spike_time[j] > 0 and t >= 0:
                        first_term = np.exp(1.0 - (t / self.tau)) / self.tau
                        second_term = t * np.exp(1.0 - (t / self.tau)) / np.square(self.tau)
                        all_terms = first_term - second_term
                        derivative_spike_response[i, j] = all_terms

            denominator = np.zeros(shape=(3))

            for i in range(denominator.shape[0]):
                total = 0.0
                for j in range(derivative_spike_response.shape[1]):
                    total += self.weights[i, j] * derivative_spike_response[i, j]
                denominator[i] = total

            self.gradient.fill(0.0)
            self.gradient[denominator > 0] = numerator[denominator > 0] / denominator[denominator > 0]


class ComputeHiddenUpstreamGradient(Operator):
    def __init__(self,
                 upstream_gradient,
                 pre_upstream_gradient,
                 post_spike_time,
                 spike_time,
                 pre_spike_time,
                 post_weights,
                 pre_weights,
                 current_period,
                 sampling_period,
                 tau=3.0):
        # signals
        self.upstream_gradient = upstream_gradient
        self.pre_upstream_gradient = pre_upstream_gradient
        self.post_spike_time = post_spike_time
        self.spike_time = spike_time
        self.pre_spike_time = pre_spike_time
        self.post_weights = post_weights
        self.pre_weights = pre_weights
        self.current_period = current_period
        #
        self.sampling_period = sampling_period
        self.tau = tau

    def __call__(self, *args, **kwargs):
        if self.current_period == self.sampling_period:
            import numpy as np

            numerator = np.zeros(shape=(10))

            derivative_spike_response = np.zeros(shape=(3, 10))

            for i in range(derivative_spike_response.shape[0]):
                for j in range(derivative_spike_response.shape[1]):
                    t = self.post_spike_time[i] - self.spike_time[j]
                    if self.post_spike_time[i] > 0 and self.spike_time[j] > 0 and t >= 0:
                        first_term = np.exp(1.0 - (t / self.tau)) / self.tau
                        second_term = t * np.exp(1.0 - (t / self.tau)) / np.square(self.tau)
                        all_terms = first_term - second_term
                        derivative_spike_response[i, j] = all_terms

            for j in range(derivative_spike_response.shape[1]):
                total = 0.0
                for i in range(derivative_spike_response.shape[0]):
                    total += self.pre_upstream_gradient[i] * self.post_weights[i, j] * derivative_spike_response[i, j]
                numerator[j] = total

            denominator = np.zeros(shape=(10))

            derivative_spike_response = np.zeros(shape=(10, 50))

            if len(self.pre_spike_time.shape) > 1:
                pre_spike_time = self.pre_spike_time.flatten()
            else:
                pre_spike_time = self.pre_spike_time

            for i in range(derivative_spike_response.shape[0]):
                for j in range(derivative_spike_response.shape[1]):
                    t = self.spike_time[i] - pre_spike_time[j]
                    if self.spike_time[i] > 0 and pre_spike_time[j] > 0 and t >= 0:
                        first_term = np.exp(1.0 - (t / self.tau)) / self.tau
                        second_term = t * np.exp(1.0 - (t / self.tau)) / np.square(self.tau)
                        all_terms = first_term - second_term
                        derivative_spike_response[i, j] = all_terms

            for i in range(denominator.shape[0]):
                total = 0.0
                for j in range(derivative_spike_response.shape[1]):
                    total += self.pre_weights[i, j] * derivative_spike_response[i, j]
                denominator[i] = total

            self.upstream_gradient.fill(0.0)
            self.upstream_gradient[denominator > 0] = numerator[denominator > 0] / denominator[denominator > 0]


class ComputeOutputGradient(Operator):
    def __init__(self,
                 output_gradient,
                 output_upstream_gradient,
                 prediction,
                 pre_spike_time,
                 current_period,
                 sampling_period,
                 learning_rate=0.0075,
                 tau=3.0):
        # signals
        self.output_gradient = output_gradient
        self.output_upstream_gradient = output_upstream_gradient
        self.prediction = prediction
        self.pre_spike_time = pre_spike_time
        self.current_period = current_period
        #
        self.sampling_period = sampling_period
        self.learning_rate = learning_rate
        self.tau = tau

    def __call__(self, *args, **kwargs):
        if self.current_period == self.sampling_period:
            firing_time = np.tile(self.prediction, (10, 1))
            firing_time = np.transpose(firing_time, (1, 0))

            pre_firing_time = np.tile(self.pre_spike_time, (3, 1))

            t = firing_time - pre_firing_time
            x = t / self.tau
            y = np.exp(1 - x)
            y = x * y
            y[(firing_time < 0) | (pre_firing_time < 0) | (y < 0)] = 0

            upstream = np.tile(self.output_upstream_gradient, (10, 1))
            upstream = np.transpose(upstream, (1, 0))

            gradient = -self.learning_rate * y
            gradient *= upstream

            self.output_gradient[:] = gradient


class ComputeHiddenGradient(Operator):
    def __init__(self,
                 output_gradient,
                 output_upstream_gradient,
                 spike_time,
                 pre_spike_time,
                 current_period,
                 sampling_period,
                 learning_rate=0.0075,
                 tau=3.0):
        # signals
        self.output_gradient = output_gradient
        self.output_upstream_gradient = output_upstream_gradient
        self.spike_time = spike_time
        self.pre_spike_time = pre_spike_time
        self.current_period = current_period
        #
        self.sampling_period = sampling_period
        self.learning_rate = learning_rate
        self.tau = tau

    def __call__(self, *args, **kwargs):
        if self.current_period == self.sampling_period:
            firing_time = np.tile(self.spike_time, (50, 1))
            firing_time = np.transpose(firing_time, (1, 0))

            pre_firing_time = self.pre_spike_time.flatten()
            pre_firing_time = np.tile(pre_firing_time, (10, 1))

            t = firing_time - pre_firing_time
            x = t / self.tau
            y = np.exp(1 - x)
            y *= x
            y[(firing_time < 0) | (pre_firing_time < 0) | (y < 0)] = 0

            upstream = np.tile(self.output_upstream_gradient, (50, 1))
            upstream = np.transpose(upstream, (1, 0))

            gradient = -self.learning_rate * y
            gradient *= upstream

            self.output_gradient[:] = gradient


class UpdateConductance(Operator):
    def __init__(self,
                 conductance,
                 pre_spike,
                 weight,
                 init_value,
                 latest_firing_time,
                 simulation_time,
                 time_constant):
        self.conductance = conductance
        self.pre_spike = pre_spike
        self.weight = weight
        self.init_value = init_value
        self.latest_firing_time = latest_firing_time
        self.simultation_time = simulation_time
        self.time_constant = time_constant

    def __call__(self, *args, **kwargs):
        # pre_spike == 0
        self.latest_firing_time[self.pre_spike == 0] - self.simultation_time

        # pre_spike == 1


class _UpdateConductance(Operator):
    def __init__(self,
                 conductance,
                 pre_spike,
                 weight,
                 post_potential,
                 time_step=0.001):
        # signals
        self.conductance = conductance
        self.pre_spike = pre_spike
        self.weight = weight
        #
        self.post_potential = post_potential
        self.time_step = time_step

    def __call__(self, *args, **kwargs):
        [self.pre_spike == 1]


if __name__ == '__main__':
    import numpy as np

    conductance = np.array([1.0, 2.0, 3.0, 4.0]).reshape((2, 2))
    weight = np.array([10.0, 5.0, 15.0, 20.0]).reshape((2, 2))
    init_conductance = np.array([0.0, 0.0, 0.0, 0.0]).reshape((2, 2))
    pre_spike = np.array([0, 1], dtype=np.int)
    simulation_time = 0.001
    time_constant = 0.001

    print(pre_spike == 0)
    print(pre_spike)
    tiled_pre_spike = np.tile(pre_spike, (2, 1))
    print(tiled_pre_spike)
    print(tiled_pre_spike == 0)

    print(conductance)
    print(conductance[tiled_pre_spike == 0])

    conductance[:] = np.multiply(tiled_pre_spike == 0, np.multiply(weight, np.exp(-simulation_time/time_constant)))

    print(conductance)


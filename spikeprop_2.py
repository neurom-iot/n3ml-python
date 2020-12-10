import numpy as np
from sklearn.datasets import load_iris
from scipy.stats import norm
import matplotlib.pyplot as plt


def load_iris_dataset():
    iris = load_iris()
    return iris['data'], iris['target']


def get_statistics(inputs):
    min_vals = np.amin(inputs, axis=0)
    max_vals = np.amax(inputs, axis=0)
    return min_vals, max_vals


def transform_firing_times(responses, max_response, max_firing_time):
    num_rows, num_cols = responses.shape

    max_response = np.tile(max_response, (num_cols, 1))
    max_response = np.transpose(max_response, (1, 0))

    firing_times = responses * max_firing_time / max_response
    firing_times = firing_times - max_firing_time
    firing_times = firing_times * -1.0
    firing_times = np.around(firing_times)

    return np.array(firing_times)


def population_encoder(table, num_neurons, min_vals, max_vals, max_firing_time, threshold):
    num_rows = min_vals.shape[0]
    num_cols = num_neurons

    beta = 1.5
    locs = np.zeros(shape=(num_rows, num_cols))
    scales = np.zeros(shape=(num_rows))

    for i in range(locs.shape[0]):
        for j in range(locs.shape[1]):
            locs[i, j] = min_vals[i]+(2*(j+1)-3)*(max_vals[i]-min_vals[i])/(2*(num_neurons-2))
        scales[i] = (max_vals[i]-min_vals[i])/(beta*(num_neurons-2))

    # print(locs)
    # print(scales)

    responses = np.zeros(shape=(num_rows, num_cols))
    for i in range(responses.shape[0]):
        for j in range(responses.shape[1]):
            responses[i, j] = norm.pdf(table[i], locs[i, j], scales[i])

    # print(responses)

    max_responses = norm.pdf(0, scale=scales)

    firing_times = transform_firing_times(responses, max_responses, max_firing_time)
    firing_times[firing_times > threshold] = -1

    return firing_times


def compute_spike_responses(t, firing_times, spike_responses, tau, num_terminals, delays):
    # print("synaptic_terminal_delays:\n{}".format(delays))

    # times - firing_times - delays
    _firing_times = np.tile(firing_times, (num_terminals, 1))
    _firing_times = np.transpose(_firing_times, (1, 0))

    _t = t - _firing_times - delays  # _times.shape: (50, 16)

    x = _t / tau
    y = np.exp(1 - x)
    y = x * y
    y[(_firing_times < 0) | (y < 0)] = 0

    spike_responses[:] = y.flatten()


def compute_voltage(spike_responses, weights, voltages):
    voltages[:] = np.matmul(weights, spike_responses)


def compute_firing_times(t, voltages, firing_times, action_threshold):
    firing_times[(firing_times < 0) & (voltages > action_threshold)] = t


def Compute_target_firing_times(label, target_firing_times, max_firing_time, earliest_firing_time):
    target_firing_times[:] = max_firing_time
    target_firing_times[label] = earliest_firing_time


def Compute_loss(target_firing_times, firing_times, error):
    _y = firing_times[firing_times > 0]
    y = target_firing_times[firing_times > 0]
    res = _y - y
    res = res ** 2
    res = np.sum(res)
    res /= 2.0
    error.fill(res)


def Compute_derivative_spike_response(firing_time, pre_firing_time, delay, tau):
    t = firing_time - pre_firing_time - delay
    if t <= 0:
        return 0
    # t > 0
    y = np.exp((1-t)/tau)/tau
    y = y - (t*(np.exp((1-t)/tau)))/(tau**2)
    return y


def Compute_output_upstream_gradient(firing_times,
                                     pre_firing_times,
                                     delays,
                                     target_firing_times,
                                     weights,
                                     tau,
                                     output_upstream_derivatives,
                                     num_hidden_inh_neurons):
    num_hidden_neurons, num_terminals = delays.shape
    num_output_neurons = firing_times.shape[0]

    # The derivative of a spike response
    # print("firing times's shape: {}".format(firing_times.shape))
    # print("pre-firing times's shape: {}".format(pre_firing_times.shape))
    # print("delays: {}".format(delays.shape))

    # TODO: Validate this approach to compute in matrix form

    # _firing_times = np.tile(firing_times, (num_hidden_neurons*num_terminals, 1))
    # _firing_times = np.transpose(_firing_times, (1, 0))
    #
    # _pre_firing_times = np.tile(pre_firing_times, (num_terminals, 1))
    # _pre_firing_times = np.transpose(_pre_firing_times, (1, 0))
    # _pre_firing_times = _pre_firing_times.flatten()
    # _pre_firing_times = np.tile(_pre_firing_times, (num_output_neurons, 1))
    #
    # _delays = delays.flatten()
    # _delays = np.tile(_delays, (num_output_neurons, 1))

    _weights = np.reshape(weights, (num_output_neurons, num_hidden_neurons, num_terminals))

    # print("firing times's shape:\n{}".format(_firing_times))
    # print("pre-firing times's shape:\n{}".format(_pre_firing_times))
    # print("delays:\n{}".format(_delays))
    # print("weights's shape: {}".format(_weights.shape))

    for j in range(num_output_neurons):
        numerator = target_firing_times[j] - firing_times[j]

        # Compute denominator
        weighted_sum = 0.0

        for i in range(num_hidden_neurons):
            for l in range(num_terminals):
                derivative = Compute_derivative_spike_response(firing_times[j],
                                                               pre_firing_times[i],
                                                               delays[i, l],
                                                               tau)
                if num_hidden_neurons-i <= num_hidden_inh_neurons:
                    derivative *= -1
                weighted_sum += _weights[j, i, l] * derivative

        denominator = weighted_sum

        if denominator != 0.0:
            output_upstream_derivatives[j] = numerator / denominator
        else:
            output_upstream_derivatives[j] = 0.0


def compute_hidden_upstream_derivatives(post_firing_times,
                                        firing_times,
                                        pre_firing_times,
                                        post_delays,
                                        post_weights,
                                        pre_delays,
                                        pre_weights,
                                        hidden_upstream_derivatives,
                                        tau,
                                        output_upstream_derivatives,
                                        num_hidden_inh_neurons):
    # print(pre_delays.shape)  # (50, 16)
    # print(post_delays.shape)  # (10, 16)
    # print("pre_weights's shape: {}".format(pre_weights.shape))  # (10, 800)
    # print("post_weights's shape: {}".format(post_weights.shape))  # (3, 160)

    num_neurons = firing_times.shape[0]
    pre_num_neurons = pre_firing_times.shape[0]
    post_num_neurons = post_firing_times.shape[0]
    num_terminals = pre_delays.shape[1]

    _pre_weights = np.reshape(pre_weights, (num_neurons, pre_num_neurons, num_terminals))
    _post_weights = np.reshape(post_weights, (post_num_neurons, num_neurons, num_terminals))

    for i in range(num_neurons):
        numerator = 0.0

        for j in range(post_num_neurons):
            temp = 0.0

            for k in range(num_terminals):
                derivative = Compute_derivative_spike_response(post_firing_times[j],
                                                               firing_times[i],
                                                               post_delays[i, k],
                                                               tau)
                if num_hidden_inh_neurons-i <= num_hidden_inh_neurons:
                    derivative *= -1
                temp += _post_weights[j, i, k] * derivative

            numerator += output_upstream_derivatives[j] * temp

        denominator = 0.0

        for h in range(pre_num_neurons):
            for l in range(num_terminals):
                derivative = Compute_derivative_spike_response(firing_times[i],
                                                               pre_firing_times[h],
                                                               pre_delays[h, l],
                                                               tau)
                denominator += _pre_weights[i, h, l] * derivative

        if denominator != 0.0:
            hidden_upstream_derivatives[i] = numerator / denominator
        else:
            hidden_upstream_derivatives[i] = 0.0


def compute_spike_response(t_j, t_i, d_k, tau):
    if t_i < 0:
        return 0
    if t_j < 0:
        return 0
    t = (t_j - t_i - d_k) / tau
    if t <= 0:
        return 0
    # t > 0
    r = np.exp(1 - t)
    r = t * r
    return r


def compute_output_gradient(firing_times,
                            pre_firing_times,
                            delays,
                            output_upstream_derivatives,
                            tau,
                            output_derivatives,
                            learning_rate):
    num_output_neurons = firing_times.shape[0]
    num_hidden_neurons, num_terminals = delays.shape

    # print("output_derivatives's shape: {}".format(output_derivatives.shape))

    _output_derivatives = output_derivatives.reshape(num_output_neurons, num_hidden_neurons, num_terminals)

    # print("output_derivatives's shape: {}".format(_output_derivatives.shape))

    for j in range(num_output_neurons):
        for i in range(num_hidden_neurons):
            for k in range(num_terminals):
                r = compute_spike_response(firing_times[j], pre_firing_times[i], delays[i, k], tau)
                _output_derivatives[j, i, k] = -learning_rate * r * output_upstream_derivatives[j]

    output_derivatives[:] = _output_derivatives.reshape(num_output_neurons, num_hidden_neurons * num_terminals)

    # print("output_derivatives:\n{}".format(output_derivatives))


def update_weights(weights, derivatives):
    weights[:] = weights + derivatives
    weights[weights < 0.0] = 0.0


if __name__ == '__main__':
    num_iters = 2000
    simulation_time = 60
    time_step = 1
    num_neurons = 12
    num_terminals = 16
    max_firing_time = 12  # 10
    threshold = 13  # to determine longer as not-to-fire  # 9
    action_threshold = 1.0
    tau = 3.0
    min_delay = 1.0
    max_delay = 9.0
    num_input_neurons = 50
    num_hidden_neurons = 10
    num_hidden_inh_neurons = 2
    num_output_neurons = 3
    num_classes = 3
    learning_rate = 1.0
    learning_rate_schedule = [1000, 1500]
    earliest_firing_time = 6

    tables, labels = load_iris_dataset()

    min_vals, max_vals = get_statistics(tables)

    indexes = np.arange(tables.shape[0])

    ref = {}

    for i in range(num_iters):
        if i % 150 == 0:
            np.random.shuffle(indexes)

        if i == learning_rate_schedule[0] or i == learning_rate_schedule[1]:
            learning_rate *= 0.1

        firing_times = population_encoder(tables[indexes[i%150]], num_neurons, min_vals, max_vals, max_firing_time, threshold)

        ref['input/firing_times'] = np.zeros(shape=(firing_times.shape[0] * firing_times.shape[1] + 2))
        ref['input/firing_times'].fill(-1)
        ref['input/firing_times'][:firing_times.shape[0] * firing_times.shape[1]] = firing_times.flatten()
        ref['input/firing_times'][-2:].fill(0)

        if i == 0:
            ref['conn_01/delays'] = np.zeros(shape=(num_input_neurons, num_terminals))
            ref['conn_01/delays'][:] = np.random.uniform(min_delay, max_delay, size=(num_input_neurons, num_terminals))

        ref['input/spike_responses'] = np.zeros(shape=(num_input_neurons * num_terminals))

        if i == 0:
            ref['conn_01/weights'] = np.zeros(shape=(num_hidden_neurons, num_input_neurons * num_terminals))
            ref['conn_01/weights'][:] = np.random.uniform(0.9, 1.1,
                                                          size=(num_hidden_neurons, num_input_neurons * num_terminals))

        ref['hidden/voltages'] = np.zeros(shape=num_hidden_neurons)

        ref['hidden/firing_times'] = np.zeros(shape=num_hidden_neurons)
        ref['hidden/firing_times'].fill(-1)

        if i == 0:
            ref['conn_02/delays'] = np.zeros(shape=(num_hidden_neurons, num_terminals))
            ref['conn_02/delays'][:] = np.random.uniform(min_delay, max_delay, size=(num_hidden_neurons, num_terminals))

        ref['hidden/spike_responses'] = np.zeros(shape=(num_hidden_neurons * num_terminals))

        if i == 0:
            ref['conn_02/weights'] = np.zeros(shape=(num_output_neurons, num_hidden_neurons * num_terminals))
            ref['conn_02/weights'][:] = np.random.uniform(0.9, 1.1,
                                                          size=(num_output_neurons, num_hidden_neurons * num_terminals))

        ref['output/voltages'] = np.zeros(shape=num_output_neurons)

        ref['output/firing_times'] = np.zeros(shape=num_output_neurons)
        ref['output/firing_times'].fill(-1)

        ref['learning/target_firing_times'] = np.zeros(shape=num_output_neurons)

        ref['learning/error'] = np.array(0)

        ref['learning/output_upstream_derivatives'] = np.zeros(shape=num_output_neurons)

        ref['learning/hidden_upstream_derivatives'] = np.zeros(shape=num_hidden_neurons)

        ref['learning/output_derivatives'] = np.zeros(shape=ref['conn_02/weights'].shape)

        ref['learning/hidden_derivatives'] = np.zeros(shape=ref['conn_01/weights'].shape)

        # res_x = np.array([_ for _ in range(0, simulation_time, time_step)])
        # res_x = np.tile(res_x, (num_input_neurons * num_terminals, 1))
        # res_x = np.transpose(res_x, (1, 0))
        #
        # res_y = []
        #
        # vol_x = np.array([_ for _ in range(0, simulation_time, time_step)])
        # vol_x = np.tile(vol_x, (num_hidden_neurons, 1))
        # vol_x = np.transpose(vol_x, (1, 0))
        #
        # vol_y = []
        #
        # h_res_x = np.array([_ for _ in range(0, simulation_time, time_step)])
        # h_res_x = np.tile(h_res_x, (num_hidden_neurons * num_terminals, 1))
        # h_res_x = np.transpose(h_res_x, (1, 0))
        #
        # h_res_y = []
        #
        o_vol_x = np.array([_ for _ in range(0, simulation_time, time_step)])
        o_vol_x = np.tile(o_vol_x, (num_output_neurons, 1))
        o_vol_x = np.transpose(o_vol_x, (1, 0))

        o_vol_y = []

        for t in range(0, simulation_time, time_step):
            compute_spike_responses(t, ref['input/firing_times'], ref['input/spike_responses'], tau, num_terminals,
                                    ref['conn_01/delays'])

            # res_y.append(ref['input/spike_responses'].copy())

            # print("spike responses:\n{}".format(ref['input/spike_responses']))  # shape: (800,)

            compute_voltage(ref['input/spike_responses'], ref['conn_01/weights'], ref['hidden/voltages'])

            # print(ref['hidden/voltages'][-2:])

            # vol_y.append(ref['hidden/voltages'].copy())

            compute_firing_times(t, ref['hidden/voltages'], ref['hidden/firing_times'], action_threshold)

            # print(ref['hidden/firing_times'])

            compute_spike_responses(t, ref['hidden/firing_times'], ref['hidden/spike_responses'], tau, num_terminals,
                                    ref['conn_02/delays'])
            ref['hidden/spike_responses'][-num_hidden_inh_neurons * num_terminals:] *= -1

            # print("hidden's spike responses:\n{}".format(ref['hidden/spike_responses']))

            # h_res_y.append(ref['hidden/spike_responses'].copy())

            compute_voltage(ref['hidden/spike_responses'], ref['conn_02/weights'], ref['output/voltages'])

            # print(ref['output/voltages'])

            o_vol_y.append(ref['output/voltages'].copy())

            compute_firing_times(t, ref['output/voltages'], ref['output/firing_times'], action_threshold)

            # print(ref['output/firing_times'])

        print(ref['output/firing_times'])

        # res_y = np.array(res_y)
        #
        # for i in range(res_x.shape[1]):
        #     plt.plot(res_x[:, i], res_y[:, i])
        # plt.show()
        #
        # vol_y = np.array(vol_y)
        #
        # for i in range(vol_x.shape[1]):
        #     plt.plot(vol_x[:, i], vol_y[:, i])
        # plt.show()
        #
        # h_res_y = np.array(h_res_y)
        #
        # for i in range(h_res_x.shape[1]):
        #     plt.plot(h_res_x[:, i], h_res_y[:, i])
        # plt.show()
        #
        if i != 0 and i % 150 == 0:
            o_vol_y = np.array(o_vol_y)

            for i in range(o_vol_x.shape[1]):
                plt.plot(o_vol_x[:, i], o_vol_y[:, i])
            plt.show()

        # Learning
        Compute_target_firing_times(labels[indexes[i%150]],
                                    ref['learning/target_firing_times'],
                                    max_firing_time,
                                    earliest_firing_time)

        print(ref['learning/target_firing_times'])

        Compute_loss(ref['learning/target_firing_times'], ref['output/firing_times'], ref['learning/error'])

        print(ref['learning/error'])

        Compute_output_upstream_gradient(ref['output/firing_times'],
                                         ref['hidden/firing_times'],
                                         ref['conn_02/delays'],
                                         ref['learning/target_firing_times'],
                                         ref['conn_02/weights'],
                                         tau,
                                         ref['learning/output_upstream_derivatives'],
                                         num_hidden_inh_neurons)

        # print(ref['learning/output_upstream_derivatives'])

        compute_hidden_upstream_derivatives(ref['output/firing_times'],
                                            ref['hidden/firing_times'],
                                            ref['input/firing_times'],
                                            ref['conn_02/delays'],
                                            ref['conn_02/weights'],
                                            ref['conn_01/delays'],
                                            ref['conn_01/weights'],
                                            ref['learning/hidden_upstream_derivatives'],
                                            tau,
                                            ref['learning/output_upstream_derivatives'],
                                            num_hidden_inh_neurons)

        # print(ref['learning/hidden_upstream_derivatives'])

        compute_output_gradient(ref['output/firing_times'],
                                ref['hidden/firing_times'],
                                ref['conn_02/delays'],
                                ref['learning/output_upstream_derivatives'],
                                tau,
                                ref['learning/output_derivatives'],
                                learning_rate)

        # print(ref['learning/output_derivatives'])

        # compute_hidden_derivatives(ref['hidden/firing_times'],
        #                            ref['input/firing_times'],
        #                            ref['conn_01/delays'],
        #                            ref['learning/hidden_upstream_derivatives'],
        #                            ref['learning/hidden_derivatives'],
        #                            tau,
        #                            learning_rate)

        # Compute hidden derivatives
        compute_output_gradient(ref['hidden/firing_times'],
                                ref['input/firing_times'],
                                ref['conn_01/delays'],
                                ref['learning/hidden_upstream_derivatives'],
                                tau,
                                ref['learning/hidden_derivatives'],
                                learning_rate)

        # print(ref['learning/hidden_derivatives'])

        update_weights(ref['conn_02/weights'],
                       ref['learning/output_derivatives'])

        update_weights(ref['conn_01/weights'],
                       ref['learning/hidden_derivatives'])

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


def compute_spike_responses(times, firing_times, spike_responses, tau, num_terminals, min_delay, max_delay):
    delays = np.random.uniform(min_delay, max_delay, size=(firing_times.shape[0], num_terminals))

    # print("synaptic_terminal_delays:\n{}".format(delays))

    # times - firing_times - delays
    _firing_times = np.tile(firing_times, (num_terminals, 1))
    _firing_times = np.transpose(_firing_times, (1, 0))

    _times = times - _firing_times - delays  # _times.shape: (50, 16)

    x = _times / tau
    y = np.exp(1 - x)
    y = x * y
    y[(_firing_times < 0) | (y < 0)] = 0

    spike_responses[:] = y.flatten()


def compute_voltage(spike_responses, weights, voltages):
    voltages[:] = np.matmul(weights, spike_responses)


def compute_firing_times(t, voltages, firing_times, action_threshold):
    firing_times[(firing_times < 0) & (voltages > action_threshold)] = t


if __name__ == '__main__':
    # Load iris dataset
    tables, labels = load_iris_dataset()

    min_vals, max_vals = get_statistics(tables)

    num_neurons = 12
    num_terminals = 16
    max_firing_time = 10
    threshold = 9  # to determine longer as not-to-fire
    action_threshold = 1.0
    tau = 3.0
    min_delay = 1.0
    max_delay = 9.0
    num_input_neurons = 50
    num_hidden_neurons = 10
    num_steps = 60
    num_hidden_inh_neurons = 2
    num_output_neurons = 3

    firing_times = population_encoder(tables[0], num_neurons, min_vals, max_vals, max_firing_time, threshold)

    ref = {}

    ref['input/firing_times'] = np.zeros(shape=(firing_times.shape[0]*firing_times.shape[1]+2))
    ref['input/firing_times'].fill(-1)
    ref['input/firing_times'][:firing_times.shape[0]*firing_times.shape[1]] = firing_times.flatten()
    ref['input/firing_times'][-2:].fill(0)

    ref['input/spike_responses'] = np.zeros(shape=(num_input_neurons*num_terminals))

    ref['conn_01/weights'] = np.zeros(shape=(num_hidden_neurons, num_input_neurons*num_terminals))
    ref['conn_01/weights'][:] = np.random.uniform(0, 0.2, size=(num_hidden_neurons, num_input_neurons*num_terminals))

    ref['hidden/voltages'] = np.zeros(shape=num_hidden_neurons)

    ref['hidden/firing_times'] = np.zeros(shape=num_hidden_neurons)
    ref['hidden/firing_times'].fill(-1)

    ref['hidden/spike_responses'] = np.zeros(shape=(num_hidden_neurons*num_terminals))

    ref['conn_02/weights'] = np.zeros(shape=(num_output_neurons, num_hidden_neurons*num_terminals))
    ref['conn_02/weights'][:] = np.random.uniform(0, 0.2, size=(num_output_neurons, num_hidden_neurons*num_terminals))

    ref['output/voltages'] = np.zeros(shape=num_output_neurons)

    ref['output/firing_times'] = np.zeros(shape=num_output_neurons)
    ref['output/firing_times'].fill(-1)

    res_x = np.array([_ for _ in range(num_steps)])
    res_x = np.tile(res_x, (num_input_neurons*num_terminals, 1))
    res_x = np.transpose(res_x, (1, 0))

    res_y = []

    vol_x = np.array([_ for _ in range(num_steps)])
    vol_x = np.tile(vol_x, (num_hidden_neurons, 1))
    vol_x = np.transpose(vol_x, (1, 0))

    vol_y = []

    h_res_x = np.array([_ for _ in range(num_steps)])
    h_res_x = np.tile(h_res_x, (num_hidden_neurons*num_terminals, 1))
    h_res_x = np.transpose(h_res_x, (1, 0))

    h_res_y = []

    o_vol_x = np.array([_ for _ in range(num_steps)])
    o_vol_x = np.tile(o_vol_x, (num_output_neurons, 1))
    o_vol_x = np.transpose(o_vol_x, (1, 0))

    o_vol_y = []

    for t in range(num_steps):
        compute_spike_responses(t, ref['input/firing_times'], ref['input/spike_responses'], tau, num_terminals, min_delay, max_delay)

        res_y.append(ref['input/spike_responses'].copy())

        # print("spike responses:\n{}".format(ref['input/spike_responses']))  # shape: (800,)

        compute_voltage(ref['input/spike_responses'], ref['conn_01/weights'], ref['hidden/voltages'])

        # print(ref['hidden/voltages'][-2:])

        vol_y.append(ref['hidden/voltages'].copy())

        compute_firing_times(t, ref['hidden/voltages'], ref['hidden/firing_times'], action_threshold)

        # print(ref['hidden/firing_times'])

        compute_spike_responses(t, ref['hidden/firing_times'], ref['hidden/spike_responses'], tau, num_terminals, min_delay, max_delay)
        ref['hidden/spike_responses'][-num_hidden_inh_neurons*num_terminals:] *= -1

        # print("hidden's spike responses:\n{}".format(ref['hidden/spike_responses']))

        h_res_y.append(ref['hidden/spike_responses'].copy())

        compute_voltage(ref['hidden/spike_responses'], ref['conn_02/weights'], ref['output/voltages'])

        # print(ref['output/voltages'])

        o_vol_y.append(ref['output/voltages'].copy())

        compute_firing_times(t, ref['output/voltages'], ref['output/firing_times'], action_threshold)

        print(ref['output/firing_times'])

    res_y = np.array(res_y)

    for i in range(res_x.shape[1]):
        plt.plot(res_x[:, i], res_y[:, i])
    plt.show()

    vol_y = np.array(vol_y)

    for i in range(vol_x.shape[1]):
        plt.plot(vol_x[:, i], vol_y[:, i])
    plt.show()

    h_res_y = np.array(h_res_y)

    for i in range(h_res_x.shape[1]):
        plt.plot(h_res_x[:, i], h_res_y[:, i])
    plt.show()

    o_vol_y = np.array(o_vol_y)

    for i in range(o_vol_x.shape[1]):
        plt.plot(o_vol_x[:, i], o_vol_y[:, i])
    plt.show()

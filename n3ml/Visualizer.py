import numpy as np
import matplotlib.pyplot as plt

def rasterplot(time, spikes, **kwargs):
    ax = plt.gca()

    n_spike, n_neuron = spikes.shape

    kwargs.setdefault("linestyle", "None")
    kwargs.setdefault("marker", "|")

    spiketimes = []

    for i in range(n_neuron):
        temp = time[spikes[:, i] > 0].ravel()
        spiketimes.append(temp)

    spiketimes = np.array(spiketimes)

    indexes = np.zeros(n_neuron, dtype=np.int)

    for t in range(time.shape[0]):
        for i in range(spiketimes.shape[0]):
            if spiketimes[i].shape[0] <= 0:
                continue
            if indexes[i] < spiketimes[i].shape[0] and \
                    time[t] == spiketimes[i][indexes[i]]:
                ax.plot(spiketimes[i][indexes[i]], i + 1, 'k', **kwargs)

                plt.draw()
                plt.pause(0.002)

                indexes[i] += 1


def test_rasterplot():
    total_time = 1000

    time = np.arange(0, total_time)
    spikes = np.zeros(shape=(total_time, 5))

    print(time)
    print(spikes)

    for i in range(spikes.shape[1]):
        spikes[:, i] = np.random.uniform(0, 1, total_time)
        spikes[spikes > 0.5] = 1
        spikes[spikes < 0.5] = 0

    print(spikes)

    rasterplot(time, spikes)


def plot_result(images, spikes, **kwargs):
    images = np.array(images)

    n_spike, n_neuron = spikes.shape

    total_time = n_spike

    time = np.arange(0, total_time)

    kwargs.setdefault("linestyle", "None")
    kwargs.setdefault("marker", "|")

    spiketimes = []

    for i in range(n_neuron):
        temp = time[spikes[:, i] > 0].ravel()
        spiketimes.append(temp)

    spiketimes = np.array(spiketimes)

    indexes = np.zeros(n_neuron, dtype=np.int)

    interval = int(total_time / 10)

    fig, ax = plt.subplots(1, 2)

    for i in range(images.shape[0]):
        ax[0].imshow(images[i])

        for t in range(interval*i, interval*(i+1), 1):
            for i in range(spiketimes.shape[0]):
                if spiketimes[i].shape[0] <= 0:
                    continue
                if indexes[i] < spiketimes[i].shape[0] and time[t] == spiketimes[i][indexes[i]]:
                    ax[1].plot(spiketimes[i][indexes[i]], i + 1, 'k', **kwargs)

                    plt.draw()
                    plt.pause(0.5)

                    indexes[i] += 1


if __name__ == '__main__':
    plot_result()
    #test_rasterplot()
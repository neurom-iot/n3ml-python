class Builder:
    @classmethod
    def build(cls, model, obj):
        from n3ml.Network import Network
        from n3ml.Source import MNISTSource
        from n3ml.Population import SRMPopulation
        from n3ml.Connection import Connection

        if isinstance(obj, Network):
            return build_network(model, obj)
        elif isinstance(obj, MNISTSource):
            return build_mnistsource(model, obj)
        elif isinstance(obj, SRMPopulation):
            return build_srmpopulation(model, obj)
        elif isinstance(obj, Connection):
            return build_connection(model, obj)

        raise NotImplementedError


def build_network(model, network):
    import numpy as np
    from n3ml.Source import Source
    from n3ml.Connection import Connection
    from n3ml.Population import Population
    from n3ml.Operators import UpdateTime

    model.signal['threshold'] = np.array(network.threshold)
    model.signal['current_time'] = np.array(0)

    model.add_op(UpdateTime(current_time=model.signal['current_time']))

    for obj in network.source + network.population:
        if isinstance(obj, Source):
            build_mnistsource(model, obj)
        elif isinstance(obj, Population):
            build_srmpopulation(model, obj)

    for obj in network.connection:
        if isinstance(obj, Connection):
            build_connection(model, obj)

    if network.learning:
        build_spikeprop(model, network)


def build_mnistsource(model, mnistsource):
    import numpy as np
    from n3ml.Operators import UpdatePeriodAndImage, PopulationEncode

    model.signal[mnistsource] = {}

    model.signal[mnistsource]['sampling_period'] = np.array(mnistsource.sampling_period)
    model.signal[mnistsource]['current_period'] = np.array(0)
    model.signal[mnistsource]['data_index'] = np.array(np.random.randint(0, mnistsource.num_images))
    model.signal[mnistsource]['image'] = np.array(mnistsource.images[model.signal[mnistsource]['data_index']])
    model.signal[mnistsource]['spike_time'] = np.zeros(
        shape=(mnistsource.rows * mnistsource.cols, mnistsource.num_neurons))
    # TODO: This implementation can be improved for consistency.
    #       For example, in multiple sources, we need ideas to manage multiple periods
    #       to simulate properly.
    model.signal['sampling_period'] = model.signal[mnistsource]['sampling_period']
    model.signal['current_period'] = model.signal[mnistsource]['current_period']

    # Fill nan values to spike time to represent not-to-fire
    model.signal[mnistsource]['spike_time'].fill(model.nan)

    model.add_op(UpdatePeriodAndImage(current_period=model.signal[mnistsource]['current_period'],
                                      image=model.signal[mnistsource]['image'],
                                      data_index=model.signal[mnistsource]['data_index'],
                                      sampling_period=model.signal[mnistsource]['sampling_period'],
                                      num_images=mnistsource.num_images,
                                      images=mnistsource.images))
    model.add_op(PopulationEncode(image=model.signal[mnistsource]['image'],
                                  spike_time=model.signal[mnistsource]['spike_time'],
                                  num_neurons=mnistsource.num_neurons,
                                  sampling_period=mnistsource.sampling_period,
                                  beta=mnistsource.beta,
                                  min_value=mnistsource.min_value,
                                  max_value=mnistsource.max_value))


def build_srmpopulation(model, srmpopulation):
    import numpy as np
    from n3ml.Operators import SpikeTime

    model.signal[srmpopulation] = {}

    model.signal[srmpopulation]['membrane_potential'] = np.zeros(shape=srmpopulation.num_neurons)
    model.signal[srmpopulation]['spike_time'] = np.zeros(shape=srmpopulation.num_neurons)

    # Fill nan values to spike time to represent not-to-fire
    model.signal[srmpopulation]['spike_time'].fill(model.nan)

    model.add_op(SpikeTime(membrane_potential=model.signal[srmpopulation]['membrane_potential'],
                           spike_time=model.signal[srmpopulation]['spike_time'],
                           threshold=model.signal['threshold'],
                           current_time=model.signal['current_time']))


def build_connection(model, connection):
    import numpy as np
    from n3ml.Source import Source
    from n3ml.Population import Population
    from n3ml.Operators import SpikeResponse, MatMul

    if isinstance(connection.pre, Population):
        pre_num_neurons = connection.pre.num_neurons
    elif isinstance(connection.pre, Source):
        pre_num_neurons = connection.pre.rows * connection.pre.cols * connection.pre.num_neurons
    else:
        raise ValueError

    # Assume that postsynaptic object always is an object of Population class
    post_num_neurons = connection.post.num_neurons

    model.signal[connection] = {}

    model.signal[connection]['spike_response'] = np.zeros(shape=pre_num_neurons)
    # TODO: How to initialise synaptic weight?
    model.signal[connection]['synaptic_weight'] = np.zeros(shape=(post_num_neurons, pre_num_neurons))

    model.add_op(SpikeResponse(current_period=model.signal['current_period'],
                               spike_time=model.signal[connection.pre]['spike_time'],
                               spike_response=model.signal[connection]['spike_response']))

    model.add_op(MatMul(weight_matrix=model.signal[connection]['synaptic_weight'],
                        inp_vector=model.signal[connection]['spike_response'],
                        out_vector=model.signal[connection.post]['membrane_potential']))


def build_spikeprop(model, network):
    pass


if __name__ == '__main__':
    import gzip
    import numpy as np
    import matplotlib.pyplot as plt

    with gzip.open('../data/train-images-idx3-ubyte.gz') as bytestream:
        bytestream.read(4)
        num_images = np.frombuffer(bytestream.read(4), dtype='>i4')[0]
        rows = np.frombuffer(bytestream.read(4), dtype='>i4')[0]
        cols = np.frombuffer(bytestream.read(4), dtype='>i4')[0]
        raw_data = np.frombuffer(bytestream.read(), dtype='>u1')

    images = raw_data.reshape(num_images, rows, cols)

    #plt.imshow(images[32321], cmap='gray')
    #plt.show()

    with gzip.open('../data/train-labels-idx1-ubyte.gz') as bytestream:
        bytestream.read(4)
        num_labels = np.frombuffer(bytestream.read(4), dtype='>i4')[0]
        raw_data = np.frombuffer(bytestream.read(), dtype='>u1')


























    labels = raw_data.reshape(num_labels)

    print(labels)


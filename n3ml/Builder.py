import numpy as np

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
    from n3ml.Source import Source, MNISTSource, IRISSource
    from n3ml.Connection import Connection
    from n3ml.Population import Population, SRMPopulation, Processing
    from n3ml.Operators import UpdateTime, UpdatePeriod
    import n3ml.Learning as Learning

    model.signal['threshold'] = np.array(network.threshold)
    model.signal['current_time'] = np.array(0)
    model.signal['current_period'] = np.array(0)

    model.add_op(UpdateTime(current_time=model.signal['current_time']))
    model.add_op(UpdatePeriod(current_period=model.signal['current_period'],
                              sampling_period=10))

    for obj in network.source + network.population:
        if isinstance(obj, Source):
            if isinstance(obj, MNISTSource):
                build_mnistsource(model, obj)
            elif isinstance(obj, IRISSource):
                build_irissource(model, obj)
        elif isinstance(obj, Population):
            if isinstance(obj, SRMPopulation):
                build_srmpopulation(model, obj)
            elif isinstance(obj, Processing):
                build_processing(model, obj)

    for obj in network.connection:
        if isinstance(obj, Connection):
            build_connection(model, obj)

    # TODO: We're going to design multiple learning algorithms in a single network.
    if isinstance(network.learning, Learning.STDP):
        pass
    elif isinstance(network.learning, Learning.SpikeProp):
        pass
        # build_spikeprop(model, network)


def build_mnistsource(model, mnistsource):
    import numpy as np
    import n3ml.Operators as Operators

    model.signal[mnistsource] = {}
    model.signal[mnistsource]['image'] = np.zeros(
        shape=(mnistsource.rows, mnistsource.cols))
    model.signal[mnistsource]['image_index'] = np.array(0)

    model.add_op(Operators.SampleImage(image=model.signal[mnistsource]['image'],
                                       image_index=model.signal[mnistsource]['image_index'],
                                       current_period=model.signal['current_period'],
                                       sampling_period=mnistsource.sampling_period,
                                       num_images=mnistsource.num_images,
                                       images=mnistsource.images))

    if mnistsource.code == 'population':
        model.signal[mnistsource]['spike_time'] = np.zeros(
            shape=(mnistsource.rows * mnistsource.cols, mnistsource.num_neurons))

        model.add_op(Operators.InitSpikeTime(spike_time=model.signal[mnistsource]['spike_time'],
                                             current_period=model.signal['current_period'],
                                             value=model.nan))

        model.add_op(Operators.PopulationEncode(image=model.signal[mnistsource]['image'],
                                                spike_time=model.signal[mnistsource]['spike_time'],
                                                num_neurons=mnistsource.num_neurons,
                                                sampling_period=mnistsource.sampling_period,
                                                beta=mnistsource.beta,
                                                min_value=mnistsource.min_value,
                                                max_value=mnistsource.max_value))
    elif mnistsource.code == 'poisson':
        model.signal[mnistsource]['firing_rate'] = np.zeros(
            shape=(mnistsource.rows * mnistsource.cols))
        model.signal[mnistsource]['spike'] = np.zeros(
            shape=(mnistsource.rows * mnistsource.cols))

        model.add_op(Operators.UpdateFiringRate(firing_rate=model.signal[mnistsource]['firing_rate'],
                                                image=model.signal[mnistsource]['image'],
                                                current_period=model.signal['current_period']))

        model.add_op(Operators.PoissonSpikeGeneration(image=model.signal[mnistsource]['image'],
                                                      spike=model.signal[mnistsource]['spike'],
                                                      firing_rate=model.signal[mnistsource]['firing_rate']))


def build_irissource(model, source):
    from n3ml.Operators import ShuffleIRISDataset
    from n3ml.Operators import SampleIRISData
    from n3ml.Operators import IRISPopulationEncoder
    from n3ml.Operators import InitSpikeTime

    model.signal[source] = {}

    model.signal[source]['data'] = np.zeros(shape=(source.dataset.shape[1]))
    model.signal[source]['data_index'] = np.array(0)

    model.add_op(ShuffleIRISDataset(data_index=model.signal[source]['data_index'],
                                    current_period=model.signal['current_period'],
                                    indexes=source.indexes,
                                    dataset=source.dataset))

    model.add_op(SampleIRISData(data=model.signal[source]['data'],
                                data_index=model.signal[source]['data_index'],
                                current_period=model.signal['current_period'],
                                indexes=source.indexes,
                                dataset=source.dataset))

    model.signal[source]['spike_time'] = np.zeros(
        shape=(source.num_neurons * source.dataset.shape[1] + 2))

    model.add_op(InitSpikeTime(spike_time=model.signal[source]['spike_time'],
                               current_period=model.signal['current_period'],
                               value=model.nan))

    model.add_op(IRISPopulationEncoder(spike_time=model.signal[source]['spike_time'],
                                       data=model.signal[source]['data'],
                                       current_period=model.signal['current_period'],
                                       sampling_period=source.sampling_period,
                                       num_neurons=source.num_neurons,
                                       beta=source.beta,
                                       min_vals=source.min_vals,
                                       max_vals=source.max_vals))


def build_srmpopulation(model, srmpopulation):
    import numpy as np
    import n3ml.Operators as Operators

    model.signal[srmpopulation] = {}

    model.signal[srmpopulation]['membrane_potential'] = np.zeros(shape=srmpopulation.num_neurons)
    model.signal[srmpopulation]['spike_time'] = np.zeros(shape=srmpopulation.num_neurons)

    model.add_op(Operators.InitSpikeTime(spike_time=model.signal[srmpopulation]['spike_time'],
                                         current_period=model.signal['current_period'],
                                         value=model.nan))

    model.add_op(Operators.SpikeTime(membrane_potential=model.signal[srmpopulation]['membrane_potential'],
                                     spike_time=model.signal[srmpopulation]['spike_time'],
                                     threshold=model.signal['threshold'],
                                     current_period=model.signal['current_period']))


def build_connection(model, connection):
    import numpy as np
    from n3ml.Source import Source, MNISTSource, IRISSource
    from n3ml.Population import Population
    from n3ml.Operators import SpikeResponse, MatMul, InitWeight

    if isinstance(connection.pre, Population):
        pre_num_neurons = connection.pre.num_neurons
    elif isinstance(connection.pre, Source):
        if isinstance(connection.pre, MNISTSource):
            pre_num_neurons = connection.pre.rows * connection.pre.cols * connection.pre.num_neurons
        elif isinstance(connection.pre, IRISSource):
            pre_num_neurons = model.signal[connection.pre]['spike_time'].shape[0]
        else:
            raise ValueError
    else:
        raise ValueError

    # Assume that postsynaptic object is always an object of Population class
    post_num_neurons = connection.post.num_neurons

    model.signal[connection] = {}

    model.signal[connection]['spike_response'] = np.zeros(shape=pre_num_neurons)
    model.signal[connection]['synaptic_weight'] = np.zeros(shape=(post_num_neurons, pre_num_neurons))

    model.add_op(InitWeight(weight=model.signal[connection]['synaptic_weight'],
                            current_time=model.signal['current_period'],
                            random_process=np.random.uniform))

    model.add_op(SpikeResponse(current_period=model.signal['current_period'],
                               spike_time=model.signal[connection.pre]['spike_time'],
                               spike_response=model.signal[connection]['spike_response']))

    model.add_op(MatMul(weight_matrix=model.signal[connection]['synaptic_weight'],
                        inp_vector=model.signal[connection]['spike_response'],
                        out_vector=model.signal[connection.post]['membrane_potential']))


def build_processing(model, processing):
    import numpy as np
    import n3ml.Operators as Operators

    model.signal[processing] = {}

    # for excitatory neurons
    model.signal[processing]['v_exc'] = np.zeros(shape=(processing.num_neurons))
    model.signal[processing]['s_exc'] = np.zeros(shape=(processing.num_neurons))
    # for inhibitory neurons
    model.signal[processing]['v_inh'] = np.zeros(shape=(processing.num_neurons))
    model.signal[processing]['s_inh'] = np.zeros(shape=(processing.num_neurons))
    # for the conductances of synapses
    model.signal[processing]['weight_e'] = np.identity(n=processing.num_neurons)
    model.signal[processing]['g_e'] = np.zeros(
        shape=(processing.num_neurons, processing.num_neurons))
    model.signal[processing]['weight_i'] = np.subtract(
        np.ones(shape=(processing.num_neurons, processing.num_neurons)),
        np.identity(n=processing.num_neurons))
    model.signal[processing]['g_i'] = np.zeros(
        shape=(processing.num_neurons, processing.num_neurons))

    Operators.InitWeight(weight=model.signal[processing]['weight_e'],
                         current_period=model.signal['current_period'],
                         value=10.4)
    Operators.InitWeight(weight=model.signal[processing]['weight_i'],
                         current_period=model.signal['current_period'],
                         value=17.0)

    Operators.UpdateConductance(conductance=model.signal[processing]['g_e'],
                                pre_spike=model.signal[processing]['s_exc'],
                                weight=model.signal[processing]['weight_e'],
                                post_potential=0.01)    # time constant: 10ms
    Operators.UpdateConductance(conductance=model.signal[processing]['g_i'],
                                pre_spike=model.signal[processing]['s_inh'],
                                weight=model.signal[processing]['weight_i'],
                                post_potential=0.02)    # time constant: 20ms



def build_spikeprop(model, network):
    """Create the operators and the tensors of the backpropagation of SpikeProp

    :param model:
    :param network:
    :return:
    """
    import numpy as np
    from n3ml.Operators import RMSE, UpdateLabel, UpdateTarget, UpdateWeight
    from n3ml.Operators import ComputeOutputUpstreamGradient
    from n3ml.Operators import ComputeHiddenUpstreamGradient
    from n3ml.Operators import ComputeOutputGradient
    from n3ml.Operators import ComputeHiddenGradient

    learning = network.learning

    model.signal[learning] = {}

    model.signal[learning]['label'] = np.array([0])
    # TODO: Modify the arguement 'shape' using num_classes for consistency
    model.signal[learning]['target'] = np.zeros(shape=(10))
    model.signal[learning]['prediction'] = model.signal[network.population[-1]]['spike_time']
    model.signal[learning]['error'] = np.array([0.0])
    model.signal[learning]['output_upstream_gradient'] = np.zeros(shape=(10))
    model.signal[learning]['hidden_upstream_gradient'] = np.zeros(shape=(100))
    model.signal[learning]['output_gradient'] = np.zeros(shape=(10, 100))
    model.signal[learning]['hidden_gradient'] = np.zeros(shape=(100, 15680))

    model.add_op(UpdateLabel(label=model.signal[learning]['label'],
                             label_index=model.signal[network.source[0]]['image_index'],
                             labels=network.source[0].labels,
                             current_period=model.signal['current_period'],
                             sampling_period=network.source[0].sampling_period))

    model.add_op(UpdateTarget(target=model.signal[learning]['target'],
                              label=model.signal[learning]['label'],
                              current_period=model.signal['current_period'],
                              sampling_period=network.source[0].sampling_period))

    model.add_op(RMSE(prediction=model.signal[learning]['prediction'],
                      target=model.signal[learning]['target'],
                      error=model.signal[learning]['error'],
                      current_period=model.signal['current_period'],
                      sampling_period=network.source[0].sampling_period))

    model.add_op(ComputeOutputUpstreamGradient(gradient=model.signal[learning]['output_upstream_gradient'],
                                               prediction=model.signal[learning]['prediction'],
                                               target=model.signal[learning]['target'],
                                               weights=model.signal[network.connection[1]]['synaptic_weight'],
                                               pre_spike_time=model.signal[network.population[0]]['spike_time'],
                                               current_period=model.signal['current_period'],
                                               sampling_period=network.source[0].sampling_period))

    model.add_op(ComputeHiddenUpstreamGradient(upstream_gradient=model.signal[learning]['hidden_upstream_gradient'],
                                               pre_upstream_gradient=model.signal[learning]['output_upstream_gradient'],
                                               post_spike_time=model.signal[learning]['prediction'],
                                               spike_time=model.signal[network.population[0]]['spike_time'],
                                               pre_spike_time=model.signal[network.source[0]]['spike_time'],
                                               post_weights=model.signal[network.connection[1]]['synaptic_weight'],
                                               pre_weights=model.signal[network.connection[0]]['synaptic_weight'],
                                               current_period=model.signal['current_period'],
                                               sampling_period=network.source[0].sampling_period))

    model.add_op(ComputeOutputGradient(output_gradient=model.signal[learning]['output_gradient'],
                                       output_upstream_gradient=model.signal[learning]['output_upstream_gradient'],
                                       prediction=model.signal[learning]['prediction'],
                                       pre_spike_time=model.signal[network.population[0]]['spike_time'],
                                       current_period=model.signal['current_period'],
                                       sampling_period=network.source[0].sampling_period))

    model.add_op(ComputeHiddenGradient(output_gradient=model.signal[learning]['hidden_gradient'],
                                       output_upstream_gradient=model.signal[learning]['hidden_upstream_gradient'],
                                       spike_time=model.signal[network.population[0]]['spike_time'],
                                       pre_spike_time=model.signal[network.source[0]]['spike_time'],
                                       current_period=model.signal['current_period'],
                                       sampling_period=network.source[0].sampling_period))

    model.add_op(UpdateWeight(weight=model.signal[network.connection[1]]['synaptic_weight'],
                              gradient=model.signal[learning]['output_gradient'],
                              current_period=model.signal['current_period'],
                              sampling_period=network.source[0].sampling_period))

    model.add_op(UpdateWeight(weight=model.signal[network.connection[0]]['synaptic_weight'],
                              gradient=model.signal[learning]['hidden_gradient'],
                              current_period=model.signal['current_period'],
                              sampling_period=network.source[0].sampling_period))


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


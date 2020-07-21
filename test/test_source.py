import unittest
import numpy as np


class TestSource(unittest.TestCase):
    def test_init(self):
        import n3ml.source

        source = n3ml.source.Source(output_size=10,
                                    distribution=np.random.uniform)

        self.assertEqual(source.output_size, 10)

    def test_build(self):
        import n3ml.model
        import n3ml.source
        import n3ml.signal
        import n3ml.operators

        model = n3ml.model.Model()
        source = n3ml.source.Source(output_size=10,
                                    distribution=np.random.uniform)

        source.build(model)

        self.assertTrue(isinstance(model.signal[source]['output'], n3ml.signal.Signal))
        self.assertEqual(model.signal[source]['output'].shape[0], 10)
        self.assertTrue(isinstance(model.operator[0], n3ml.operators.Sample))


if __name__ == '__main__':
    unittest.main()

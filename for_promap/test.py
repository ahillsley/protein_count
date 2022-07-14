import numpy as np
import unittest
import pandas as pd
import estimate_params
from estimate_params import extract_traces
from trace_model import TraceModel
from fluorescence_model import FluorescenceModel, EmissionParams




class TestTraceModel(unittest.TestCase):
    def test_generate_trace(self):
        sim_trace_len = 100
        trace_simulator = TraceModel(EmissionParams(), 0.1, sim_trace_len)
        trace_simulator.set_params(0.5, 0.5)
        trace = trace_simulator.generate_trace(1)

        self.assertAlmostEqual(len(trace), sim_trace_len)

        return

    def test_p_trace_given_y(self):
        sim_trace_len = 1000
        trace_simulator = TraceModel(EmissionParams(), 0.1, sim_trace_len)
        trace_simulator.set_params(0.5, 0.5)
        trace = trace_simulator.generate_trace(1)

        probability = trace_simulator.p_trace_given_y(trace, 1)

        self.assertLessEqual(probability, 0)


class TestIntensityTrace(unittest.TestCase):
    def test(self):
        return


class TestFluorescenceModel(unittest.TestCase):
    def test_p_x_given_z(self):
        z = 5
        f_model = FluorescenceModel(EmissionParams(mu_i=100, mu_b=200))

        sample = f_model.sample_x_i_given_z_i(z)

        probs = np.zeros((10))
        for i in range(len(probs)):
            probs[i] = f_model.p_x_i_given_z_i(sample, i)
        most_likely = np.argmax(probs)
        
        self.assertEqual(z, most_likely)
        return


if __name__ == '__main__':
    unittest.main()

import numpy as np
import unittest
import pandas as pd
#from fit_real_data import read_image, clean_spots, process_image
import calibrate
from calibrate import normalize_trace, extract_traces, process_image
from trace_model import TraceModel
from fluorescence_model import ModelParams



class TestCalibration(unittest.TestCase):
    
   def test_normalize_trace(self):
       image_file_path = '../../Images/0525_5nM_1.tif'
       spots_file_path = '../../Images/0525_5_nM_1_spots.csv'
       
       img = calibrate.read_image(image_file_path)
       spots = pd.read_csv(spots_file_path)
       
       all_traces = extract_traces(img, spots)
       test_traces = all_traces[np.random.randint(0,all_traces.shape[0], 1),:]
       for trace in test_traces:
          scaled_trace, scale, background_mu = normalize_trace(trace)
          
          # check that signal peak is greater than background peak
          self.assertGreaterEqual(1/scale, 0)
          
          # check that trace is shifted to mean 1
          self.assertAlmostEqual(np.mean(scaled_trace), 1, delta = 0.1)
"""       
   def test_fits(self):
        image_file_path = '../../Images/0525_5nM_1.tif'
        spots_file_path = '../../Images/0525_5_nM_1_spots.csv'
        
        img = calibrate.read_image(image_file_path)
        spots = pd.read_csv(spots_file_path)
        yx_spots = calibrate.clean_spots(spots, img)
        
        test_spots = yx_spots[np.random.randint(0, yx_spots.shape[0], 2)]
        p_values = process_image(img, test_spots)
        
        # check probabilites are between 0 and 1
        self.assertGreaterEqual(np.min(p_values), 0)
        self.assertLessEqual(np.max(p_values), 1)
"""
class TestTraceModel(unittest.TestCase):
    def test_generate_trace(self):
        sim_trace_len = 100
        trace_simulator = TraceModel(ModelParams(), 0.1, sim_trace_len)
        trace_simulator.set_params(0.5, 0.5)
        trace = trace_simulator.generate_trace(1)
        
        self.assertAlmostEqual(len(trace), sim_trace_len)
        
        return
    
    def test_p_trace_given_y(self):
        sim_trace_len = 1000
        trace_simulator = TraceModel(ModelParams(), 0.1, sim_trace_len)
        trace_simulator.set_params(0.5, 0.5)
        trace = trace_simulator.generate_trace(1)
        
        probability = trace_simulator.p_trace_given_y(trace, 1)
        
        self.assertLessEqual(probability, 0)


class TestIntensityTrace(unittest.TestCase):
    def test(self):
        return

class TestFluorescenceModel(unittest.TestCase):
    def test(self):
        return
    
    
if __name__ == '__main__':
    unittest.main()
    
    
    

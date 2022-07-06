import numpy as np
import unittest
import pandas as pd
from fit_real_data import read_image, clean_spots, process_image
from calibrate import normalize_trace, extract_traces

class TestFitRealData(unittest.TestCase):

    def test_fits(self):
        image_file_path = '../../Images/0525_5nM_1.tif'
        spots_file_path = '../../Images/0525_5_nM_1_spots.csv'
        
        img = read_image(image_file_path)
        spots = pd.read_csv(spots_file_path)
        yx_spots = clean_spots(spots, img)
        
        test_spots = yx_spots[np.random.randint(0, yx_spots.shape[0], 2)]
        p_values = process_image(img, test_spots)
        
        # check probabilites are between 0 and 1
        self.assertGreaterEqual(np.min(p_values), 0)
        self.assertLessEqual(np.max(p_values), 1)


class TestCalibration(unittest.TestCase):
    
   def test_normalize_trace(self):
       image_file_path = '../../Images/0525_5nM_1.tif'
       spots_file_path = '../../Images/0525_5_nM_1_spots.csv'
       
       img = read_image(image_file_path)
       spots = pd.read_csv(spots_file_path)
       
       all_traces = extract_traces(img, spots)
       test_traces = all_traces[np.random.randint(0,all_traces.shape[0], 5),:]
       for trace in test_traces:
          scale = normalize_trace(trace)
          
          self.assertGreaterEqual(1/scale, 0)
       

#class TestTraceModel(unittest.TestCase):
    # a


#class TestIntensityTrace(unittest.TestCase):
    # a

#class TestFluorescenceModel(unittest.TestCase):
    #a
    
    
if __name__ == '__main__':
    unittest.main()
    
    
    

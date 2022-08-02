from fluorescence_model import EmissionParams, FluorescenceModel
from trace_model import TraceModel
from intensity_trace import IntensityTrace
from estimate_params import read_image, clean_spots, pixel_t_trace
from literature_algs import extract_eps
import numpy as np
import scipy
import time
import pandas as pd
import matplotlib.pyplot as plt

# load and pre-process image and spots
image_path = '../../Images/N=6/N6_roi.tif'
spot_path = '../../Images/N=6/N6_roi_spots.csv'
image = read_image(image_path)
spots = pd.read_csv(spot_path)
spots = spots.rename(columns = {'X' : 'x', 'Y': 'y'})
spot_list = clean_spots(spots, image)
max_int = np.argmax(image, axis=2)

# identify good looking spots
good_ones = [33, 22, 25, 27]
i = 1
x,y  = spot_list[good_ones[i],:]
trace = pixel_t_trace(image, y, x, 3)

# Rough Initial guesses for parameters
spot_roi = np.ravel(image[(y-2):(y+2),(x-2):(x+2),:])
mu_b_initial = float(scipy.stats.mode(spot_roi)[0])
sigma_b_initial = np.sqrt(np.var(np.log(spot_roi)))
mu_1_peak = extract_eps(trace)
mu_i_initial = mu_1_peak - mu_b_initial
sigma_i_initial = 0.5

e_params = EmissionParams(mu_i=mu_i_initial, sigma_i=sigma_i_initial,
                          mu_b=mu_b_initial, sigma_b=sigma_b_initial)

t_model = TraceModel(e_params, 0.1, len(trace))
#t_model.set_params(0.04, 0.05)


''' exp 1: 
    - re-estimate all paramteres for each y
    - compare probabilities for each optimal value 
    - probs = = [-26086.83412553, -25909.31711038, -25920.38960636, -25924.88928099]
    best_ps =  [0.04898035, 0.00408261],
               [0.02040906, 0.01224584],
               [0.00816422, 0.01224584],
               [0.00408261, 0.00816422]
    '''
    
ys = [1, 2, 3, 4, 5, 6]

points = 50
best_ps = np.zeros((len(ys),2))
probs = np.zeros((len(ys)))

start_time = time.time()
for i, y in enumerate(ys):
    t_model.p_on = None
    t_model.p_off = None
    best_ps[i,0], best_ps[i,1] = t_model._line_search_params(trace, y,
                                                             points=points,
                                                             p_on_max=0.2,
                                                             p_off_max=0.2)
    probs[i] = t_model.p_trace_given_y(trace, y)
    print(f'fit y = {y}')
print('all ys fit')
print(f'ran in {time.time() - start_time}s')


''' Exp 2:
    try all combinations of p_on / p_off / y and maximize likelyhood'''
p_ons = np.linspace(1e-6, 0.1, 20)
probs = np.zeros((len(p_ons), len(p_ons), len(ys)))
for i, y in enumerate(ys):
    for j, p_on in enumerate(p_ons):
        for z, p_off in enumerate(p_ons):
            t_model.set_params(p_on, p_off)
            probs[j,z,i] = t_model.p_trace_given_y(trace, y)
            print(probs[j,z,i])
    print(f'done y = {y}')
    















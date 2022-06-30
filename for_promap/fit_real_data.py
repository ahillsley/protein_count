import numpy as np
import skimage.io, scipy
import pandas as pd
from FluorescenceModel import ModelParams
from TraceModel import TraceModel
from IntensityTrace import IntensityTrace
import matplotlib.pyplot as plt


image_file_path = '../../Images/0525_5nM_1.tif'
spots_file_path = '../../Images/0525_5_nM_1_spots.csv'

def read_image(file_path):
    img = skimage.io.imread(file_path)
    np_img = np.array(img)
    np_img = np.moveaxis(np_img, 0,2)
    
    return np_img

def max_int(img):
    return np.max(img, axis=2)

def pixel_t_trace(img, y ,x, pxl_size):
    if pxl_size == 0:
        crop = img[y,x,:]
    else:
        crop = img[(y-pxl_size):(y+pxl_size), (x-pxl_size):(x+pxl_size),:]
    
        crop = np.max(crop, axis=0)
        crop = np.max(crop, axis=0)
    #trace = img[x,y,:]
    
    return crop
def background_simple(img, trace, x, y, tile=7):
    baseline = np.zeros((len(trace)))
    a = y-tile
    if a < 0: 
        a=0
    b = y+tile
    if b > img.shape[0]:
        b = img.shape[0]
    c = x-tile
    if c < 0:
        c=0
    d = x+tile
    if d > img.shape[1]:
        d = img.shape[1]
        
    for i in range(len(trace)):
        baseline[i] = np.mean(img[a:b,c:d,i])
    
    background_subtracted = trace - baseline
    return background_subtracted

def clean_spots(spots, img):
    spots = np.vstack((np.asarray(spots['y'], ), 
                       np.asarray(spots['x']))).astype('int')
    
    bad_spots = np.concatenate((np.where(spots <= 5)[1], np.where(spots >= img.shape[0]-5)[1]))
    clean_spots = np.concatenate((np.expand_dims(np.delete(spots[0,:], bad_spots),1), 
                                  np.expand_dims(np.delete(spots[1,:], bad_spots),1)),
                                 axis = 1)
    return clean_spots


img = read_image(image_file_path)
spots = pd.read_csv(spots_file_path)
trace = pixel_t_trace(img, 39, 91, 3)

intensity_model = IntensityTrace(trace, 0.1)
scale = intensity_model.xy_normalize()

p_on_estimate, p_off_estimate = intensity_model.fit_grid_search(points=10, p_on_max=1e-2)
trace_params = ModelParams(p_on_estimate, p_off_estimate)
trace_model = TraceModel(trace_params, 0.1, len(intensity_model.trace))
on_intensity, on_noise, background_noise, p_on_estimate, p_off_estimate, states = \
                    trace_model.fit_viterbi(intensity_model.trace, 1)
        


                    
def image_fitting(image, spots):
    ''' 
    Args:
        Image: np_array
            a stack of images in shape (y, x, time)
        
        spots: np_array, list
            list of spots in format y, x
    '''
    spots = spots.astype('int')
    p_values = np.zeros((np.max(spots.shape), 4))
    for s in range(len(p_values)):
        trace = pixel_t_trace(image, spots[0,s], spots[1,s],3)
        intensity_model = IntensityTrace(trace, 0.1)
        scale = intensity_model.xy_normalize()
        p_on_grid, p_off_grid = intensity_model.fit_grid_search(points=10, p_on_max=1e-2)
        trace_params = ModelParams(p_on_estimate, p_off_estimate)
        trace_model = TraceModel(trace_params, 0.1, len(intensity_model.trace))
        on_intensity, on_noise, background_noise, p_on_vit, p_off_vit, states = \
                            trace_model.fit_viterbi(intensity_model.trace, 1)
                            
        p_values[s,:] = p_on_grid, p_off_grid, p_on_vit, p_off_vit
        
        print(p_values[s,:])
        
    return p_values
        

img = read_image(image_file_path)
spots = pd.read_csv(spots_file_path)
yx_spots = clean_spots(spots, img)
spot_num = 0
p_values = np.zeros((np.max(yx_spots.shape), 4))
#for spot_num in range(len(p_values)):
for spot_num in range(10):
    trace = pixel_t_trace(img, yx_spots[spot_num, 0], yx_spots[spot_num, 1],3)
    bs_trace = background_simple(img, trace, yx_spots[spot_num, 1],yx_spots[spot_num, 0])
    intensity_model = IntensityTrace(bs_trace, 0.1)
    scale = intensity_model.xy_normalize()

    p_on_grid, p_off_grid = intensity_model.fit_grid_search(points=10, p_on_max=1e-2)
    trace_params = ModelParams(p_on_grid, p_off_grid)
    trace_model = TraceModel(trace_params, 0.1, len(intensity_model.scaled_trace))
    on_intensity, on_noise, background_noise, p_on_vit, p_off_vit, states = \
                     trace_model.fit_viterbi(intensity_model.scaled_trace, 1)
    #plt.plot(states); plt.plot(intensity_model.scaled_trace)      
    print(spot_num)
    print(f'from grid_search {np.around(p_on_grid,4)} / {np.around(p_off_grid,4)}- - - - from viterbi {np.around(p_on_vit,4)} / {np.around(p_off_vit,4)}')
    p_values[spot_num,:] = p_on_grid, p_off_grid, p_on_vit, p_off_vit
    del trace, bs_trace, intensity_model, scale, p_on_grid, p_off_grid, \
        trace_params, trace_model,  on_intensity, on_noise, background_noise, \
            p_on_vit, p_off_vit, states
                    
                    
for spot_num in range(len(yx_spots)):
    trace = pixel_t_trace(img, yx_spots[spot_num, 0], yx_spots[spot_num, 1],3)
    plt.plot(trace)
    plt.figure()




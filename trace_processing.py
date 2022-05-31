import numpy as np
import skimage.io, scipy
import matplotlib.pyplot as plt
from pomegranate import *
import pandas as pd


file_path = '../../Images/0525_5nM_1.tif'
''' 39, 91 is a good sample trace'''

'''
goals:
    - estimate parameters needed to create model (p_on, p_off, μ, σ, σ_background)
    - σ_background: DONE -- fit mixture model and isolate background
    - μ, σ: will be linked once can determine distribution of "on" events
    - p_on, p_off:  1) acurately detect events and directly measure
                    2) fit all other paramters, then do grid search
'''

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

def plot_trace(trace):
    x = np.linspace(0,(len(trace)*0.1/60), num=len(trace))
    
    plt.plot(x,trace, 'k')
    plt.xlabel('time (min)')
    plt.ylabel('Intensity')


class exp_trace:
    
    def __init__(self, y, x, img):
        self.y = y
        self.x = x
        self.img = img
        return
    
    def pixel_t_trace(self, pxl_size):
        if pxl_size == 0:
            crop = self.img[self.y,self.x,:]
        else:
            crop = self.img[(self.y-pxl_size):(self.y+pxl_size), 
                       (self.x-pxl_size):(self.x+pxl_size),:]
        
            crop = np.max(crop, axis=0)
            crop = np.max(crop, axis=0)
        #trace = img[x,y,:]
        
        return crop
    
    def background_simple(self, trace, tile=7):
        baseline = np.zeros((len(trace)))
        a = self.y-tile
        if a < 0: 
            a=0
        b = self.y+tile
        if b > self.img.shape[0]:
            b = self.img.shape[0]
        c = self.x-tile
        if c < 0:
            c=0
        d = self.x+tile
        if d > self.img.shape[1]:
            d = self.img.shape[1]
            
        for i in range(len(trace)):
            baseline[i] = np.mean(self.img[a:b,c:d,i])
        
        return baseline
    
    def get_σ_bg(self, trace):
        X=np.expand_dims(np.ravel(trace),1)
        model = GeneralMixtureModel.from_samples([NormalDistribution, NormalDistribution], 2, X)
        
        noise_index = np.argmax(np.exp(model.weights))
        noise_params = model.distributions[noise_index].parameters
        
        #noise_model = NormalDistribution(noise_params[0], noise_params[1])
        
        return noise_params[0], noise_params[1]
    
        
        
        

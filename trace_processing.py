import numpy as np
import skimage.io, scipy
import matplotlib.pyplot as plt
from pomegranate import *
import pandas as pd
from fluorescence_model import FluorescenceModel, model_params
from sim_trace import intensityTrace


file_path = '../Images/0525_5nM_1.tif'
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
    
    def xy_norm(x_trace):
        X = np.expand_dims(np.ravel(x_trace),1)
        model = GeneralMixtureModel.from_samples([LogNormalDistribution, LogNormalDistribution], 2, X)
        peak_1 = model.distributions[0].parameters[0]
        peak_2 = model.distributions[1].parameters[0]
        scale = np.abs(np.exp(peak_1) - np.exp(peak_2))
        
        return scale
    
    def get_σ_bg(self, trace):
        X=np.expand_dims(np.ravel(trace),1)
        model = GeneralMixtureModel.from_samples([NormalDistribution, NormalDistribution], 2, X)
        
        noise_index = np.argmax(np.exp(model.weights))
        noise_params = model.distributions[noise_index].parameters
        
        #noise_model = NormalDistribution(noise_params[0], noise_params[1])
        
        return noise_params[0], noise_params[1]
    
  
    def emmison_m(K, trace, step):
        B = np.zeros((K,1))
        for state in range(K):
            B[state,0] = model.p_x_i_given_zi(trace[step], state)
        return B
      
    def viterbi(x_trace, A, Pi):
        K = A.shape[0]
        T = len(x_trace)
        T1 = np.zeros((K,T), 'd')
        T2 = np.zeros((K,T), 'B')
        
        for state in range(K):
            T1[state,0] = Pi[state] * model.p_x_i_given_z_i(x_trace[0], state)
        T2[:,0] = 0
        
        for i in range(1, T):           #observations
            for state in range(K):      #states
                return
        
    def viterbi_mu(y,t,delta, trans_m,s):
        temp = np.zeros((y+1))
        for i in range(y+1):
            temp[i] = delta[i,t-1] * trans_m[i,s]
        return np.max(temp), np.argmax(temp)
        
    def viterbi(x_trace, y, T, model, trans_m, p_init):
        "initialize"
        #y = 2
        #T = 100
        
        delta = np.zeros((y+1, T))
        sci = np.zeros((y+1, T))
        
        ''' initial values '''
        for s in range(y+1):
            delta[s,0] = p_init[s] * model.p_x_i_given_z_i(x_trace[0], s)
        sci[:,0] = 0
        
        ''' Propagation'''
        for t in range(1, T):
            for s in range(y+1):
                state_probs, ml_state = viterbi_mu(y,t,delta, trans_m,s)
                delta[s,t] =  state_probs * model.p_x_i_given_z_i(x_trace[t], s)
                sci[s,t] = ml_state
        
        ''' build to optimal model trajectory output'''
        x = np.zeros((T))
        x[-1] = np.argmax(delta[:, T-1])
        for i in reversed(range(1,T)):
            x[i-1] = sci[int(x[i]), i]
            
        return x, delta, sci
    
    def scale_viterbi(x_trace, y, T, model, trans_m, p_init):
        "initialize"
        delta = np.zeros((y+1, T))
        sci = np.zeros((y+1, T))
        scale = np.zeros((T))
        ''' initial values '''
        for s in range(y+1):
            delta[s,0] = p_init[s] * model.p_x_i_given_z_i(x_trace[0], s)
        sci[:,0] = 0
        
        ''' Propagation'''
        for t in range(1, T):
            for s in range(y+1):
                state_probs, ml_state = viterbi_mu(y,t,delta, trans_m,s)
                delta[s,t] =  state_probs * model.p_x_i_given_z_i(x_trace[t], s)
                sci[s,t] = ml_state
            scale[t] = 1 / np.sum(delta[:,t])
            delta[:,t] = delta[:,t] * scale[t]
        
        ''' build to optimal model trajectory output'''
        x = np.zeros((T))
        x[-1] = np.argmax(delta[:, T-1])
        for i in reversed(range(1,T)):
            x[i-1] = sci[int(x[i]), i]
            
        return x, delta, sci
            
        
        
        
y = 15
x_trace = sim_trace.gen_trace(y)       
p_init, trans_m = sim_trace.markov_trace(y)

x, delta, sci = scale_viterbi(x_trace, y, 3000, fmodel, trans_m, p_init)


''' full processing'''
img = read_image(file_path)
e_trace = exp_trace(39, 91, img)
trace = e_trace.pixel_t_trace(3)
sb_trace = trace - e_trace.background_simple(trace)
scale = xy_norm(trace)
norm_trace = sb_trace/scale + (1 - np.mean(sb_trace/scale))
params = model_params(0.01,0.1)
fmodel = FluorescenceModel(params)
sim_trace = intensityTrace(params, 0.1, 3000, fmodel)
p_init, trans_m = sim_trace.markov_trace(1)
x, delta, sci = scale_viterbi(norm_trace, 1, len(norm_trace), fmodel, trans_m, p_init)
plt.plot(x, c='r'); plt.plot(norm_trace, c='k')

        

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

import numpy as np
import matplotlib.pyplot as plt
from pomegranate import GeneralMixtureModel, LogNormalDistribution
from scipy.optimize import curve_fit
from trace_model import TraceModel
from fluorescence_model import ModelParams


params = ModelParams(0.02, 0.1, 1, 0.1, 0.2, 1)
trace_model = TraceModel(params, 0.1, 1000)
true_trace = trace_model.generate_trace(1)


def fit_fluorescence(trace):
    # returns log values
    X=np.expand_dims(np.ravel(true_trace),1)
    model = GeneralMixtureModel.from_samples([LogNormalDistribution, LogNormalDistribution], 2, X)
    
    background_noise = model.distributions[0].parameters[1]
    background_intensity = model.distributions[0].parameters[0]
    on_noise = model.distributions[1].parameters[1]
    on_intensity = model.distributions[1].parameters[0]
    
    return np.exp(on_intensity), on_noise, background_noise

def fit_viterbi(trace, y):
    
    # viterbi alg finds most likely state at each timepoint
    p_initial, transition_m = trace_model._markov_trace(y)
    states, delta, sci = scale_viterbi(true_trace, y, len(trace),
                                  trace_model.fluorescence_model, 
                                  transition_m, p_initial)
    
    # use state information to estimate fluorescence model parameters
    on_intensity = np.mean(trace[np.nonzero(states)])
    on_noise = np.std(trace[np.nonzero(states)])
    background_noise = np.std(trace[np.where(states == 0)])
    
    # state change information to estimate bright and dark times
    state_change = np.where(states[:-1] != states[1:])[0]
    bright_times = np.zeros(())
    dark_times = np.zeros(())
    on_off = np.split(states, state_change+1)
    for i in range(len(on_off)):
        event_length = len(on_off[i])
        if np.mean(on_off[i]) ==0:
           dark_times = np.append(dark_times, event_length)
        else:
           bright_times = np.append(bright_times, event_length)
    
    mean_bright = np.mean(bright_times) * trace_model.step_time
    mean_dark = np.mean(dark_times) * trace_model.step_time
    
    #build cdf to fit exponential distribution to data
    dark_cdf_x = np.sort(dark_times)
    dark_cdf_y = np.arange(len(dark_cdf_x))/len(dark_cdf_x)
    lam_dark, pcov_d = curve_fit(exp_cdf, dark_cdf_x, dark_cdf_y)
    
    bright_cdf_x = np.sort(bright_times)
    bright_cdf_y = np.arange(len(bright_cdf_x))/len(bright_cdf_x)
    lam_bright, pcov_d = curve_fit(exp_cdf, bright_cdf_x, bright_cdf_y)
    
    p_on_estimate = lam_dark * np.exp(-lam_dark * trace_model.step_time)
    p_off_estimate = lam_bright * np.exp(-lam_bright * trace_model.step_time)
     
    return on_intensity, on_noise, background_noise, \
        p_on_estimate, p_off_estimate

def fit_grid_search(trace, points=100, p_on_max=0.5, p_off_max=0.5):
    
    p_ons = np.linspace(1e-6, p_on_max, points)
    p_offs = np.linspace(1e-6, p_off_max, points)
    p_on_prob = np.zeros((len(p_ons)))
    p_off_prob = np.zeros((len(p_offs)))
    
    for i in range(len(p_ons)):
        test_params = ModelParams(p_ons[i], p_offs[50], 1, 0.1, 0.1, 1)
        test_model = TraceModel(test_params, 0.1, len(trace))
        p_on_prob[i] = test_model.p_trace_given_y(trace, 1)
    
    p_on_estimate = p_ons[np.argmax(p_on_prob)]
    for j in range(len(p_offs)):
        test_params = ModelParams(p_on_estimate, p_offs[j], 1, 0.1, 0.1, 1)
        test_model = TraceModel(test_params, 0.1, len(trace))
        p_off_prob[j] = test_model.p_trace_given_y(trace, 1)
    p_off_estimate = p_offs[np.argmax(p_off_prob)]
    
    return p_on_estimate, p_off_estimate

def fit_lbFCS(trace):
    
    return


def viterbi_mu(y,t,delta, trans_m,s):
    temp = np.zeros((y+1))
    for i in range(y+1):
        temp[i] = delta[i,t-1] * trans_m[i,s]
    return np.max(temp), np.argmax(temp)

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

def build_cdf(times):
    times_sort = np.sort(np.array(times))
    y_s     = np.arange(len(times_sort))/len(times_sort)
    
    plt.plot(times_sort, y_s)
    plt.ylabel('P(X >= x)')
    plt.figure()
    
    return times_sort, y_s


def exp_cdf(x, a):
    return 1 - np.exp(-a * x)


    













import logging
import numpy as np
import scipy
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd

from fluorescence_model import FluorescenceModel
from sim_trace import intensityTrace

def pred_y_sweep():
   model = FluorescenceModel(p_on=1, μ=1.0, σ=0.1, σ_background=0.1, q=0, )
   trace = intensityTrace(0.01, 0.5, 0.1, 1000, model)
   probs = np.zeros((20,20))
   for i in range(20):
       probs[:,i] = trace.demo_run(i)
   
   for i in range(1,20,2):
       plt.plot(np.log(probs[:,int(i)]))
       plt.scatter(i, np.max((np.log(probs[:,i]))))
   plt.xlabel('True number binding sites')
   plt.ylabel('log p( trace | y )')
   plt.figure()
   
   for i in range(1,20):
       plt.scatter(i, np.argmax((np.log(probs[:,i]))))
   plt.xlabel('TRUE number binding sites')
   plt.ylabel('PREDICTED number binding sites')
   
   
   p_ons = np.linspace(0,1,11)
   p_offs = np.linspace(0,1,11)
   rate_probs = np.zeros((25,11,11))
   c = 0
   for i in range(len(p_ons)):
       for j in range(len(p_offs)):
           c+=1
           trace = intensityTrace(p_ons[i], p_offs[j], 0.1, 100)
           rate_probs[:,i,j] = trace.demo_run(5)
           print(c)
   
   return

def effect_of_p_on(y, repeats):
    test_p_ons = np.logspace(-4,-1,50)
    test_ys = range(y-3, y+4)
    if y-3 < 0:
        test_ys = range(0, y+4)
    probs = np.zeros((len(test_ys), len(test_p_ons), repeats))
    
    for j in range(repeats):
        fluorescent_model = FluorescenceModel(p_on=1, μ=1.0, σ=0.1, σ_background=0.1, q=0, )
        true_trace = intensityTrace(0.003, 0.1, 0.1, 1000, fluorescent_model)
    
        x_trace = true_trace.gen_trace(y)
        fig, ax = plt.subplots(1,1)
        ax.plot(x_trace)
        fig.savefig('true_trace.png')
    
        
        for i in range(len(test_p_ons)):
            test_trace = intensityTrace(test_p_ons[i], 0.1, 0.1, 1000, fluorescent_model)
            y_pos = 0
            for y_test in test_ys:
                log_prob = test_trace.compare_runs(x_trace, y_test)
                print(log_prob)
                probs[y_pos,i,j] = log_prob
                y_pos += 1
        
            print(np.argmin(probs[:,i]))
        
    df = pd.DataFrame(probs)
    df.to_csv('test.csv')
    
    return




def test_job(y, p_on):
    fluorescent_model = FluorescenceModel(p_on=1, μ=1.0, σ=0.1, σ_background=0.1, q=0, )
    true_trace = intensityTrace(p_on, 0.1, 0.1, 1000, fluorescent_model)
    
    x_trace = true_trace.gen_trace(y)
    fig, ax = plt.subplots(1,1)
    ax.plot(x_trace)
    fig.savefig('true_trace.png')
    
    return
    
    
    
    
    
    
    
    
    
    
    
    
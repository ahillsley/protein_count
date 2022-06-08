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

def p_on_off_sweep(y, repeats):
    test_p_ons = np.linspace(0.001,0.999, 11)
    test_p_offs = np.linspace(0.001,0.999, 11)
    test_ys = range(y-2, y+3)
    if y-3 < 0:
        test_ys = range(0, y+3)
    probs = np.zeros((len(test_p_offs), len(test_p_ons),len(test_ys) ,repeats))
    fluorescent_model = FluorescenceModel(p_on=1, μ=1.0, σ=0.1, σ_background=0.1, q=0, )
    
    for q in range(repeats):
    
        for j in range(len(test_p_offs)):
            for i in range(len(test_p_ons)):
                test_trace = intensityTrace(test_p_ons[i], test_p_offs[j], 0.1, 1000, fluorescent_model)
                x_trace = test_trace.gen_trace(y)
                y_pos = 0
                for y_test in test_ys:
                    log_prob = test_trace.compare_runs(x_trace, y_test)
                    print(log_prob)
                    probs[j,i,y_pos,q] = log_prob
                    y_pos += 1
        
    #df = pd.DataFrame(probs)
    #df.to_csv('220603_p_on_var.csv')
    np.save('p_on_off_sweep.npy', probs)
    
    return


def fine_sweep_percent(p_on, p_off, y, repeats):
    test_p_ons = np.linspace((p_on-0.05*p_on), (p_on+0.05*p_on), 10)
    test_p_offs = np.linspace((p_off-0.05*p_off), (p_off+0.05*p_off), 10)
    
    for q in range(repeats):
        fluorescent_model = FluorescenceModel(p_on=1, μ=1.0, σ=0.1, σ_background=0.1, q=0, )
        true_trace = intensityTrace(p_on, p_off, 0.1, 1000, fluorescent_model)
        x_trace = true_trace.gen_trace(y)
        
        for j in range(len(test_p_offs)):
            for i in range(len(test_p_ons)):
                print('hello')
    return


def test_job(y, p_on):
    fluorescent_model = FluorescenceModel(p_on=1, μ=1.0, σ=0.1, σ_background=0.1, q=0, )
    true_trace = intensityTrace(p_on, 0.1, 0.1, 1000, fluorescent_model)
    
    x_trace = true_trace.gen_trace(y)
    fig, ax = plt.subplots(1,1)
    ax.plot(x_trace)
    fig.savefig('true_trace.png')
    
    return
    
    
    
    
    
    
    
    
    
    
    
    
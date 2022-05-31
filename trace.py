import logging
import numpy as np
import scipy
import scipy.stats as stats
import matplotlib.pyplot as plt
from fluorescence_model import FluorescenceModel


#model = FluorescenceModel()


class intensityTrace:
    '''
    
    '''
    
    
    def __init__(self, p_on, p_off, step_time, num_frames):
        
        self.p_on = p_on
        self.p_off = p_off
        self.step_time = step_time
        self.num_frames = num_frames
        
    
    def c_trans_matrix(self, y):
        ''' 
        - each entry represents the probability of going from state i to state j
        - is a stochastic matrix so each row should sum to 1
        '''
        size = y+1 # possible states range from 0 - y inclusive
        trans_m = np.zeros((size,size))
    
        for i in range(size):
            for j in range(size):
                p = 0
                for z in range(i+1):
                    p += stats.binom.pmf(z, i, self.p_off)* \
                        stats.binom.pmf(j-i+z, y-i, self.p_on)
                trans_m[i,j] = p
            
        return trans_m




    def markov_trace(self, y, limit=False):
    
        c_state = np.ones(y+1) / (y+1)
    
        trans_m = self.c_trans_matrix(y)
        num_steps = self.num_frames
        if limit == True:
            # If only care about steady state probabilities, no need to run full run
            num_steps = 10
    
        prob_trace = np.zeros((y+1, num_steps))
        for i in range(num_steps):
            c_state = c_state @ trans_m
            prob_trace[:,i] = c_state
        
        return prob_trace, trans_m
    

    def forward_pass(self, prob_trace, trans_m):
        p_init = prob_trace[:,-1]
        initial_state = list(stats.multinomial.rvs(1, p_init)).index(1)
        states = [initial_state]
        for i in range(self.num_frames-1):
            p_tr = trans_m[states[-1],:]
            new_state = list(stats.multinomial.rvs(1, p_tr)).index(1)
            states.append(new_state)
        return p_init, states


    def x_given_z_trace(self, states, model):
        #model = FluorescenceModel(p_on=1, μ=1.0, σ=0.1, σ_background=0.1, q=0)
        x_trace = np.zeros((len(states)))
        for i in range(len(states)):
            x_trace[i] = model.sample_x_given_y(y=[states[i]])
        return x_trace
 

    def alpha(self, forward, x_trace, trans_m, y, s, t, model):
        p = 0
        for i in range(y+1):
            p+= forward[i, (t-1)] * trans_m[i,s]*model.p_x_i_given_z_i(x_trace[t], s)
    
        return p   
  
  
    def forward_alg(self, x_trace, trans_m, y, p_init, model):
        forward = np.zeros((y+1, len(x_trace)))
        for s in range(y+1):
            forward[s,0] = p_init[s]*model.p_x_i_given_z_i(x_trace[0], s)
        for i in range(len(x_trace)-1):
            t = i+1
            for s in range(y+1):
                forward[s,t] = self.alpha(forward, x_trace, trans_m, y, s, t, model)
            
        fwrd_prob = np.sum(forward[:,-1])
    
        return fwrd_prob
    
    def gen_trace(self, y):
        fluorescent_model = FluorescenceModel(p_on=1, μ=1.0, σ=0.1, σ_background=0.1, q=0, )
        prob_trace, trans_m = self.markov_trace(y)
        p_init, states = self.forward_pass(prob_trace, trans_m)
        x_trace = self.x_given_z_trace(states, fluorescent_model)
        plt.plot(np.linspace(0,(len(x_trace)*self.step_time/60), num=len(x_trace)), x_trace)
        return x_trace
        
    def demo_run(self, y):
        fluorescent_model = FluorescenceModel(p_on=1, μ=1.0, σ=0.1, σ_background=0.1, q=0, )
        prob_trace, trans_m = self.markov_trace(y)
        p_init, states = self.forward_pass(prob_trace, trans_m)
        x_trace = self.x_given_z_trace(states, fluorescent_model)
        probs = np.zeros((20))
        for y in range(20):
            prob_trace_t, trans_m_t = self.markov_trace(y, limit=True)
            p_init_t = prob_trace_t[:,-1]
            fwrd_prob = self.forward_alg(x_trace, trans_m_t, y, p_init_t, fluorescent_model)
            probs[y] = fwrd_prob
            print(fwrd_prob)
        print(f'most likely {np.argmax(probs)} ')
        return probs

   

    def log_alpha(self, forward, x_trace, trans_m, y, s, t, model):
        p = 0
        for i in range(y+1):
            p+= np.exp(forward[i, (t-1)] + np.log(trans_m[i,s]) )
        
        alpha_bar = np.log(model.p_x_i_given_z_i(x_trace[t], s)) + np.log(p)
        return alpha_bar

    def forward_log(self, x_trace, trans_m, y, p_init, model):
        forward = np.zeros((y+1, len(x_trace)))
        for s in range(y+1):
            forward[s,0] = np.log(p_init[s])+ np.log(model.p_x_i_given_z_i(x_trace[0], s))
        for i in range(len(x_trace)-1):
            t = i+1
            for s in range(y+1):
                forward[s,t] = self.log_alpha(forward, x_trace, trans_m, y, s, t, model)
            
        fwrd_prob = np.sum(forward[:,-1])
    
        return fwrd_prob
    
    def single_run(self, y_true, y_test):
        fluorescent_model = FluorescenceModel(p_on=1, μ=1.0, σ=0.1, σ_background=0.1, q=0, )
        prob_trace, trans_m = self.markov_trace(y_true)
        p_init, states = self.forward_pass(prob_trace, trans_m)
        x_trace = self.x_given_z_trace(states, fluorescent_model)
        
        prob_trace_t, trans_m_t = self.markov_trace(y_test, limit=True)
        p_init_t = prob_trace_t[:,-1]
        fwrd_prob = self.forward_log(x_trace, trans_m_t, y_test, p_init_t, fluorescent_model)
        
        return fwrd_prob
        
    

def pred_y_sweep():
   trace = intensityTrace(0.5, 0.5, 0.1, 100)
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
           

     
        

            
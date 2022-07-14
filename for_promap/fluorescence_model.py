import numpy as np
from scipy import integrate, stats
import matplotlib.pyplot as plt
from pomegranate import GeneralMixtureModel, LogNormalDistribution


class ModelParams:
    def __init__(self, 
                 log_mean_intensity=1, 
                 sigma=0.1, 
                 mean_background=1, 
                 sigma_background=0.1,
                 label_eff=1):
        self.log_mean_intensity = log_mean_intensity
        self.sigma = sigma
        self.mean_background = mean_background
        self.sigma_background = sigma_background
        self.label_eff = label_eff


class FluorescenceModel:
    '''
    - Deals with the intensity measurements
    - The emmission probabilities of the hidden markov model

    Args:
        mu:
            the mean log intensity of a bound fluorophore

        sigma:
            the standard deviation of the log intensity of a bound fluorophore

        sigma_background:
            the standard deviation of the log intensity of the background,
                the mean is assumed to be 0

        label_eff:
            the labeling efficiency of y

    '''

    def __init__(self, model_params):

        self.log_mean_intensity = model_params.log_mean_intensity
        self.sigma = model_params.sigma
        self.sigma2 = model_params.sigma**2
        self.mean_background = model_params.mean_background
        self.sigma2_background = model_params.sigma_background**2
        self.sigma_background = model_params.sigma_background
        self.label_eff = model_params.label_eff



    
    def sample_x_i_given_z_i(self, z):
        
        mean = np.log(z * self.log_mean_intensity *np.exp(self.sigma2 / 2))
        
        signal = np.random.normal(mean, self.sigma)
        
        return self._bring_out(signal)
    
    
    def p_x_i_given_z_i(self, x_i, z):
        
        x = self._bring_in(x_i)
        
        mean = np.log(z * self.log_mean_intensity *np.exp(self.sigma2 / 2))
        
        result = integrate.romberg(lambda x: self._normal(x, mean, self.sigma2),
                                   x, x + (1/256))
        
        return result


    def _normal(self, x, mu, sigma2):
        
        return 1.0 / (np.sqrt(2.0 * np.pi * sigma2)) * \
                np.exp(-(x - mu)**2/(2.0 * sigma2))

    def _bring_in(self, x):
        return np.log(x)
        
    def _bring_out(self, x):
        return np.exp(x)


    
    def _approx_phi(self, x, mu, sigma):
        ''' 
        - method to approximate the integral value of p_x_i_given z_i
        -from Zelen & Severo approximation of the standard Normal CDF
         - Does not work for large differences in x and mu
         '''
        b_consts = [0.2316419, 0.319381530, -0.356563782, 1.781477937,
                       -1.821255978, 1.330274429 ]
        
        norm_pdf = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(1 / 2) * ((x - mu)/sigma) **2)
            
        t = 1 / (1 + b_consts[0] * x) 
        
        phi = 1 - norm_pdf * (b_consts[1] * t + b_consts[2] * t**2 + \
                              b_consts[3] * t**3 + b_consts[4] * t**4 + \
                              b_consts[5] * t**5)
            
        return phi
        
    def _integrate_from_cdf(self, x, mu, sigma):
        ''' 
        - method to approximate the integral value of p_x_i_given z_i
        - Does not work for large differences in x and mu with a small sigma
        - need to change to cdf
        '''
        a = self._normal(x, mu, sigma**2)
        b = self._normal(x+ (1/256), mu, sigma**2)
        
        prob = np.abs(a-b)
        return prob
        
    
    def _log_normal(self, x, mu, sigma2):

        return 1.0 / (x * np.sqrt(2.0 * np.pi * sigma2)) * \
            np.exp(-(np.log(x) - mu)**2/(2.0 * sigma2))
            
            



f_model = FluorescenceModel((ModelParams(1,.1)))
x = np.zeros(10000)
for i in range(len(x)):
    x[i] = f_model.sample_x_i_given_z_i(1)
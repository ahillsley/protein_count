import numpy as np
from scipy import integrate, stats
from pomegranate import GeneralMixtureModel, LogNormalDistribution


class ModelParams:
    def __init__(self, u=1, sigma=0.1, mean_background=1, sigma_background=0.1,
                 label_eff=1):
        self.u = u
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

        self.mu = model_params.u
        self.sigma = model_params.sigma
        self.sigma2 = model_params.sigma**2
        self.mean_background = model_params.mean_background
        self.sigma2_background = model_params.sigma_background**2
        self.sigma_background = model_params.sigma_background
        self.label_eff = model_params.label_eff


    def sample_x_i_given_z_i(self, z):
        '''
        sample x from lognormal distribution around z

        Args:

            z (int):
                the hidden state, or number of bound fluorophores

        '''

        
        if z == 0:
            signal = 0
        else:
            mu = np.log(self.mu * z)
            mean = np.exp(mu + self.sigma2 / 2)
            signal = np.random.lognormal(mean, self.sigma)
        
        background = np.random.lognormal(self.mean_background,
                                         self.sigma_background)

        return signal + background

    def p_x_i_given_z_i(self, x_i, z_i):
        '''
        compute p( x | z).

        Args:

            x_i (float):
                the observed flourescence intensity at time i

            z_i (int):
                the hidden state, or number of bound fluorophores at time i

        '''

        #mu = np.log(self.mu * z_i) # Eq 18,  DOI: Biophysj 106.101428
        
        if z_i == 0:
            mu = 0
            sigma2 = self.sigma2_background
        else:
            mu = np.log(self.mu * z_i)
            sigma2 = self.sigma2

        result = integrate.romberg(lambda x: self._log_normal(x_i, mu, sigma2),
                                x_i -1/256, x_i)
        
        return result

    def _log_normal(self, x, mu, sigma2):

        return 1.0 / (x * np.sqrt(2.0 * np.pi * sigma2)) * \
                np.exp(-(np.log(x) - mu)**2/(2.0 * sigma2))
                
    
    def approx_phi(self, x, mu, sigma):
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
        
    def integrate_from_cdf(self, x, mu, sigma):
        ''' 
        - method to approximate the integral value of p_x_i_given z_i
        - Does not work for large differences in x and mu with a small sigma
        '''
        a = stats.lognorm.pdf(x, sigma, mu)
        b = stats.lognorm.pdf(x + (1/256), sigma, mu)
        
        prob = a-b
        return prob
        
    

    

import numpy as np
from scipy import integrate, stats
import matplotlib.pyplot as plt


class ModelParams:
    def __init__(self,
                 mu_i=1,
                 sigma_i=0.1,
                 mu_b=1,
                 sigma_b=0.1,
                 label_eff=1):
        self.mu_i = mu_i
        self.sigma_i = sigma_i
        self.mu_b = mu_b
        self.sigma_b = sigma_b
        self.label_eff = label_eff


class FluorescenceModel:
    '''
    - Deals with the intensity measurements
    - The emmission probabilities of the hidden markov model

    Args:
        mu_i:
            the mean log intensity of a bound fluorophore

        sigma_i:
            the standard deviation of the log intensity of a bound fluorophore

        mu_b:
            the mean log intensity of athe background        

        sigma_b:
            the standard deviation of the log intensity of the background,
                the mean is assumed to be 0

        label_eff:
            the labeling efficiency of y

    '''

    def __init__(self, model_params):

        self.mu_i = model_params.mu_i
        self.sigma_i = model_params.sigma_i
        self.sigma_i2 = model_params.sigma_i**2
        self.mu_b = model_params.mu_b
        self.sigma_b = model_params.sigma_b
        self.sigma_b2= model_params.sigma_b**2
        self.label_eff = model_params.label_eff

    def sample_x_i_given_z_i(self, z):
        
        if z == 0:
            signal = -np.inf
        else:
            mean_i = np.log(z * self.mu_i * np.exp(self.sigma_i2 / 2))
            signal = np.random.normal(mean_i, self.sigma_i)
        
        mean_b = np.log(self.mu_b)
        background = np.random.normal(mean_b, self.sigma_b)

        # need to split into sperate exps because changing background should 
        # not change the estimate of mu_i
        return self._bring_out(signal) + self._bring_out(background)

    def p_x_i_given_z_i(self, x_i, z):

        x = self._bring_in(x_i)

        if z ==0:
            mean_i = -np.inf #
        else:
            mean_i = np.log(z * self.mu_i * np.exp(self.sigma_i2 / 2))
        
        mean_b = np.log(self.mu_b)

        mean = np.log(np.exp(mean_i) + np.exp(mean_b))
        sigma2 = self.sigma_i2 + self.sigma_b2

        result = integrate.romberg(lambda x: self._normal(x, mean,
                                    sigma2), x, x + (1/256))

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
                    -1.821255978, 1.330274429]

        norm_pdf = 1 / (sigma * np.sqrt(2 * np.pi)) * \
            np.exp(- (1 / 2) * ((x - mu) / sigma) ** 2)

        t = 1 / (1 + b_consts[0] * x)

        phi = 1 - norm_pdf * (b_consts[1] * t + b_consts[2] * t**2 +
                              b_consts[3] * t**3 + b_consts[4] * t**4 +
                              b_consts[5] * t**5)

        return phi

    def _integrate_from_cdf(self, x, mu, sigma):
        '''
        - method to approximate the integral value of p_x_i_given z_i
        - Does not work for large differences in x and mu with a small sigma
        - need to change to cdf
        '''
        a = self._normal(x, mu, sigma**2)
        b = self._normal(x + (1/256), mu, sigma**2)

        prob = np.abs(a - b)
        return prob

    def _log_normal(self, x, mu, sigma2):

        return 1.0 / (x * np.sqrt(2.0 * np.pi * sigma2)) * \
            np.exp(-(np.log(x) - mu)**2/(2.0 * sigma2))


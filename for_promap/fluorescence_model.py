import numpy as np
from scipy import integrate
from pomegranate import GeneralMixtureModel, LogNormalDistribution


class ModelParams:
    def __init__(self, u=1, sigma=0.1, sigma_background=0.1,
                 label_eff=1):
        self.u = u
        self.sigma = sigma
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
        self.sigma2_background = model_params.sigma_background**2
        self.label_eff = model_params.label_eff

    def fit_fluorescence(self, trace):
        '''
        fit all the parameters needed in fluorescence model
            mu, sigma, sigma_background
        '''
        X = np.expand_dims(np.ravel(trace), 1)
        model = GeneralMixtureModel.from_samples([LogNormalDistribution,
                                                  LogNormalDistribution], 2, X)
        sigma_background = model.distributions[0].parameters[1]
        sigma = model.distributions[1].parameters[1]
        mu = model.distributions[1].parameters[0]

        return mu, sigma, sigma_background

    def sample_x_i_given_z_i(self, z):
        '''
        sample x from lognormal distribution around z

        Args:

            z (int):
                the hidden state, or number of bound fluorophores

        '''

        mu = self.mu + np.log(z)

        mu = 0 if z == 0 else mu
        sigma = self.sigma2_background if z == 0 else self.sigma

        x = np.random.lognormal(mu, sigma)

        return x

    def p_x_i_given_z_i(self, x_i, z_i):
        '''
        compute p( x | z).

        Args:

            x_i (float):
                the observed flourescence intensity at time i

            z_i (int):
                the hidden state, or number of bound fluorophores at time i

        '''

        if z_i == 0:
            mu = 0
            sigma2 = self.sigma2_background
        else:
            mu = self.mu + np.log(z_i)
            sigma2 = self.sigma2

        result = integrate.romberg(lambda x: self._log_normal(x_i, mu, sigma2),
                                   x_i - self.sigma, x_i + self.sigma)
        return result

    def _log_normal(self, x, mu, sigma2):

        return 1.0 / (x * np.sqrt(2.0 * np.pi * sigma2)) * \
                np.exp(-(np.log(x) - mu)**2/(2.0 * sigma2))

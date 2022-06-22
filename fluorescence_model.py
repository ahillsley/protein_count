import logging
import numpy as np
import scipy.stats
from scipy import integrate  

logger = logging.getLogger(__name__)


class model_params:
    def __init__(self, p_on, p_off, u=1, σ=0.1, σ_background=0.1, q=0, label_eff=1):
        self.p_on = p_on
        self.p_off = p_off
        self.u = u
        self.σ = σ
        self.σ_background = σ_background
        self.q = q
        self.label_eff = label_eff

class FluorescenceModel:
    '''Simple fluorescence model for amino acid counts in proteins (y), dye
    activity (z), and fluorescence measurements per dye (x)::

        y_i ∈ ℕ (count of amino acid i)
        z_i ∈ ℕ (number of active dyes for amino acid i)
        x_i ∈ ℝ (total flourescence of active dyes for amino acid i)

        # independent per dye
        p(x|y) = Σ_i p(x_i|y_i)

        # fluorescence depends on dye activity z_i
        p(x_i|y_i) = Σ_z_i p(x_i|z_i)p(z_i|y_i)

        # dye activity is binomial
        p(z_i|y_i) ~ B(y_i, p_on)

        # flourescence follows log-normal distribution
        p(x*_i|z_i) = sqrt(2πσ_i²)^-1 exp[ -(x*_i - μ_i - ln z_i + q_z_i)² ]

    Args:

        p_on:
            Probability of a dye to be active.

        μ:
            Mean log intensity of a dye.

        σ:
            Standard deviation of log intensity of a dye.

        σ_background:
            Standard deviation of log intensity of background (mean is assumed
            to be 0).

        q:
            Dye-dye interaction factor (see Mutch et al., Biophusical Journal,
            2007, Deconvolving Single-Molecule Intensity Distributions for
            Quantitative Microscopy Measurements)
    '''

    def __init__(self, model_params):

        self.p_on = model_params.p_on
        self.μ = model_params.u
        self.σ = model_params.σ
        self.σ2 = model_params.σ**2
        self.σ2_background = model_params.σ_background**2
        self.q = model_params.q

    def p_x_given_y(self, x, y):
        '''Compute p(x|y).

        Args:

            x (ndarray, float, shape (n,)):

                Measured fluorescences per dye. -1 to indicate no measurement.

            y (ndarray, int, shape (n,) or (m, n)):

                Number of amino acids, congruent with x. If a 2D array is
                given, p(x|y) is computed for each row in y and an array of
                probabilities is returned.
        '''

        x = np.array(x, dtype=np.float64)
        y = np.array(y, dtype=np.int32)

        amino_acids = np.nonzero(x >= 0)[0]
        logger.debug("Found measurements for amino acids %s", amino_acids)

        p = np.ones((y.shape[0],)) if len(y.shape) == 2 else 1
        for i in amino_acids:

            p_x_i_given_y_i = np.zeros((y.shape[0],)) \
                    if len(y.shape) == 2 else 0

            y_i = y[:, i] if len(y.shape) == 2 else y[i]
            max_y_i = np.max(y_i)

            logger.debug(
                "Amino acid %s occurs at most %d times",
                i, max_y_i)

            for z_i in range(max_y_i + 1):
                p_x_i_given_y_i += \
                    self.p_x_i_given_z_i(x[i], z_i) * \
                    self.p_z_i_given_y_i(z_i, y_i)

            p *= p_x_i_given_y_i

        return p

    def sample_x_given_y(self, y, num_samples=1):
        '''Sample x ~ p(x|y).

        Args:

            y (ndarray, int, shape (n,) or (m, n)):

                Number of amino acids, congruent with x. If a 2D array is
                given, x ~ p(x|y) is sampled for each row ``num_samples``
                times.

        Returns:

            samples x as an ndarray of shape ``(num_samples, n)`` if ``y`` is
            1D, or ``(num_samples, m, n)`` if ``y`` is 2D.
        '''

        y = np.array(y)

        if num_samples > 1:
            size = (num_samples,) + y.shape
        else:
            size = y.shape

        z = np.random.binomial(y, self.p_on, size=size)

        μ = self.μ + np.log(z) - self.q
        σ = np.ones_like(μ)*self.σ

        μ[z == 0] = 0
        σ[z == 0] = self.σ2_background

        x = np.random.lognormal(μ, σ)

        return x

    def maximum_a_posterior(self, x, ys):
        '''Finds the maximum a posterior y* = argmax_y p(y|x) for x amongst the
        given ys.

        Args:

            x (ndarray, float, shape (n,)):

                Measurements for one protein.

            y (ndarray, int, shape (m, n)):

                Number of amino acids for ``m`` proteins.

        Returns:
            The index ``i`` into ``ys``, such that ``y* = ys[i]``.'''

        # p(y|x) = p(x|y)*p(y)/C ⇒ for uniform p(y), the posterior is
        # proportional to p(x|y)
        posterior = self.p_x_given_y(x, ys)

        return np.argmax(posterior)

    def p_x_i_given_z_i(self, x_i, z_i):
        ''' - not sure the proper range to integrate over
            - this is a minimal range, can probably increase
        '''
        if z_i == 0:
            μ = 0
            σ2 = self.σ2_background
        else:
            μ = self.μ + np.log(z_i) - self.q
            σ2 = self.σ2

        #return  self.log_normal(x_i, μ, σ2)
        
        result, uncer = integrate.quad(lambda x: self.log_normal(x_i, μ, σ2), 
                                       x_i-self.σ, x_i+self.σ)
        
        return result

    def log_normal(self, x, μ, σ2):
        ''' -returns the PDF of a lognormal distribution around μ
            - the PDF is likelyhoodm not probability and intregrates to 1
            - need to integrate over select range to get probability'''
        return \
            1.0 / (x * np.sqrt(2.0 * np.pi * σ2)) * \
            np.exp(-(np.log(x) - μ)**2/(2.0 * σ2))
            
            

    def p_z_i_given_y_i(self, z_i, y_i):
 
        return scipy.stats.binom.pmf(z_i, y_i, self.p_on)

            
            
            
            
            
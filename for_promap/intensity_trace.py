import numpy as np
from pomegranate import GeneralMixtureModel, NormalDistribution
from fluorescence_model import ModelParams
from trace_model import TraceModel


class IntensityTrace():
    '''
    Used to normalize and fit parameters to a trace of intensity over time

    Args:

        trace:
            1D np.array of intensity values

        step_time:
            the time

    '''
    def __init__(self, trace, step_time):
        self.trace = trace
        self.length = len(trace)
        self.scaled_trace = []

    def xy_normalize(self):

        X = np.expand_dims(np.ravel(self.trace), 1)
        model = GeneralMixtureModel.from_samples([NormalDistribution,
                                                  NormalDistribution], 2, X)

        background_peak = np.argmin([model.distributions[0].parameters[0],
                                     model.distributions[1].parameters[0]])
        background_distribution = model.distributions[background_peak]

        background_peak_mu = background_distribution.parameters[0]
        background_peak_sigma = background_distribution.parameters[1]

        threshold = background_peak_mu + 4 * background_peak_sigma
        trim_X = np.expand_dims(X[X > threshold], 1)

        model_trimmed = GeneralMixtureModel.from_samples([NormalDistribution,
                                                          NormalDistribution],
                                                         2, trim_X)
        signal_peaks = np.asarray([model_trimmed.distributions[0].parameters[0],
                                   model_trimmed.distributions[1].parameters[0]])
        signal_peak_mu = np.max(signal_peaks[~np.isnan(signal_peaks)])

        scale = 1.0/(signal_peak_mu - background_peak_mu)

        # scale, such that distance between peaks is 1
        self.scaled_trace = self.trace * scale

        # shift, such that mean background is 0
        self.scaled_trace -= background_peak_mu * scale

        # shift, such that mean background is 1
        self.scaled_trace += 1

        return scale

    def fit_grid_search(self, points=100, p_on_max=0.5, p_off_max=0.5):
        '''
        '''
        if self.scaled_trace == []:
            print('need to run xy_normalize first')
            return

        p_ons = np.linspace(1e-6, p_on_max, points)
        p_offs = np.linspace(1e-6, p_off_max, points)
        p_on_prob = np.zeros((len(p_ons)))
        p_off_prob = np.zeros((len(p_offs)))

        for i in range(len(p_ons)):
            test_params = ModelParams(p_ons[i], p_offs[int(points/2)])
            test_model = TraceModel(test_params, 0.1, len(self.scaled_trace))
            p_on_prob[i] = test_model.p_trace_given_y(self.scaled_trace, 1)

        p_on_estimate = p_ons[np.argmax(p_on_prob)]
        for j in range(len(p_offs)):
            test_params = ModelParams(p_on_estimate, p_offs[j], 1, 0.1, 0.1, 1)
            test_model = TraceModel(test_params, 0.1, len(self.scaled_trace))
            p_off_prob[j] = test_model.p_trace_given_y(self.scaled_trace, 1)
        p_off_estimate = p_offs[np.argmax(p_off_prob)]

        return p_on_estimate, p_off_estimate

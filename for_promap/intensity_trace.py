import numpy as np
from pomegranate import GeneralMixtureModel, NormalDistribution
from fluorescence_model import ModelParams
from trace_model import TraceModel


def normalize_trace(trace):

    X = np.expand_dims(np.ravel(trace), 1)
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
    normalized_trace = trace * scale

    # shift, such that mean background is 0
    normalized_trace -= background_peak_mu

    # shift, such that mean background is 1
    normalized_trace += 1

    return normalized_trace


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
        self.normalized_trace = normalize_trace(trace)

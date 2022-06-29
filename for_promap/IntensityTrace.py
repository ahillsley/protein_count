import numpy as np
from pomegranate import GeneralMixtureModel, LogNormalDistribution


class IntensityTrace():

    def __init__(self, trace, step_time):
        self.trace = trace
        self.length = len(trace)

    def xy_normalize(x_trace):

        X = np.expand_dims(np.ravel(x_trace), 1)
        model = GeneralMixtureModel.from_samples([LogNormalDistribution,
                                                  LogNormalDistribution], 2, X)
        peak_1 = model.distributions[0].parameters[0]
        peak_2 = model.distributions[1].parameters[0]
        scale = np.abs(np.exp(peak_1) - np.exp(peak_2))

        return scale

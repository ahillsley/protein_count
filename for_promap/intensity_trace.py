import numpy as np
from pomegranate import GeneralMixtureModel, NormalDistribution
from fluorescence_model import EmissionParams
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

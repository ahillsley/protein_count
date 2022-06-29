import numpy as np
import scipy.stats as stats
from FluorescenceModel import FluorescenceModel

class TraceModel:
    ''' Simple Model
    Args:

        p_on:
            the probability of a single strand of imager DNA binding to
            a single docker during time (step_time)

        p_off:
            the probability of a bound imager and docker strand
            dissociating during time (step_time)

        step_time (seconds):
            length of a single measurement, equivalent to the camera exposure
            time or 1 / camera frame rate

        num_frames:
            the number of frames in the trace, multiply by step time to get
            length of trace
    '''

    def __init__(self, model_params, step_time, num_frames):

        # currently working with p_on/off, might need to switch to k_on/off
        self.p_on = model_params.p_on
        self.p_off = model_params.p_off
        self.step_time = step_time
        self.num_frames = num_frames
        self.fluorescence_model = FluorescenceModel(model_params)

    def p_trace_given_y(self, trace, y):

        p_initial, transition_m = self._markov_trace(y)
        log_fwrd_prob = self._forward_alg(trace, y, transition_m, p_initial)

        return log_fwrd_prob

    def fit_params():
        '''
        Fit all the parameters needed for the trace model
            p_on, p_off
        '''

        return

    def generate_trace(self, y):

        p_initial, transition_m = self._markov_trace(y)
        # generate list of states
        initial_state = list(stats.multinomial.rvs(1, p_initial)).index(1)
        states = [initial_state]
        for i in range(self.num_frames-1):
            p_tr = transition_m[states[-1], :]
            new_state = list(stats.multinomial.rvs(1, p_tr)).index(1)
            states.append(new_state)

        # generate observations from list of states
        x_trace = np.ones((len(states)))
        for i in range(len(states)):
            x_trace[i] = self.fluorescence_model.sample_x_i_given_z_i(states[i])

        return x_trace

    def estimate_y(self, trace, guess, search_width):

        log_probs = np.zeros((search_width*2+1))
        low_bound = 0 if guess - search_width < 0 else guess - search_width
        for i, y in enumerate(range(low_bound, guess+search_width+1)):
            log_probs[i] = self.p_trace_given_y(trace, y)

        return log_probs

    def _markov_trace(self, y):
        c_state = np.ones(y+1) / (y+1)
        transition_m = self._create_transition_m(y)
        prob_trace = np.zeros((y+1, 10))
        for i in range(10):
            c_state = c_state @ transition_m
            prob_trace[:, i] = c_state
        p_initial = prob_trace[:, -1]

        # correct minor rounding errors
        if np.sum(p_initial) > 1.0:
            pos = np.argmax(p_initial[:])
            p_initial[pos] = p_initial[pos] - (np.sum(p_initial)-1)

        # check and fix minor rounding errors in transition matrix
        # -- maybe have np round down in _create_transition_m
        for i in range(transition_m.shape[1]):
            prob_total = np.sum(transition_m[i, :])
            if prob_total > 1.0:
                pos = np.argmax(transition_m[i, :])
                transition_m[i, pos] = transition_m[i, pos] - (prob_total-1)

        return p_initial, transition_m

    def _create_transition_m(self, y):

        size = y+1  # possible states range from 0 - y inclusive
        transition_m = np.zeros((size, size))

        for i in range(size):
            for j in range(size):
                p = 0
                for z in range(i+1):
                    p += stats.binom.pmf(z, i, self.p_off) * \
                        stats.binom.pmf(j-i+z, y-i, self.p_on)
                transition_m[i, j] = p

        return transition_m

    def _forward_alg(self, x_trace, y, transition_m, p_init):

        forward = np.zeros((y+1, len(x_trace)))
        scale_factors = np.zeros((len(x_trace)))

        # Initialize
        for s in range(y+1):
            forward[s, 0] = p_init[s] * \
                self.fluorescence_model.p_x_i_given_z_i(x_trace[0], s)
        scale_factors[0] = 1 / np.sum(forward[:, 0])
        forward[:, 0] = forward[:, 0] * scale_factors[0]

        # Propagate
        for t in range(1, len(x_trace)):
            for state in range(y+1):
                forward[state, t] = self._alpha(forward, x_trace, transition_m,
                                                y, state, t)

            scale_factors[t] = 1 / np.sum(forward[:, t])
            forward[:, t] = forward[:, t] * scale_factors[t]

        # Total likelyhood
        log_fwrd_prob = -1*(np.sum(np.log(scale_factors)))

        return log_fwrd_prob

    def _alpha(self, forward, x_trace, transition_m, y, s, t):
        # recursive element of the forward algorithm
        p = 0
        for i in range(y+1):
            p += forward[i, (t-1)] * transition_m[i, s] * \
                self.fluorescence_model.p_x_i_given_z_i(x_trace[t], s)

        return p

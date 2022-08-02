import numpy as np
import warnings
import scipy


def k_v_step_detect(trace, tolerance):
     '''
     -Bennett Kalafut, Koen Visscher,
     -An objective, model-independent method for detection of non-uniform steps in noisy signals,
     -Computer Physics Communications,
     -Volume 179, Issue 10,
     -2008,
     
     - could be used to estimate p_on and p_off
     '''
      
     # need to add while loop to keep adding steps
     k = 1  # number of steps added
     n = len(trace)
     step_points = [0, n]
     
     bic_s = np.zeros((n))
     bic_0 = 2 * np.log(n) + n * np.log(kv_sigma2_j(step_points, trace))
     bic_best = bic_0
     bic_loop = 0
     
     for i in range(100):
         for i in range(n):
             step_points_test = np.sort(np.append(step_points, i))
             sigma2 = kv_sigma2_j(step_points_test, trace)
             bic_s[i] = (k + 2) * np.log(n) + n * np.log(sigma2)
         ml_step_point = np.argmin(bic_s)
         
         bic_loop = np.min(bic_s)
         step_points = np.append(step_points, ml_step_point)
         
         if np.abs(bic_best - bic_loop) < tolerance:
             break
         else:
             bic_best = bic_loop
     
     step_points = step_points[:-1]
     steps = np.sort(step_points)
         
     return steps[1:-1]


def kv_sigma2_j(step_points, trace):
    sigma = 0
    for i in range(len(step_points)-1):
       
        indicies = np.linspace(step_points[i]+1, step_points[i+1], 
                               step_points[i+1]-step_points[i], dtype='int')
        mu_i = np.mean(trace[indicies[:-1]])
        
        for l in indicies[:-1]:
            sigma  += (trace[l] - mu_i)**2
    sigma = (1/len(trace)) * sigma
    
    return sigma
    
        
                    

def ibfcs_norm(trace):
    num_bins = 100
    y = np.histogram(trace, num_bins, (0,np.max(trace)))[0]
    x = np.arange(0, np.max(trace), np.max(trace)/num_bins)
    
    y2 = np.histogram(2 * trace, num_bins, (0,np.max(trace)))[0]
    y2 = y2 * 1.5
    
    return

def auto_correlate(trace):
    # from mark
    acov = np.zeros(len(trace))
    sample_avg = np.mean(trace)
    
    for k in range(len(acov)):
        ck = 0
        for j in range(len(trace) - k):
            ck += (trace[j] - sample_avg) * (trace[j+k] - sample_avg)
        acov[k] = ck/len(trace)
        
    return acov / acov[0]


def autocorrelate(a, m=16, deltat=1, normalize=False,
                  copy=True, dtype=None):
    """
    Autocorrelation of a 1-dimensional sequence on a log2-scale.
    This computes the correlation similar to
    :py:func:`numpy.correlate` for positive :math:`k` on a base 2
    logarithmic scale.
        :func:`numpy.correlate(a, a, mode="full")[len(a)-1:]`
        :math:`z_k = \Sigma_n a_n a_{n+k}`
    Parameters
    ----------
    a : array-like
        input sequence
    m : even integer
        defines the number of points on one level, must be an
        even integer
    deltat : float
        distance between bins
    normalize : bool
        True: normalize the result to the square of the average input
        signal and the factor :math:`M-k`.
        False: normalize the result to the factor :math:`M-k`.
    copy : bool
        copy input array, set to ``False`` to save memory
    dtype : object to be converted to a data type object
        The data type of the returned array and of the accumulator
        for the multiple-tau computation.
    Returns
    -------
    autocorrelation : ndarray of shape (N,2)
        the lag time (1st column) and the autocorrelation (2nd column).
    Notes
    -----
    .. versionchanged :: 0.1.6
       Compute the correlation for zero lag time.
    The algorithm computes the correlation with the convention of the
    curve decaying to zero.
    For experiments like e.g. fluorescence correlation spectroscopy,
    the signal can be normalized to :math:`M-k`
    by invoking ``normalize = True``.
    For normalizing according to the behavior
    of :py:func:`numpy.correlate`, use ``normalize = False``.
    Examples
    --------
    >>> from multipletau import autocorrelate
    >>> autocorrelate(range(42), m=2, dtype=np.float_)
    array([[  0.00000000e+00,   2.38210000e+04],
           [  1.00000000e+00,   2.29600000e+04],
           [  2.00000000e+00,   2.21000000e+04],
           [  4.00000000e+00,   2.03775000e+04],
           [  8.00000000e+00,   1.50612000e+04]])
    """
    assert isinstance(copy, bool)
    assert isinstance(normalize, bool)

    if dtype is None:
        dtype = np.dtype(a[0].__class__)
    else:
        dtype = np.dtype(dtype)


    if dtype.kind != "f":
        warnings.warn("Input dtype is not float; casting to np.float_!")
        dtype = np.dtype(np.float_)

    # If copy is false and dtype is the same as the input array,
    # then this line does not have an effect:
    trace = np.array(a, dtype=dtype, copy=copy)

    # Check parameters
    if m // 2 != m / 2:
        mold = m
        m = np.int_((m // 2 + 1) * 2)
        warnings.warn("Invalid value of m={}. Using m={} instead"
                      .format(mold, m))
    else:
        m = np.int_(m)

    N = N0 = trace.shape[0]

    # Find out the length of the correlation function.
    # The integer k defines how many times we can average over
    # two neighboring array elements in order to obtain an array of
    # length just larger than m.
    k = np.int_(np.floor(np.log2(N / m)))

    # In the base2 multiple-tau scheme, the length of the correlation
    # array is (only taking into account values that are computed from
    # traces that are just larger than m):
    lenG = m + k * (m // 2) + 1

    G = np.zeros((lenG, 2), dtype=dtype)

    normstat = np.zeros(lenG, dtype=dtype)
    normnump = np.zeros(lenG, dtype=dtype)

    traceavg = np.average(trace)

    # We use the fluctuation of the signal around the mean
    if normalize:
        # trace -= traceavg #Not interested in signal around mean, correlation should drop to 1 when normalized!!
        # assert traceavg != 0, "Cannot normalize: Average of `a` is zero!"
        if traceavg ==0: traceavg=1

    # Otherwise the following for-loop will fail:
    assert N >= 2 * m, "len(a) must be larger than 2m!"

    # Calculate autocorrelation function for first m+1 bins
    # Discrete convolution of m elements
    for n in range(0, m + 1):
        G[n, 0] = deltat * n
        # This is the computationally intensive step
        G[n, 1] = np.sum(trace[:N - n] * trace[n:])
        normstat[n] = N - n
        normnump[n] = N
    # Now that we calculated the first m elements of G, let us
    # go on with the next m/2 elements.
    # Check if len(trace) is even:
    if N % 2 == 1:
        N -= 1
    # Add up every second element
    trace = (trace[:N:2] + trace[1:N:2]) / 2
    N //= 2
    # Start iteration for each m/2 values
    for step in range(1, k + 1):
        # Get the next m/2 values via correlation of the trace
        for n in range(1, m // 2 + 1):
            npmd2 = n + m // 2
            idx = m + n + (step - 1) * m // 2
            if len(trace[:N - npmd2]) == 0:
                # This is a shortcut that stops the iteration once the
                # length of the trace is too small to compute a corre-
                # lation. The actual length of the correlation function
                # does not only depend on k - We also must be able to
                # perform the sum with respect to k for all elements.
                # For small N, the sum over zero elements would be
                # computed here.
                #
                # One could make this for-loop go up to maxval, where
                #   maxval1 = int(m/2)
                #   maxval2 = int(N-m/2-1)
                #   maxval = min(maxval1, maxval2)
                # However, we then would also need to find out which
                # element in G is the last element...
                G = G[:idx - 1]
                normstat = normstat[:idx - 1]
                normnump = normnump[:idx - 1]
                # Note that this break only breaks out of the current
                # for loop. However, we are already in the last loop
                # of the step-for-loop. That is because we calculated
                # k in advance.
                break
            else:
                G[idx, 0] = deltat * npmd2 * 2**step
                # This is the computationally intensive step
                G[idx, 1] = np.sum(trace[:N - npmd2] *
                                   trace[npmd2:])
                normstat[idx] = N - npmd2
                normnump[idx] = N
        # Check if len(trace) is even:
        if N % 2 == 1:
            N -= 1
        # Add up every second element
        trace = (trace[:N:2] + trace[1:N:2]) / 2
        N //= 2

    if normalize:
        G[:, 1] /= traceavg**2 * normstat
    else:
        G[:, 1] /= normstat

    return G
        
def fit_ac_lin(ac,max_it=10):
    ''' 
    Linearized iterative version of AC fit. 
    '''
    ###################################################### Define start parameters
    popt=np.empty([2]) # Init
    
    popt[0]=ac[1,1]-1                                               # Amplitude
    
    l_max=8                                                         # Maximum lagtime   
    try: l_max_nonan=np.where(np.isnan(-np.log(ac[1:,1]-1)))[0][0]  # First lagtime with NaN occurence
    except: l_max_nonan=len(ac)-1
    l_max=min(l_max,l_max_nonan)                                    # Finite value check
    
    popt[1]=(-np.log(ac[l_max,1]-1)+np.log(ac[1,1]-1))              # Correlation time tau corresponds to inverse of slope                          
    popt[1]/=(l_max-1)
    popt[1]=1/popt[1]
    
    ###################################################### Fit boundaries
    lowbounds=np.array([0,0])
    upbounds=np.array([np.inf,np.inf])
    
    ###################################################### Apply iterative fit
    if max_it==0: return popt[0],popt[1],l_max,0,np.nan
    
    else:
        popts=np.zeros((max_it,2))
        for i in range(max_it):
            l_max_return=l_max # Returned l_max corresponding to popt return
            try:
                ### Fit
                popts[i,:],pcov=scipy.optimize.curve_fit(ac_monoexp_lin,
                                                         ac[1:l_max,0],
                                                         -np.log(ac[1:l_max,1]-1),
                                                         popt,
                                                         bounds=(lowbounds,upbounds),
                                                         method='trf')
                
                ### Compare to previous fit result
                delta=np.max((popts[i,:]-popt)/popt)*100
                if delta<0.25: break
            
                ### Update for next iteration
                popt=popts[i,:]
                l_max=int(np.round(popt[1]*0.8))       # Optimum lagtime
                l_max=np.argmin(np.abs(ac[:,0]-l_max)) # Get lagtime closest to optimum (multitau!)
                l_max=max(3,l_max)                     # Make sure there are enough data points to fit
                l_max=min(l_max,l_max_nonan)           # Make sure there are no NaNs before maximum lagtime
                
            except:
                popt=np.ones(2)*np.nan
                delta=np.nan
                break
        
        return popt[0],popt[1],popts[0,0],popts[0,1],ac[l_max_return,0],i+1,
    

def ac_monoexp_lin(l,A,tau):
    '''
    Linearized fit function for mono-exponential fit of autocorrelation function: ``-log(g(l)-1)=l/tau-log(A)``
    '''
    g=l/tau-np.log(A)
    return g
    
    
def solve(Y, A, tau):
    
    k_off = 1 / (tau * (1 + 1 / (Y * A)))
    k_on = k_off / (Y * A)
    
    return k_off, k_on


def extract_eps(p):
    
    ### Get histogram of single photon counts
    y = np.histogram(p,800,(0,4000))[0]  # Only values returned in numba
    x = np.arange(2.5,4000,5)            # These are the bins
    
    ### Get histogram of double photon counts
    y2 = np.histogram(2*p,800,(0,4000))[0]  # Only values returned in numba
    y2 = y2 * 1.5                           # Multiply values to roughly equal heights!!!
    
    y = y.astype(np.float32)
    y2 = y2.astype(np.float32)
    ### Smooth substraction using lagtimes
    ### i.e. subtracting doubled photon histograms over and over by moving it to the right
    y_diff = y.copy()
    y2_lag = y2.copy()
    for l in range(0,100):
        y_diff -= y2_lag               # Substract y2_lag from y
        y2_lag = np.append(0,y2_lag)   # Add zero to start
        y2_lag = y2_lag[:-1]           # Remove last entry 
    
    y_diff[y_diff<0] = 0               # Asign zero to all negative entries after smooth substraction
    
    ### Calculate mean of y_diff
    eps_mean = np.sum(x * y_diff) / np.sum(y_diff)
    
    ### Calculate median of y_diff
    y_diff_cum = np.cumsum(y_diff / np.sum(y_diff))
    median_idx = np.argmin(np.abs(y_diff_cum-0.5))
    eps_median = x[median_idx]
    
    ### Which value is used for cut out of original distribution
    eps = (eps_median + eps_mean)/2
    
    ### Cut out first peak p based on eps
    in_first_peak = np.abs(p-eps) < 0.4 * eps
    eps = np.median(p[in_first_peak])
    
    return eps #,x, y, y2, y_diff
    
    
    
    
    
    
    
    
    
    
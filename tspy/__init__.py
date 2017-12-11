"""
Author:         Andrea Pasqualini
Institution:    Bocconi University
Created on:     December 25, 2016
Encoding:       UTF-8
"""
import numpy as np
from scipy import linalg as la
# from scipy import signal as sg
from scipy import stats as st

def lag(y, p=1, missing=None):
    """
    Given a matrix of time series 'y', this function computes its lagged
    version. The output will be of the same shape as 'y', so a 'missing' value
    for the virtually discarded rows can be provided.
    At its essence, this function is simply a wrapper for numpy.roll().

    Parameters
    ----------
    y : float, numpy.ndarray
         The set of time series to be lagged. Variables must be in columns,
         with rows representing the observations. It must be that y.shape =
         (T,N), where T is the number of observations and N is the number of
         variables.
    p : int (optional)
         If the input is y_t, the output will be y_{t-p}. If a lead is to be
         obtained instead of a lag, 'p' must be negative. Default value is '1',
         so that the function would return the first lagged value of 'y'.
    missing : float (optional)
              A scalar value to replace rows to be discarded. Default is
              numpy.NaN.

    Returns
    -------
    ylag : float, numpy.ndarray
           The lagged/led version of y, by 'p' lags/leads.
    """
    T = y.shape[0]  # no. of rows
    # N = y.shape[1]  # no. of cols

    if p > T or p < -(T+1):
        raise ValueError('Number of lags inconsistent with no. of observations')

    if missing is None:
        missing = np.nan

    ylag = np.roll(y, p, axis=0)
    if p > 0:
        ylag[:p, :] = missing  # NumPy's broadcasting is awesome!!
    elif p < 0:
        ylag[p:, :] = missing  # NumPy's broadcasting is awesome!!
    else:
        raise ValueError('p=0 is not a valid input value')
    return ylag


def lagcat(y, p, missing=None):
    """
    This function builds a matrix of regressors that would be typically
    employed in VAR or VECM models.
    At its essence, this function is a wrapper for myfuns.lag().
    WARNING: if 'y' is a matrix of integers, the 'missing' parameter behaves
             strangely.

    Parameters
    ----------
    y : float, numpy.ndarray
         A matrix with the time series of interest. It must satisfy
         y.shape = (T,N), where T is the number of observations in time and
         N is the number of variables.
    p : int
         The number of contiguous lags to be included in the final array.
         At the moment, it must be positive and greater or equal to 1.
         Hence this function currently supports only lagging and not leading.
    missing : float (optional)
              A scalar value to replace rows to be discarded. Default is
              numpy.NaN.

    Returns
    -------
    ylagcat : float, numpy.ndarray
              A matrix with the lagged variables as specified by the parameter
              'p'. It will satisfy ylagcat.shape = (T, N*p).
    """
    if p < 1:
        raise ValueError('Parameter p must be at least 1')

    T = y.shape[0]
    N = y.shape[1]
    ylagcat = np.empty((T, N*p), dtype=float)
    for j in range(p):
        ylagcat[:, (N*j):N*(j+1)] = lag(y, j+1, missing=missing)
    return ylagcat


def ols(y, x, const=True, everything=False):
    """
    Runs a Least-Squares regression of a vector y on a vector x.
    The regression equation is specified as follows in matrix notation:
    ---- LaTeX syntax ---------------------------------------------------------
        y = x \beta + \epsilon
    ---- End of LaTeX syntax --------------------------------------------------

    Parameters
    ----------
    y : numpy.array
         The vector on the LHS of the regression equation. It must satisfy
         y.shape = (T,1) or y.shape = (T,), where T is the number of
         observations available.
    x : numpy.array
         The matrix on the RHS of the regression equation. It must satisfy
         x.shape = (T,K), where T is the number of observations available and K
         is the number of regressors.
    const : bool (optional)
            Specifies whether a constant should be included in the regression
            equation. If so, then the constant will be the first value of the
            returned vector 'beta'. Default value is 'True'.
    everything : bool (optional)
                 Specifies whether the returned value of the function consists
                 only of the vector of estimated coefficients (as opposed to
                 the whole dictionary containing also diagnostics and other
                 information). Default value is 'False'.

    Returns
    -------
    beta : numpy.array
           The vector of estimated coefficients. This is returned if
           everything=False.
    results : dict
              A dictionary containing the following elements
                  - beta: the vector of estimated coefficients
                  - se: standard errors associated to beta
                  - resid: the vector of estimation errors
                  - fitted: the vector of fitted values
                  - conf_int: 95% confidence intervals
                  - tstat: t-statistics associated to beta
                  - r2: R-squared of the regression
                  - r2adj: adjusted R-squared of the regression
                  - T: the number of observations used in the regression
                  - K: the number of regressors
                  - TODO: complete documentation on returned values
              This is returned if everything=True.
    """
    T = y.shape[0]

    if x.shape[0] != T:
        raise ValueError('x and y have different no. of observations')

    if const:
        x = np.concatenate((np.ones((T, 1)), x), axis=1)

    K = x.shape[1]
    if T < 10000:       # prefer numerical accuracy over algorithm speed
        q, r = la.qr(x, mode='economic')
        inv_xx = la.solve(r.T * r, np.eye(K))  # (r'r) * a = I - solve for a
    else:               # prefer algorithm speed over numerical accuracy
        inv_xx = la.solve(x.T * x, np.eye(K))  # (x'x) * a = I - solve for a
    beta = np.dot(inv_xx, np.dot(x.T, y))

    if not everything:
        return beta         # provide the estimate beta to the user and exit
    else:
        yhat = np.dot(x, beta)                  # fitted values
        eps = y - yhat                          # regression residuals
        sigma = np.dot(eps.T, eps) / (T-K)      # var-cov matrix of the residuals
        std = np.sqrt(sigma * np.diag(inv_xx))  # standard deviation of beta
        tcrit = st.t.ppf(0.025, T)              # critical val on student-t for bidirectional testing
        conf_int = np.array([beta - tcrit * std,
                             beta + tcrit * std])  # 95% confidence interval
        tstat = beta / std                      # t-statistics
        rss = np.dot(eps.T, eps)                # residual sum of squared
        tss = np.dot((y - np.mean(y)).T, (y - np.mean(y)))  # total sum of squared
        r2 = 1 - rss/tss                        # R-squared
        arss = rss / (T-K)
        atss = tss / (T-1)
        if atss != 0:
            r2adj = 1 - (arss / atss)         # adjusted R-squared
        else:
            r2adj = None
        deps = eps[1:T] - eps[0:T-1]          # first difference of residuals
        dw = np.dot(deps.T, deps) / np.dot(eps.T, eps)  # Durbin-Watson stat
        results = {'beta': beta,
                   'se': std,
                   'fitted': yhat,
                   'resid': eps,
                   'sigma': sigma,
                   'conf_int': conf_int,
                   'tstat': tstat,
                   'r2': r2,
                   'r2adj': r2adj,
                   'dw': dw,
                   'meth': 'ols'}
        return results


def c_sja(n, p):  # needed for Johansen cointegration test
    jcp0 = [[2.9762,  4.1296,  6.9406],
            [9.4748, 11.2246, 15.0923],
            [15.7175, 17.7961, 22.2519],
            [21.8370, 24.1592, 29.0609],
            [27.9160, 30.4428, 35.7359],
            [33.9271, 36.6301, 42.2333],
            [39.9085, 42.7679, 48.6606],
            [45.8930, 48.8795, 55.0335],
            [51.8528, 54.9629, 61.3449],
            [57.7954, 61.0404, 67.6415],
            [63.7248, 67.0756, 73.8856],
            [69.6513, 73.0946, 80.0937]]
    jcp1 = [[2.7055,  3.8415,  6.6349],
            [12.2971, 14.2639, 18.5200],
            [18.8928, 21.1314, 25.8650],
            [25.1236, 27.5858, 32.7172],
            [31.2379, 33.8777, 39.3693],
            [37.2786, 40.0763, 45.8662],
            [43.2947, 46.2299, 52.3069],
            [49.2855, 52.3622, 58.6634],
            [55.2412, 58.4332, 64.9960],
            [61.2041, 64.5040, 71.2525],
            [67.1307, 70.5392, 77.4877],
            [73.0563, 76.5734, 83.7105]]
    jcp2 = [[2.7055,  3.8415,  6.6349],
            [15.0006, 17.1481, 21.7465],
            [21.8731, 24.2522, 29.2631],
            [28.2398, 30.8151, 36.1930],
            [34.4202, 37.1646, 42.8612],
            [40.5244, 43.4183, 49.4095],
            [46.5583, 49.5875, 55.8171],
            [52.5858, 55.7302, 62.1741],
            [58.5316, 61.8051, 68.5030],
            [64.5292, 67.9040, 74.7434],
            [70.4630, 73.9355, 81.0678],
            [76.4081, 79.9878, 87.2395]]
    if p > 1 or p < -1:
        jc = np.zeros((1, 3))
    elif n > 12 or n < 1:
        jc = np.zeros((1, 3))
    elif p == -1:
        jc = jcp0[n-1, :]
    elif p == 0:
        jc = jcp1[n-1, :]
    elif p == 1:
        jc = jcp2[n-1, :]
    else:
        raise ValueError('Something went wrong with c_sja()')
    return jc


def c_sjt(n, p):  # needed for Johansen cointegration test
    jcp0 = [[2.9762,   4.1296,   6.9406],
            [10.4741,  12.3212,  16.3640],
            [21.7781,  24.2761,  29.5147],
            [37.0339,  40.1749,  46.5716],
            [56.2839,  60.0627,  67.6367],
            [79.5329,  83.9383,  92.7136],
            [106.7351, 111.7797, 121.7375],
            [137.9954, 143.6691, 154.7977],
            [173.2292, 179.5199, 191.8122],
            [212.4721, 219.4051, 232.8291],
            [255.6732, 263.2603, 277.9962],
            [302.9054, 311.1288, 326.9716]]
    jcp1 = [[2.7055,   3.8415,   6.6349],
            [13.4294,  15.4943,  19.9349],
            [27.0669,  29.7961,  35.4628],
            [44.4929,  47.8545,  54.6815],
            [65.8202,  69.8189,  77.8202],
            [91.1090,  95.7542,  104.9637],
            [120.3673, 125.6185, 135.9825],
            [153.6341, 159.5290, 171.0905],
            [190.8714, 197.3772, 210.0366],
            [232.1030, 239.2468, 253.2526],
            [277.3740, 285.1402, 300.2821],
            [326.5354, 334.9795, 351.2150]]
    jcp2 = [[2.7055,   3.8415,   6.6349],
            [16.1619,  18.3985,  23.1485],
            [32.0645,  35.0116,  41.0815],
            [51.6492,  55.2459,  62.5202],
            [75.1027,  79.3422,  87.7748],
            [102.4674, 107.3429, 116.9829],
            [133.7852, 139.2780, 150.0778],
            [169.0618, 175.1584, 187.1891],
            [208.3582, 215.1268, 228.2226],
            [251.6293, 259.0267, 273.3838],
            [298.8836, 306.8988, 322.4264],
            [350.1125, 358.7190, 375.3203]]
    if p > 1 or p < -1:
        jc = np.zeros((1, 3))
    elif n > 12 or n < 1:
        jc = np.zeros((1, 3))
    elif p == -1:
        jc = jcp0[n-1, :]
    elif p == 0:
        jc = jcp1[n-1, :]
    elif p == 1:
        jc = jcp2[n-1, :]
    else:
        raise ValueError('Something went wrong with c_sjt()')
    return jc


# TODO: Johansen cointegration test
# TODO: refactor aux functions 'c_sja' and 'c_sjt' into other module


def spectrum(y, num=None, smooth=False, smooth_window=None, smooth_method='flat'):
    """
    This function computes the spectrum of the time series X evaluated at the
    frequencies freq_grid.

    For an explanation of what the code does in detail, see Appendix D of
    [BGP].
    The notation in this code follows the one in the paper.

    Parameters
    ----------
    y : float, numpy.ndarray
         The (real-valued) time series of interest. It must be a vector that
         satisfies y.shape = (T,) or y.shape(T,1).
    num : int
          Specifies the whole series length and coincides to len(y)
          after padding it with zeros. The default length is 1024 (FFT
          algorithms work better with series lengths that are powers of 2).
          It is passed to input parameter 'n' in the function numpy.fft.rfft().
    smooth : boolean
             takes a kernel-weighted average around each frequency, with
             window width 'smooth_window'. Such smoothing makes the Schuster's
             Periodogram consistent.
    smooth_window : int
                    Specifies the bandwidth of the kernel averaging W. Must be
                    odd and must be specified if 'smooth=True'.
    smooth_method : {'flat', 'hamming', 'hanning', 'bartlett', 'blackman'}
                    Has an effect only if 'smooth=True'. If smooth_method is
                    'flat', then the rolling average uses flat weights (i.e., it
                    becomes an arithmetic average as opposed to a weighted
                    average).

    References
    ----------
    .. [BGP] Beaudry, P., Galizia, D. and Portier, F. (2016), "Putting the
             Cycle back into Business Cycle Analysis."
    """
    if smooth:
        if smooth_window % 2 == 0:
            raise ValueError('\'smooth_window\' must be an odd integer.')

    if y.ndim > 2:
        raise ValueError('Input data is not a vector-like array.')
    elif y.ndim == 2:
        x = y[~np.isnan(y)].reshape((-1,))
    elif y.ndim == 1:
        x = y[~np.isnan(y)]

    if num is not None:
        N = num
    else:
        N = 1024
    T = len(x)

    # Computing the DFT of the time series and normalizing by length of series
    X = np.fft.rfft(x, n=N) / np.sqrt(T)
    # Computing the power spectrum
    S = (np.abs(X) ** 2) / (2*np.pi)  # [*] TODO: figure out this one
    Ns = len(S)
    # Note: the result is in the domain [0,pi], defined on equi-spaced points.
    #       Our relevant grid can be obtained by using
    #       np.fft.rfftfreq(T, d=1/(2*np.pi))
    # Note: [BGP] might be wrong, we're sampling T/2+1 points in the interval
    #       [0,pi] and not T points! EDIT: well, we're sampling T
    #       non-necessarily-distinct points. Then we find out N/2 coincide.
    # [*] I discovered I get the correct thing by dividing by 2pi, but I don't
    #     understand why. I must look this up.

    # Does the user want to smooth the periodogram?
    if not smooth:  # if not
        return S  # then return the periodogram and exit the function
    else:  # if yes
        W = smooth_window  # length of the Hamming window (must be odd!)
        if smooth_method == 'flat':
            weights = np.ones((W,))
        elif smooth_method == 'hamming':
            weights = np.hamming(W)
        elif smooth_method == 'hanning':
            weights = np.hanning(W)
        elif smooth_method == 'bartlett':
            weights = np.bartlett(W)
        elif smooth_method == 'blackman':
            weights = np.blackman(W)
        Sw = np.zeros((Ns,))
        for k in range(Ns):
            lr = int((W - 1) / 2)
            if k < lr:  # if there are few points on the left
                Sw[k] = (np.dot(S[: k + lr + 1], weights[lr - k:]) /
                         np.sum(weights[lr - k:]))
            elif k > (Ns - 1) - lr:  # if there are few points on the right
                Sw[k] = (np.dot(S[k - lr:], weights[: -(k - (Ns - 1) + lr)]) /
                         np.sum(weights[: -(k - (Ns - 1) + lr)]))
            else:
                Sw[k] = np.dot(S[k - lr: k + lr + 1], weights) / np.sum(weights)
        return Sw  # then return the smoothed periodogram and exit


def bandpass(y, K, lower=None, upper=None, strip_bias=False, return_nans=False):
    """
    This function implements a bilateral, symmetric band-pass filter.
    The returned object is the same time-series as in input, filtered at the
    specified frequencies. Either 'lower' or 'upper' or both must be provided.

    Parameters
    ----------
    y : float, numpy.ndarray
         The time-series of interest. It must satisfy y.shape = (T,) or
         y.shape = (T,1), where T is the length of the time-series.
    K : int
         The one-directional length of the moving average. Since the filter is
         bilateral and symmetric, the total width of the average will be 2K+1
    lower : float or None
            Specifies the lower bound on the frequencies to let pass. If 'lower'
            is given and upper=None, then a high-pass filter is implemented.
    upper : float or None
            Specifies the upper bound on the frequencies to let pass. If 'upper'
            is given and lower=None, then a low-pass filter is implemented.
    strip_bias : bool
                 Since the filter is symmetric and bilateral, the first K and
                 the last K filtered observations will be biased and potentially
                 meaningless. If strip_bias=True, then the first K and the last
                 K filtered observations will be replaced by NaNs.
                 Default is False.
    return_nans : bool
                  If the input vector 'y' contains some NaN values, this
                  parameter specifies whether the output should be of the same
                  length as the input, so that the output will have NaNs in the
                  same positions as the input vector 'y'.
                  Default is False.

    References
    ----------
    .. [BK] Baxter, M. and King, R. (1999), "Measuring Business Cycles:
            Approximate Band-Pass Filters for Economic Time Series." The Review
            of Economics and Statistics. 81(4), 575-593.
    """

    def lowpass(K, threshold):
        """
        This function computes the weights of a low-pass filter as in [BK].
        """
        h = np.arange(start=0, stop=K, step=1, dtype=int) + 1
        tmp = np.sin(h * threshold) / (h * np.pi)
        tmp = np.append(np.flipud(tmp), tmp)
        # WARNING: with Numpy >= 1.12, np.flipud must be replaced with np.flip
        weights = np.insert(tmp, int(len(tmp) / 2), threshold / np.pi)
        return weights

    def highpass(K, threshold):
        """
        This function computes the weights of a high-pass filter as in [BK].
        """
        h = np.arange(start=0, stop=K, step=1, dtype=int) + 1
        tmp = - (np.sin(h * threshold) / (h * np.pi))
        tmp = np.append(np.flipud(tmp), tmp)
        # WARNING: with Numpy >= 1.12, np.flipud must be replaced with np.flip
        weights = np.insert(tmp, int(len(tmp) / 2), 1 - threshold / np.pi)
        return weights

    if y.ndim > 2:
        raise ValueError('Input data is not a vector-like array.')
    elif y.ndim == 2:
        x = y[~np.isnan(y)].reshape((-1,))
    elif y.ndim == 1:
        x = y[~np.isnan(y)]
    T = len(x)

    if lower is not None and upper is not None:
        # generic band-pass filter
        bhu = lowpass(K, upper)
        bhl = lowpass(K, lower)
        weights = bhu - bhl
    elif lower is not None and upper is None:
        # high-pass filter
        weights = highpass(K, lower)
    elif lower is None and upper is not None:
        # low-pass filter
        weights = lowpass(K, upper)
    else:
        raise ValueError(
            'Neither \'lower\' nor \'upper\' parameters were provided.')

    z = np.zeros((T,))  # preallocating the vector of results
    for t in range(T):
        if t < K:  # if there are not enough points on the left
            if strip_bias:
                z[t] = np.nan
            else:
                z[t] = np.dot(x[: t + K + 1], weights[K - t:])
        elif t >= (T - 1) - K:  # if there are not enough points on the right
            if strip_bias:
                z[t] = np.nan
            else:
                z[t] = np.dot(x[t - K + 1:], weights[: -((t + 1) - (T - 1) + K)])
        else:  # if there are enough points around
            z[t] = np.dot(x[t - K: t + K + 1], weights)
    if return_nans:
        Y = y.reshape(-1,)
        z_nan = np.zeros((len(Y)), dtype=float)
        z_nan[np.isnan(Y)] = np.nan
        z_nan[~np.isnan(Y)] = z
        return z_nan
    else:
        return z

import numpy as np
from scipy.fft import rfft, dct
import pywt
from sklearn.metrics import pairwise_distances, pairwise_kernels


def dft2(x, y):
    return np.sqrt(np.sum(np.abs(rfft(x) - rfft(y))**2))


def _dct(x, y):
    return np.sum(np.abs(dct(x) - dct(y)))


def _dwt(x, y, wavelet):
    """
    Manhattan distance between discrete wavelet transforms

    Inputs:
    wavelet: a built-in wavelet from https://pywavelets.readthedocs.io/en/latest/ref/wavelets.html#built-in-wavelets-wavelist
    """
    return np.sum(np.abs(pywt.dwt(x, wavelet)[0] - pywt.dwt(y, wavelet)[0]))


def cos_exp_kernel(x, y, n_freqs=5, l=1):
    """
    The c-exp kernel

    Inputs:
    x, y: (n_samples, 1) input vectors
    n_freqs: number of frequencies to include in the sum
    l: bandwidth of the kernel

    Returns:
    Kernel values given x, y
    """

    cos_term = np.sum([np.cos(2*np.pi*n*(x-y)) for n in range(n_freqs)])
    return cos_term*np.exp(-(0.5/(l**2))*(x-y)**2)


def CEXP(X, n_freqs=20, l=10):
    """
    Transforms an array of function values using the integral operator induced by the cos-exp kernel.
    The function values are assumed to be on [0,1]

    Inputs:
    X: (n_samples, n_obs) array of function values
    n_freqs: number of frequencies to include in the sum
    l: bandwidth of the kernel

    Returns:
    cos_exp_X: (n_samples, n_obs) array of function values where each function has been passed
                through the integral operator induced by the cos-exp kernel
    """
    n_obs = X.shape[1]
    obs_grid = np.linspace(0, 1, n_obs)
    T_mat = pairwise_kernels(obs_grid.reshape(-1, 1), metric=cos_exp_kernel, n_freqs=n_freqs, l=l)
    cos_exp_X = (1./n_obs) * np.dot(X, T_mat)
    return cos_exp_X


# median heuristic for kernel width
def width(X, Y=None, metric='euclidean'):
    """
    Computes the median heuristic for the kernel bandwidth
    """
    if Y is None:
        Y = X

    dist_mat = pairwise_distances(X, Y, metric=metric)
    width_XY = np.median(dist_mat[dist_mat > 0])
    return width_XY


def K_ID(X, Y=None):
    """
    Forms the kernel matrix K using the SE-T kernel with bandwidth gamma
    where T is the identity operator

    Inputs:
    X, Y: (n_samples, n_obs) array of observed functional samples from the distribution of X, Y

    Returns:
    K: matrix formed from the kernel values of all pairs of samples from the two distributions
    """
    if Y is None:
        Y = X
    if len(X.shape) == 1:
        X = np.reshape(X, [-1, 1])
    if len(Y.shape) == 1:
        Y = np.reshape(Y, [-1, 1])

    dist_mat = pairwise_distances(X, Y, metric='euclidean')
    gamma = width(X)
    K = np.exp(-dist_mat**2/(2*gamma**2))
    return K


def K_dft2(X, Y=None):
    """
    Forms the kernel matrix K using the SE-T kernel with bandwidth gamma
    equal to the median of distances between the discrete Fourier transforms of the
    functional samples and where T is the Fourier distance

    Inputs:
    X, Y: (n_samples, n_obs) array of observed functional samples from the distribution of X, Y

    Returns:
    K: matrix formed from the kernel values of all pairs of samples from the two distributions
    """
    if Y is None:
        Y = X
    if len(X.shape) == 1:
        X = np.reshape(X, [-1, 1])
    if len(Y.shape) == 1:
        Y = np.reshape(Y, [-1, 1])

    dist_mat = pairwise_distances(X, Y, metric=dft2)
    gamma = width(X, metric=dft2)

    K = np.exp(-dist_mat**2/(2*gamma**2))
    return K


def K_dct(X, Y=None):
    """
    Forms the kernel matrix K using the Fourier-exponential kernel with bandwidth gamma
    equal to the median of distances between the discrete cosine transforms of the
    functional samples

    Inputs:
    X, Y: (n_samples, n_obs) array of observed functional samples from the distribution of X, Y

    Returns:
    K: matrix formed from the kernel values of all pairs of samples from the two distributions
    """
    if Y is None:
        Y = X
    if len(X.shape) == 1:
        X = np.reshape(X, [-1, 1])
    if len(Y.shape) == 1:
        Y = np.reshape(Y, [-1, 1])

    dist_mat = pairwise_distances(X, Y, metric=_dct)
    gamma = width(X)

    K = np.exp(-dist_mat**2/(2*gamma**2))
    return K


def K_dwt(X, Y=None, wavelet='coif2'):
    """
    Forms the kernel matrix K using the Fourier-exponential kernel with bandwidth gamma
    equal to the median of distances between the discrete cosine transforms of the
    functional samples

    Inputs:
    X, Y: (n_samples, n_obs) array of observed functional samples from the distribution of X, Y
    wavelet: a built-in wavelet from https://pywavelets.readthedocs.io/en/latest/ref/wavelets.html#built-in-wavelets-wavelist,
             default: 'coif2'

    Returns:
    K: matrix formed from the kernel values of all pairs of samples from the two distributions
    """
    if Y is None:
        Y = X
    if len(X.shape) == 1:
        X = np.reshape(X, [-1, 1])
    if len(Y.shape) == 1:
        Y = np.reshape(Y, [-1, 1])

    dist_mat = pairwise_distances(X, Y, metric=_dwt, wavelet=wavelet)
    gamma = width(X)

    K = np.exp(-dist_mat**2/(2*gamma**2))
    return K


def K_CEXP(X, Y=None):
    """
    Forms the kernel matrix K using the SE-T kernel with bandwidth gamma
    where T is the cosine-exponential operator

    Inputs:
    X, Y: (n_samples, n_obs) array of observed functional samples from the distribution of X, Y

    Returns:
    K: matrix formed from the kernel values of all pairs of samples from the two distributions
    """
    if Y is None:
        Y = X
    if len(X.shape) == 1:
        X = np.reshape(X, [-1, 1])
    if len(Y.shape) == 1:
        Y = np.reshape(Y, [-1, 1])

    X = CEXP(X)
    Y = CEXP(Y)

    dist_mat = pairwise_distances(X, Y, metric='euclidean')
    gamma = width(X)

    K = np.exp(-dist_mat ** 2 / (2 * gamma ** 2))
    return K

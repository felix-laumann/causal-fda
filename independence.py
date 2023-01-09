import numpy as np
from scipy.stats import percentileofscore
from kernels import K_ID, K_CEXP, K_dct, K_dft, K_dft1, K_dft2, K_dwt
from numba import njit
from tqdm.notebook import tqdm
from multiprocessing import cpu_count, get_context


# UTILS
def reshape(array, d):
    for x in array:
        yield x.reshape(-1, d)


# MARGINAL INDEPENDENCE TEST
def HSIC(K, L, n_samples, biased):

    if biased is True:
        # centering matrix...
        H = np.eye(n_samples) - (1 / n_samples) * (np.ones((n_samples, n_samples)))
        # ...to center K
        K_c = H @ K @ H
        statistic = 1 / (n_samples ** 2) * np.sum(np.multiply(K_c.T, L))

    else:
        np.fill_diagonal(K, 0)
        np.fill_diagonal(L, 0)
        KL = np.dot(K, L)
        statistic = np.trace(KL)/(n_samples*(n_samples-3)) + \
                    np.sum(K)*np.sum(L)/(n_samples*(n_samples-3)*(n_samples-1)*(n_samples-2)) - \
                    2*np.sum(KL)/(n_samples*(n_samples-3)*(n_samples-2))

    return statistic


def marginal_null_dist(K, L, n_samples, n_perms, biased):
    """
    Approximates the null distribution by permuting L, the kernel matrix of Y
    """
    # initialising HSIC
    HSIC_arr = np.zeros(n_perms)

    # create permutations by reshuffling L except the main diagonal
    for perm in range(n_perms):
        index_perm = np.random.permutation(L.shape[0])
        L_perm = L[index_perm, index_perm[:, None]]

        if biased is True:
            # centering matrix...
            H = np.eye(n_samples) - (1 / n_samples) * (np.ones((n_samples, n_samples)))
            # ...to center K
            K_c = H @ K @ H
            HSIC_arr[perm] = 1 / (n_samples ** 2) * np.sum(np.multiply(K_c.T, L_perm))
        else:
            HSIC_arr[perm] = np.trace(np.dot(K, L_perm))/(n_samples*(n_samples-3)) + np.sum(K)*\
                             np.sum(L_perm)/(n_samples*(n_samples-3)*(n_samples-1)*(n_samples-2)) - \
                             2*np.sum(np.dot(K, L_perm))/(n_samples*(n_samples-3)*(n_samples-2))

    statistics = np.sort(HSIC_arr)

    return np.sort(statistics)


def marginal_indep_test(X, Y, n_perms, alpha, make_K, biased):
    """
    Performs a marginal independence test with HSIC as a test statistic
    and returns an 'accept' (0) or 'reject' (1) statement for the null hypothesis of marginal independence

    Inputs:
    X: (n_samples, n_obs) array of function values of first distribution
    Y: (n_samples, n_obs) array of function values of second distribution
    n_perms: number of permutations performed when bootstrapping the null distribution
    alpha: rejection threshold of the test
    make_K: function called to construct the kernel matrix

    Returns:
    reject: 1 if null hypothesis rejected, 0 if null hypothesis accepted

    """
    n_samples = X.shape[0]

    # compute kernel matrices
    K = make_K(X)
    L = make_K(Y)

    statistic = HSIC(K, L, n_samples, biased)
    statistics_sort = marginal_null_dist(K, L, n_samples, n_perms, biased)

    p_value = 1 - (percentileofscore(statistics_sort, statistic) / 100)
    if p_value < alpha:
        reject = 1
    else:
        reject = 0

    return reject, p_value


# CONDITIONAL INDEPENDENCE TEST
# generate CPT copies of X when the conditional distribution is Gaussian
# i.e. X | Z=Z_i ~ N(mu[i], var[i])
def generate_X_CPT_Gaussian(n_steps, n_perms, X, mean, cov):
    log_lik_mat = -1/2 * (X-mean) @ np.linalg.pinv(cov) @ (X-mean).T
    perm_mat = generate_X_CPT(n_steps, n_perms, log_lik_mat)
    return X[perm_mat]


# generate CPT copies of X in general case
# log_lik_mat[i,j] = q(X[i]|Z[j]) where q(x|z) is the conditional density for X|Z
def generate_X_CPT(n_steps, n_perms, log_lik_mat, perm_init=[]):
    n = log_lik_mat.shape[0]
    if len(perm_init)==0:
        perm_init = np.arange(n, dtype=int)
    perm = generate_X_CPT_MC(n_steps, log_lik_mat, perm_init)
    perm_mat = np.zeros((n_perms, n),dtype=int)
    for p in range(n_perms):
        perm_mat[p] = generate_X_CPT_MC(n_steps, log_lik_mat, perm)
    return perm_mat


def generate_X_CPT_MC(n_steps, log_lik_mat, perm):
    n = len(perm)
    n_pairs = np.floor(n/2).astype(int)
    for i in range(n_steps):
        perms = np.random.choice(n, n, replace=False)
        inds_i = perms[0:n_pairs]
        inds_j = perms[n_pairs:(2*n_pairs)]
        # for each k=1, ..., n_pairs, decide whether to swap perm[inds_i[k]] with perm[inds_j[k]]
        log_odds = log_lik_mat[perm[inds_i], inds_j] + log_lik_mat[perm[inds_j], inds_i] - \
                   log_lik_mat[perm[inds_i], inds_i] - log_lik_mat[perm[inds_j], inds_j]

        swaps = np.random.binomial(1, 1/(1+np.exp(-np.maximum(-500, log_odds))))

        perm[inds_i], perm[inds_j] = perm[inds_i] + swaps * (perm[inds_j]-perm[inds_i]), perm[inds_j] - \
                                     swaps * (perm[inds_j]-perm[inds_i])
    return perm


@njit
def HSCIC(k_X, k_Y, k_Z, W):
    term1 = k_Z.T @ W
    term2 = W @ k_Z
    x_term2 = k_X @ term2
    y_term2 = k_Y @ term2
    first = term1 @ np.multiply(k_X, k_Y) @ term2
    second = term1 @ np.multiply(x_term2, y_term2)
    third = (term1 @ x_term2) * (term1 @ y_term2)
    return (first - 2 * second + third)[0, 0]


def cond_null_dist_perm(perm_X_CPT, k_Y, k_Zs, W, make_K):
    k_X_CPT = make_K(perm_X_CPT)
    return np.sum([HSCIC(k_X_CPT, k_Y, k_Z, W) for k_Z in k_Zs])


def cond_null_dist(X_CPT, k_Y, Z_arr, W, make_K, n_perms):
    """
    Approximates the null distribution
    """
    n_nodes, n_samples, d = Z_arr.shape

    Z = np.zeros((n_samples, n_nodes * d))
    for node in range(n_nodes):
        Z[:, node * d:(node + 1) * d] = Z_arr[node]

    k_Zs = [make_K(Z, reshaped_z) for z, reshaped_z in zip(Z, reshape(Z, n_nodes * d))]

    with get_context('spawn').Pool(cpu_count()) as pool:
        jobs = [pool.apply_async(cond_null_dist_perm, (x_CPT, k_Y, k_Zs, W, make_K))
                for x_CPT in reshape(X_CPT[:n_perms], d)]

        return np.sort([job.get() for job in jobs])


def cond_indep_test(X, Y, Z_arr, lamb, alpha, n_perms, n_steps, make_K, pretest):
    if len(X.shape)==1:
        X = X.reshape(-1, 1)
        Y = Y.reshape(-1, 1)
        Z_arr = Z_arr.reshape(-1, 1)

    n_nodes, n_samples, n_obs = Z_arr.shape

    Z = np.zeros((n_samples, n_nodes*n_obs))
    for node in range(n_nodes):
        Z[:, node*n_obs:(node+1)*n_obs] = Z_arr[node]

    # test statistic
    statistic = 0
    k_X = make_K(X)
    k_Y = make_K(Y)
    k_Z = make_K(Z)

    if (len(X.shape)==1) or (X.shape[1]==1):
        X_mean = np.tile(np.mean(X), n_samples)
        X_cov = np.tile(np.cov(X), n_samples)
    elif len(X.shape)>1:
        X_mean = np.mean(X, axis=0)
        X_cov = np.cov(X, rowvar=False)
    else:
        raise ValueError('Shape of X not supported.')

    # CPT
    X_CPT = generate_X_CPT_Gaussian(n_steps, n_perms, X, X_mean, X_cov).squeeze()

    W = np.linalg.inv(k_Z + lamb * np.identity(n_samples))

    if pretest is True:
        # permute once to compute test statistic
        k_X_CPT = make_K(X_CPT[np.random.randint(0, n_perms)].reshape(-1, n_obs))
        for z in Z:
            k_Z = make_K(Z, z.reshape(-1, n_nodes * n_obs))
            statistic += HSCIC(k_X_CPT, k_Y, k_Z, W)
    else:
        for z in Z:
            k_Z = make_K(Z, z.reshape(-1, n_nodes * n_obs))
            statistic += HSCIC(k_X, k_Y, k_Z, W)

    statistics_sort = cond_null_dist(X_CPT, k_Y, Z_arr, W, make_K, n_perms)

    # compute p-value
    p_value = 1 - (percentileofscore(statistics_sort, statistic)/100)
    if p_value < alpha:
        reject = 1
    else:
        reject = 0

    return reject, p_value


def opt_lambda(X, Y, Z, lambs, n_pretests, n_perms, n_steps, alpha, K):
    """
    Function to find optimal lambda value

    Inputs:
    lambs: range to iterate over for optimal value for regularisation of kernel ridge regression to compute HSCIC
    n_pretests: number of tests to find optimal value for lambda  (only for conditional independence test)
    K: the kernel function to use

    Returns:
    lamb_opt: optimal lambda value
    rejects_opt: number of rejections out of n_pretests with lamb_opt
    """

    if K == 'K_ID':
        make_K = K_ID
    elif K == K_ID:
        make_K = K_ID
    else:
        raise ValueError('Only K_ID is supported currently.')

    rejects_lamb = {}
    # find optimal lambda
    print('Finding optimal lambda:')
    for j, lamb in enumerate(lambs):
        print('Evaluating lambda =', lamb, '(value', j+1, 'of {})...'.format(len(lambs)))
        rejects_lamb_list = np.zeros(n_pretests)
        p_values_lamb_list = np.zeros(n_pretests)

        for i in tqdm(range(n_pretests)):
            rejects_lamb_list[i], p_values_lamb_list[i] = cond_indep_test(X, Y, Z, lamb, alpha, n_perms, n_steps,
                                                                          make_K, pretest=True)
            if np.sum(rejects_lamb_list) > 2*alpha*n_pretests:
                rejects_lamb[lamb] = n_pretests
                break
            else:
                rejects_lamb[lamb] = np.mean(rejects_lamb_list)
                continue
        print('...Completed with a rejection rate of {}.'.format(rejects_lamb[lamb]))

    # select lambda which gave percentage of rejections closest to alpha
    rejects_opt = min(rejects_lamb.values(), key=lambda x: abs(alpha - x))
    lamb_opt = lambs[[i for i, rej in enumerate(rejects_lamb.values()) if rej == rejects_opt][-1]]

    if rejects_opt > 2*alpha:
        raise ValueError('Failed to find optimal lambda in the given range. Change your range of lambdas.')
    else:
        print('Optimal lambda is', lamb_opt, 'and resulted in a rejection rate of {}.'.format(rejects_opt))

    return lamb_opt, rejects_opt


# JOINT INDEPENDENCE TEST
def make_K_list(X_array, n_nodes, make_K):
    """
    Computes the kernel matrices of the variables in X_list, where each column represents one variable.
    Returns a list of the kernel matrices of each variable.
    """
    k_list = list(make_K(X_array[i]) for i in range(n_nodes))
    return k_list


def dHSIC(k_list):
    """
    Computes the dHSIC statistic
    """
    n_nodes = len(k_list)
    n_samples = k_list[0].shape[0]

    term1, term2, term3 = 1, 1, 2 / n_samples
    for j in range(n_nodes):
        term1 = term1 * k_list[j]
        term2 = term2 * np.sum(k_list[j]) / (n_samples ** 2)
        term3 = term3 * np.sum(k_list[j], axis=0) / n_samples
    term1_sum = np.sum(term1)
    term3_sum = np.sum(term3)
    return term1_sum / (n_samples ** 2) + term2 - term3_sum


def joint_null_dist(k_list, n_samples, n_nodes, n_perms):
    """
    Approximates the null distribution by permuting all variables
    """
    # initiating statistics
    statistics = np.zeros(n_perms)

    for i in range(n_perms):
        term1 = k_list[0]
        term2 = np.sum(k_list[0]) / (n_samples ** 2)
        term3 = 2 * np.sum(k_list[0], axis=0) / (n_samples ** 2)

        for j in range(1, n_nodes):
            index_perm = np.random.permutation(k_list[j].shape[0])
            k_perm = k_list[j][index_perm, index_perm[:, None]]

            term1 = term1 * k_perm
            term2 = term2 * np.sum(k_perm) / (n_samples ** 2)
            term3 = term3 * np.sum(k_perm, axis=0) / n_samples

        term1_sum = np.sum(term1)
        term3_sum = np.sum(term3)

        statistics[i] = term1_sum / (n_samples**2) + term2 - term3_sum

    return np.sort(statistics)


def joint_indep_test(X_array, n_perms, alpha, make_K):
    """
    Performs a joint independence test with dHSIC as a test statistic
    and returns an 'accept' (0) or 'reject' (1) statement for the null hypothesis of joint independence

    Inputs:
    X_array: (n_vars, n_samples, n_obs) array of variables, each having dimensions (n_samples, n_obs)
    n_perms: number of permutations performed when bootstrapping the null
    alpha: rejection threshold of the test
    make_K: function called to construct the kernel matrix

    Returns:
    reject: 1 if null rejected, 0 if null accepted
    p_value: p-value of test
    """
    n_nodes, n_samples, n_obs = X_array.shape

    # compute list of kernel matrices
    K_list = make_K_list(X_array, n_nodes, make_K)

    # statistic and p-value
    statistic = dHSIC(K_list)
    statistics_sort = joint_null_dist(K_list, n_samples, n_nodes, n_perms)

    p_value = 1 - (percentileofscore(statistics_sort, statistic) / 100)
    if p_value < alpha:
        reject = 1
    else:
        reject = 0

    return reject, p_value


# TEST POWER
def test_power(X, Y=None, Z=None, edges_dict=None, n_trials=200, n_perms=1000, alpha=0.05, K='K_ID', test='marginal',
               lamb_opt=1e-4, n_steps=50, biased=True):
    """
    Computes the test power by conducting multiple independence tests in parallel
    and returning the percentage of null hypothesis rejections

    Inputs:
    X: (n_samples * n_tests, n_obs) array of samples from the first distribution
        or list of arrays for joint independence test
    Y: (n_samples * n_tests, n_obs) array of samples from the second distribution
    Z: (n_samples * n_tests, n_obs) array of samples from the third distribution
       (only for conditional independence tests)
    edges_dict: dictionary of form key: descendent, value: parents (only for joint independence test)
    n_trials: number of old_trials to compute percentage of rejections over
    n_perms: number of permutations performed when bootstrapping the null distribution
    alpha: rejection threshold of the test
    make_K: function called to construct the kernel matrix
    test: (str) which test to perform ('marginal', 'conditional', or 'joint')
    lamb_opt: range to iterate over for optimal value for regularisation of kernel ridge regression to compute HSCIC
           (only for conditional independence test)
    n_steps: number of MC iterations in the CPT (only for conditional independence test)
    biased: Boolean parameter to determine whether to compute the biased or unbiased estimator of HSIC
            (only for marginal independence test)

    Returns:
    power: the percentage of rejections of the null hypothesis
    """
    if test=='joint':
        n_vars, n_samples, n_obs = X[0].shape
    else:
        n_samples_n_trials, n_obs = X.shape
        n_samples = int(n_samples_n_trials / n_trials)

    rejects = np.zeros(n_trials)
    p_values = np.zeros(n_trials)

    if K == 'K_ID':
        make_K = K_ID
    elif K == 'K_dft':
        make_K = K_dft
    elif K == 'K_dft1':
        make_K = K_dft1
    elif K == 'K_dft2':
        make_K = K_dft2
    elif K == 'K_dct':
        make_K = K_dct
    elif K == 'K_dwt':
        make_K = K_dwt
    elif K == 'K_CEXP':
        make_K = K_CEXP
    else:
        raise ValueError('Kernel not implemented')

    for i in tqdm(range(n_trials)):
        if test == 'marginal':
            X_i = X[i * n_samples:(i + 1) * n_samples]
            Y_i = Y[i * n_samples:(i + 1) * n_samples]
            rejects[i], p_values[i] = marginal_indep_test(X_i, Y_i, n_perms, alpha, make_K, biased)
        elif test == 'conditional':
            X_i = X[i * n_samples:(i + 1) * n_samples]
            Y_i = Y[i * n_samples:(i + 1) * n_samples]
            Z_i = Z[:, i * n_samples:(i + 1) * n_samples]

            if i == 0:
                if type(lamb_opt) == float:
                    pass
                elif len(lamb_opt) == 1:
                    lamb_opt = lamb_opt[0]
                else:
                    raise ValueError('Lambda must be a number.')

            rejects[i], p_values[i] = cond_indep_test(X_i, Y_i, Z_i, lamb_opt, alpha, n_perms, n_steps,
                                                      make_K, pretest=False)

        elif test == 'joint':
            edges_dict_i, X_i = edges_dict[i], X[i]

            rejects[i], p_values[i] = joint_indep_test(X_i, n_perms, alpha, make_K)

        else:
            raise ValueError("Only independence tests of type 'marginal', 'conditional', or 'joint' are supported.")

    # compute percentage of rejections
    power = np.mean(rejects)

    return power

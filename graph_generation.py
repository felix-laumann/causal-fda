from graph_utils import n_DAGs, conditions, Meek_init, Meek_rules
import numpy as np
import networkx as nx
from independence import marginal_indep_test, cond_indep_test, opt_lambda


def generate_DAGs(n_nodes, y=1, prob=0.5):
    """
    Generate all possible DAGs given number of variables/nodes

    Inputs:
    n_nodes: number of variables/nodes in the graph (maximum 6)
    y: free parameter (default: 1)
    prob: probability of edge creation (default: 0.5)

    Returns:
    dict_DAGs: dictionary of DAGs where each DAG is another dictionary of form key: descendent, value: parents
    """
    if n_nodes > 6:
        raise ValueError('Not more than six variables supported due to the large number of candidate DAGs.')

    dict_DAGs = {}
    a_n = n_DAGs(n_nodes, y)

    len_dict_DAGs = 0
    i = 0
    while len(dict_DAGs) < a_n:
        # creating random directed graphs
        G = nx.fast_gnp_random_graph(n_nodes, prob, seed=i, directed=True)
        i += 1
        list_edges = []
        for (u, v) in G.edges():
            # removing the edges that result in 2-node loops
            if (u, v) and (v, u) not in list_edges:
                list_edges.append((u, v))

        # removing the edges that result in 3-node loops
        # 3-node graph
        if (0, 1) and (1, 2) and (2, 0) in list_edges:
            pass
        elif (1, 0) and (2, 1) and (0, 2) in list_edges:
            pass

        # 4-node graph
        elif (1, 2) and (2, 3) and (3, 1) in list_edges:
            pass
        elif (2, 1) and (3, 2) and (1, 3) in list_edges:
            pass
        elif (0, 2) and (2, 3) and (3, 0) in list_edges:
            pass
        elif (2, 0) and (3, 2) and (0, 3) in list_edges:
            pass
        elif (0, 1) and (1, 3) and (3, 0) in list_edges:
            pass
        elif (1, 0) and (3, 1) and (0, 3) in list_edges:
            pass

        # 5-node graph
        elif (0, 3) and (3, 4) and (4, 0) in list_edges:
            pass
        elif (3, 0) and (4, 3) and (0, 4) in list_edges:
            pass
        elif (0, 4) and (4, 2) and (2, 0) in list_edges:
            pass
        elif (4, 0) and (2, 4) and (0, 2) in list_edges:
            pass
        elif (0, 4) and (4, 1) and (1, 0) in list_edges:
            pass
        elif (4, 0) and (1, 4) and (0, 1) in list_edges:
            pass
        elif (4, 1) and (1, 3) and (3, 4) in list_edges:
            pass
        elif (1, 4) and (3, 1) and (4, 3) in list_edges:
            pass
        elif (4, 1) and (1, 2) and (2, 4) in list_edges:
            pass
        elif (1, 4) and (2, 1) and (4, 2) in list_edges:
            pass
        elif (4, 2) and (2, 3) and (3, 4) in list_edges:
            pass
        elif (2, 4) and (3, 2) and (4, 3) in list_edges:
            pass

        # 6-node graph
        elif (0, 4) and (4, 5) and (5, 0) in list_edges:
            pass
        elif (4, 0) and (5, 4) and (0, 5) in list_edges:
            pass
        elif (0, 5) and (5, 3) and (3, 0) in list_edges:
            pass
        elif (5, 0) and (3, 5) and (0, 3) in list_edges:
            pass
        elif (0, 5) and (5, 2) and (2, 0) in list_edges:
            pass
        elif (5, 0) and (2, 5) and (0, 2) in list_edges:
            pass
        elif (0, 5) and (5, 1) and (1, 0) in list_edges:
            pass
        elif (5, 0) and (1, 5) and (0, 1) in list_edges:
            pass

        # removing the edges that result in 4-node loops
        # 4-node graph
        elif (0, 1) and (1, 2) and (2, 3) and (3, 0) in list_edges:
            pass
        elif (1, 0) and (2, 1) and (3, 2) and (0, 3) in list_edges:
            pass

        # 5-node graph
        elif (4, 1) and (1, 2) and (2, 3) and (3, 4) in list_edges:
            pass
        elif (1, 4) and (2, 1) and (3, 2) and (4, 3) in list_edges:
            pass
        elif (4, 0) and (0, 2) and (2, 3) and (3, 4) in list_edges:
            pass
        elif (0, 4) and (2, 0) and (3, 2) and (4, 3) in list_edges:
            pass
        elif (4, 0) and (0, 1) and (1, 3) and (3, 4) in list_edges:
            pass
        elif (0, 4) and (1, 0) and (3, 1) and (4, 3) in list_edges:
            pass
        elif (4, 0) and (0, 1) and (1, 2) and (2, 0) in list_edges:
            pass
        elif (0, 4) and (1, 0) and (2, 1) and (0, 2) in list_edges:
            pass

        # 6-node graph
        elif (0, 5) and (5, 4) and (4, 1) and (1, 0) in list_edges:
            pass
        elif (5, 0) and (4, 5) and (1, 4) and (0, 1) in list_edges:
            pass
        elif (5, 2) and (2, 3) and (3, 4) and (4, 5) in list_edges:
            pass
        elif (2, 5) and (3, 2) and (4, 3) and (5, 4) in list_edges:
            pass
        elif (0, 3) and (3, 4) and (4, 5) and (5, 0) in list_edges:
            pass
        elif (3, 0) and (4, 3) and (5, 4) and (0, 5) in list_edges:
            pass
        elif (0, 1) and (1, 2) and (2, 5) and (5, 0) in list_edges:
            pass
        elif (1, 0) and (2, 1) and (5, 2) and (0, 5) in list_edges:
            pass
        elif (0, 1) and (1, 2) and (2, 3) and (3, 0) in list_edges:
            pass
        elif (1, 0) and (2, 1) and (3, 2) and (0, 3) in list_edges:
            pass
        elif (1, 4) and (4, 3) and (3, 2) and (2, 1) in list_edges:
            pass
        elif (4, 1) and (3, 4) and (2, 3) and (1, 2) in list_edges:
            pass

        # removing the edges that result in 5-node loops
        # 5-node graphs
        elif (0, 1) and (1, 2) and (2, 3) and (3, 4) and (4, 0) in list_edges:
            pass
        elif (1, 0) and (2, 1) and (3, 2) and (4, 3) and (0, 4) in list_edges:
            pass

        # 6-node graph
        elif (5, 1) and (1, 2) and (2, 3) and (3, 4) and (4, 5) in list_edges:
            pass
        elif (1, 5) and (2, 1) and (3, 2) and (4, 3) and (5, 4) in list_edges:
            pass
        elif (5, 0) and (0, 2) and (2, 3) and (3, 4) and (4, 5) in list_edges:
            pass
        elif (0, 5) and (2, 0) and (3, 2) and (4, 3) and (5, 4) in list_edges:
            pass
        elif (5, 0) and (0, 1) and (1, 3) and (3, 4) and (4, 5) in list_edges:
            pass
        elif (0, 5) and (1, 0) and (3, 1) and (4, 3) and (5, 4) in list_edges:
            pass
        elif (5, 0) and (0, 1) and (1, 2) and (2, 4) and (4, 5) in list_edges:
            pass
        elif (0, 5) and (1, 0) and (2, 1) and (4, 2) and (5, 4) in list_edges:
            pass
        elif (5, 0) and (0, 1) and (1, 2) and (2, 3) and (3, 5) in list_edges:
            pass
        elif (0, 5) and (1, 0) and (2, 1) and (3, 2) and (5, 3) in list_edges:
            pass
        elif (0, 1) and (1, 2) and (2, 3) and (3, 4) and (4, 0) in list_edges:
            pass
        elif (1, 0) and (2, 1) and (3, 2) and (4, 3) and (0, 4) in list_edges:
            pass

        # removing the edges that result in 6-node loops
        # 6-node graphs
        elif (0, 1) and (1, 2) and (2, 3) and (3, 4) and (4, 5) and (5, 0) in list_edges:
            pass
        elif (1, 0) and (2, 1) and (3, 2) and (4, 3) and (5, 4) and (0, 5) in list_edges:
            pass

        # if no loops are present, accept graph as DAG
        else:
            DAG = nx.DiGraph(list_edges)
            dict_DAGs[len_dict_DAGs] = DAG
            len_dict_DAGs += 1

    return dict_DAGs


def generate_DAGs_pd(pd_graph):
    """
    Generate all possible DAGs given a partially directed graph

    Inputs:
    pd_graph: partially directed graph

    Returns:
    dict_DAGs: dictionary of DAGs
    """
    # build the same function as generate_DAGs but only consider the ones that
    # can be constructed with partially directed graph pd_graph




    return dict_DAGs


def partially_direct(sparse_graph):
    """
    Generate partially directed graph a sparse graph

    Inputs:
    sparse_graph: list of tuples including two nodes that are connected by an edge

    Returns:
    pd_graph: partially directed graph of form key: descendent, value: parents; if two nodes are both descendants and
              parents of each other it means that the edge is undirected
    """
    pd_graph_init = Meek_init(sparse_graph)
    pd_graph = Meek_rules(pd_graph_init)

    return pd_graph


def sparsify_graph(X_array, lambs, n_pretests, n_perms, n_steps, alpha, make_K):
    """
    Generate sparsified graph given a number of variables/nodes

    Inputs:
    X_array: (n_nodes, n_samples, n_preds) array with data according to dependencies specified in dictionary edges
    lambs: range to iterate over for optimal value for regularisation of kernel ridge regression to compute HSCIC
           (only for conditional independence test)
    n_pretests: number of tests to find optimal value for lambda
    n_perms: number of permutations performed when bootstrapping the null distribution
    n_steps: number of MC iterations in the CPT
    alpha: rejection threshold of the test
    make_K: function called to construct the kernel matrix

    Returns:
    sparse_graph: sparse graph where edges are undirected but known to be of causal nature; is returned as list where
                  entries are tuples of two nodes that are connected by an edge
    """
    n_nodes, n_samples, n_preds = X_array.shape

    # generate list of edges with conditional sets
    edges_conditions = conditions(n_nodes)

    rejects = np.zeros(len(edges_conditions))
    p_values = np.zeros(len(edges_conditions))

    # iterate over each entry in list
    i = 0
    for e_c in edges_conditions:
        if e_c[2]==():    # perform marginal independence test if conditional set is empty
            rejects[i], p_values[i] = marginal_indep_test(X_array[e_c[0]], X_array[e_c[1]], n_perms, alpha, make_K,
                                                          biased=True)
        else:
            # find optimal lambda for conditional independence test
            lamb_opt, rejects_opt = opt_lambda(X_array[e_c[0]], X_array[e_c[1]], X_array[e_c[2]], lambs, n_pretests,
                                               n_perms, n_steps, alpha, make_K)
            # perform conditional independence test
            rejects[i], p_values[i] = cond_indep_test(X_array[e_c[0]], X_array[e_c[1]], X_array[e_c[2]], lamb_opt,
                                                      alpha, n_perms, n_steps, make_K, pretest=False)

        # skip tuples that include same edge if conditional independence is found
        r = 1
        if rejects[i]==0:
            # continue to next entry in list that does not have the same edge
            for i_r in range(i, len(edges_conditions)):
                if (edges_conditions[i][0], edges_conditions[i][1])==(edges_conditions[i_r][0], edges_conditions[i_r][1]):
                    r+=1
        i+=r

    # only keep entries in edges_conditions that are conditionally dependent
    sparse_graph = [(i, e) for i, e in enumerate(edges_conditions) if rejects[i]==1]

    return sparse_graph


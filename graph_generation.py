from graph_utils import n_DAGs, conditions, Meek_init, Meek_rules, both_edges, combinations_tuple
from causaldag import rand
import numpy as np
from itertools import product
import networkx as nx
from independence import marginal_indep_test, cond_indep_test, opt_lambda


def generate_DAGs(n_nodes, prob=0.5, discover=True):
    """
    Generate all possible DAGs given number of variables/nodes

    Inputs:
    n_nodes: number of variables/nodes in the graph (maximum 6)
    y: free parameter (default: 1)
    prob: probability of edge creation (default: 0.5)
    discover: (boolean) whether function is used to search over all possible configurations of DAGs or not

    Returns:
    dict_DAGs or Gs: list of DAGs where each DAG is another dictionary of form key: descendent, value: parents
    """
    if discover is True and n_nodes > 6:
        raise ValueError('Not more than six variables supported due to the large number of candidate DAGs.')

    if discover is True:
        a_n = n_DAGs(n_nodes)
    else:
        a_n = 1

    # creating random directed graphs
    Gs = []
    while len(Gs) < a_n:
        G = rand.directed_erdos(n_nodes, density=prob, size=1, as_list=True)
        if G[0] not in Gs:
            Gs.extend(G)

    if discover is True:
        # represent as dictionary
        dict_DAGs = {}
        for i, G in enumerate(Gs):
            dict_DAGs[i] = {}
            for node in G.to_nx().nodes():
                dict_DAGs[i][node] = []
            for edge in G.to_nx().edges():
                dict_DAGs[i][edge[1]].append(edge[0])
        return dict_DAGs
    else:
        return Gs


def generate_DAGs_pd(pd_graph):
    """
    Generate all possible DAGs given a partially directed graph

    Inputs:
    pd_graph: partially directed graph

    Returns:
    dict_DAGs: dictionary of DAGs
    """
    # convert dictionary into networkx MultiDiGraph
    pdgraph = nx.MultiDiGraph(pd_graph)

    # create list of undirected edges
    undirect_edges = []
    for (i, j) in combinations_tuple(range(pdgraph.number_of_nodes()), 2):
        if both_edges(pdgraph, i, j):
            undirect_edges.append(tuple((i, j)))

    # create all possible configurations of undirected edges
    # P stand for parent and D for descendant; P/D is the role of the first entry in each tuple
    confs = list(product('PD', repeat=len(undirect_edges)))

    # create dictionary of DAGs
    dict_DAGs = {}

    # have every configuration added to partially directed graph
    for i, conf in enumerate(confs):
        remove_edges = []
        for (node, edge) in zip(conf, undirect_edges):
            if node == 'P':    # meaning first node of undirected edge in tuple is parent of second node in tuple
                # remove edge that points from second to first node in pd_graph because first is parent of second note
                remove_edges.append(tuple((edge[1], edge[0])))
            elif node == 'D':  # meaning first node of undirected edge in tuple is descendant of second node in tuple
                # remove edge that points from first to second node in pd_graph because first is descendant of second note
                remove_edges.append(tuple((edge[0], edge[1])))
            else:
                continue
        G = pdgraph.copy()
        G.remove_edges_from(remove_edges)

        # graph is represented as dictionary with descendant as key and parents as value
        dict_DAGs[i] = {}
        for node in G.nodes():
            dict_DAGs[i][node] = []
        for edge in G.edges():
            dict_DAGs[i][edge[1]].append(edge[0])

    return dict_DAGs


def partially_direct(sparse_graph, analyse):
    """
    Estimate Markov equivalence class, a collection of partially directed graphs, of the data-generating DAG

    Inputs:
    sparse_graph: list of tuples including two nodes that are connected by an edge

    Returns:
    pd_graph: partially directed graph of form key: descendent, value: parents; if two nodes are both descendants and
              parents of each other it means that the edge is undirected
    """
    pd_graph_init = Meek_init(sparse_graph)
    pd_graph = Meek_rules(pd_graph_init)
    if analyse:
        print('Sparse graph', sparse_graph)
        print('Meek init:', pd_graph_init)
        print('Partially directed graph (CPDAG):', pd_graph.edges())

    # convert output to dictionary
    dict_DAG = {}
    for node in pd_graph.nodes():
        dict_DAG[node] = []
    for edge in pd_graph.edges():
        dict_DAG[edge[1]].append(edge[0])
    return dict_DAG


def sparsify_graph(X_array, lambs, n_pretests, n_perms, n_steps, alpha, make_K, l_cond, r_opts):
    """
    Generate undirected, sparsified graph given a number of variables/nodes (also called skeleton)

    Inputs:
    X_array: (n_nodes, n_samples, n_preds) array with data according to dependencies specified in dictionary edges
    lambs: range of optimal value for regularisation of kernel ridge regression to compute HSCIC based on pre-tests
           (only for conditional independence test)
    n_pretests: number of pretests to find optimal lambda value
    n_perms: number of permutations performed when bootstrapping the null distribution
    n_steps: number of MC iterations in the CPT
    alpha: rejection threshold of the test
    make_K: function called to construct the kernel matrix
    lamb_cond: array of optimal lambda values for each size of conditional sets
    rejects_opt: number of rejections with optimal lambda
    i: trial number

    Returns:
    sparse_graph: sparse graph where edges are undirected but known to be of causal nature; is returned as list where
                  entries are tuples of two nodes that are connected by an edge
    lamb_cond: array of optimal lambda values for each size of conditional sets
    rejects_opts: rejection rate with optimal lambda
    """
    n_nodes, n_samples, n_preds = X_array.shape

    # generate list of edges with conditional sets
    edges_conditions = conditions(n_nodes)

    rejects = np.zeros(len(edges_conditions))
    p_values = np.zeros(len(edges_conditions))

    sparse_graph = []

    # iterate over each entry in list
    j = 0
    while j < len(edges_conditions):
        e_c = edges_conditions[j]

        if len(e_c[2]) == 0:    # perform marginal independence test if conditional set is empty
            lamb_cond = np.zeros(n_nodes - 2)
            rejects_opts = np.zeros(n_nodes - 2)
            rejects[j], p_values[j] = marginal_indep_test(X_array[e_c[0]], X_array[e_c[1]], n_perms, alpha, make_K,
                                                          biased=True)
        else:
            if type(lambs) == float:
                l = len(e_c[2])
                lamb_cond = np.zeros(n_nodes - 2)
                rejects_opts = np.zeros(n_nodes - 2)
                lamb_cond[:] = lambs
                # perform conditional independence test
                rejects[j], p_values[j] = cond_indep_test(X_array[e_c[0]], X_array[e_c[1]],
                                                          X_array[list(e_c[2])].reshape(len(list(e_c[2])), n_samples, n_preds),
                                                          lamb_cond[l - 1], alpha, n_perms, n_steps, make_K, pretest=False)

            else:
                l = len(e_c[2])
                if l_cond[l-1] == 0:
                    # find optimal lambda for conditional set of size 1
                    l_cond[l-1], r_opts[l-1] = opt_lambda(X_array[e_c[0]], X_array[e_c[1]], X_array[list(e_c[2])].reshape(len(list(e_c[2])), n_samples, n_preds),
                                                                   lambs, n_pretests, n_perms, n_steps, alpha, K='K_ID')
                    # perform conditional independence test
                    rejects[j], p_values[j] = cond_indep_test(X_array[e_c[0]], X_array[e_c[1]],
                                                              X_array[list(e_c[2])].reshape(len(list(e_c[2])), n_samples, n_preds),
                                                              l_cond[l-1], alpha, n_perms, n_steps, make_K, pretest=False)
                else:
                    # perform conditional independence test
                    rejects[j], p_values[j] = cond_indep_test(X_array[e_c[0]], X_array[e_c[1]],
                                                              X_array[list(e_c[2])].reshape(len(list(e_c[2])), n_samples, n_preds),
                                                              l_cond[l-1], alpha, n_perms, n_steps, make_K, pretest=False)

        lamb_cond = l_cond
        rejects_opts = r_opts

        # skip tuples that include same edge if conditional independence is found
        if rejects[j] == 0:
            r = 0
            if tuple((e_c[0], e_c[1])) in sparse_graph:
                sparse_graph.remove(tuple((e_c[0], e_c[1])))

            # continue to next entry in list that is not about the same edge
            for j_r in range(j, len(edges_conditions)):
                if (e_c[0], e_c[1]) == (edges_conditions[j_r][0], edges_conditions[j_r][1]):
                    r += 1
        else:
            r = 1
            # only keep entries in sparse_graph that are conditionally dependent
            if tuple((e_c[0], e_c[1])) not in sparse_graph:
                sparse_graph.append(tuple((e_c[0], e_c[1])))

        j += r

    return sparse_graph, lamb_cond, rejects_opts


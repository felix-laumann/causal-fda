from graph_utils import n_DAGs, conditions, cond_set_init, max_p_init, Meek_rules, both_edges, combinations_tuple
from causaldag import rand
import numpy as np
import pickle
from tqdm import tqdm
from itertools import product, permutations
import networkx as nx
from multiprocessing import cpu_count, get_context
from independence import marginal_indep_test, cond_indep_test, opt_lambda


def generate_DAGs(n_nodes, analyse, prob=0.5, discover=True, save=False):
    """
    Generate all possible DAGs given number of variables/nodes

    Inputs:
    n_nodes: number of variables/nodes in the graph (maximum 5)
    y: free parameter (default: 1)
    prob: probability of edge creation (default: 0.5)
    discover: (boolean) whether function is used to search over all possible configurations of DAGs or not

    Returns:
    dict_DAGs or Gs: list of DAGs where each DAG is another dictionary of form key: descendent, value: parents
    """
    if discover is True and n_nodes > 5:
        raise ValueError('Not more than five variables supported due to the large number of candidate DAGs.')

    if discover is True:
        a_n = n_DAGs(n_nodes)
    else:
        a_n = 1

    if analyse:
        print('Number of graphs to generate:', a_n)

    # creating random directed graphs
    Gs = []
    i_G = 0
    if analyse:
        p_bar = tqdm(total=a_n)
    while i_G < a_n:
        G = rand.directed_erdos(n_nodes, density=prob, size=1, as_list=True)
        if G[0] not in Gs:
            Gs.extend(G)
            i_G += 1
            if analyse:
                p_bar.update(1)
    if analyse:
        p_bar.close()

    if discover is True:
        # represent as dictionary
        dict_DAGs = {}
        for i, G in enumerate(Gs):
            dict_DAGs[i] = {}
            for node in G.to_nx().nodes():
                dict_DAGs[i][node] = []
            for edge in G.to_nx().edges():
                dict_DAGs[i][edge[1]].append(edge[0])

        if save:
            cand_DAGs = open('tests/results/candidate_DAGs_{}_nodes_dis.pkl'.format(n_nodes), 'wb')
            pickle.dump(dict_DAGs, cand_DAGs)
            cand_DAGs.close()
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

    for (i, j) in combinations_tuple(range(pdgraph.number_of_nodes()+1), 2):
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


def partially_direct(sparse_graph, full_results, sepsets_results, init, analyse):
    """
    Estimate Markov equivalence class, a collection of partially directed graphs, of the data-generating DAG

    Inputs:
    sparse_graph: list of tuples including two nodes that are connected by an edge
    full_results: the complete results with edge, rejection, p-value and conditional set
    sepsets_results: results including the separation sets of each pair of nodes
    init: with which method the undirected skeleton graph shall be directed ('cond_set', 'max_p')

    Returns:
    pd_graph: partially directed graph of form key: descendent, value: parents; if two nodes are both descendants and
              parents of each other it means that the edge is undirected
    """
    # initial orientation based on unshielded triplets
    if init == 'cond_set':
        pd_graph_init = cond_set_init(sparse_graph, full_results, sepsets_results)
    elif init == 'max_p':
        pd_graph_init = max_p_init(sparse_graph, full_results)
    else:
        raise ValueError("Method not supported. Choose between 'cond_set' and 'max_p'.")

    # further orientation based on Meek's rules
    pd_graph = Meek_rules(pd_graph_init)

    if analyse:
        print('Sparse graph', sparse_graph.edges())
        print('Initial orientation:', pd_graph_init)
        print('Partially directed graph (CPDAG):', pd_graph.edges())

    # convert output to dictionary
    dict_DAG = {}
    for node in pd_graph.nodes:
        dict_DAG[node] = []
    for edge in pd_graph.edges:
        dict_DAG[edge[1]].append(edge[0])

    return dict_DAG


def sparsify_graph(X_array, lambs, n_pretests, n_perms, n_steps, alpha, make_K, l_cond, r_opts, analyse, find_lambda=False):
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
    l_cond: array of optimal lambda values for each size of conditional sets
    r_opt: number of rejections with optimal lambda
    find_lambda: (boolean) whether to search for lambda or not

    Returns:
    sparse_graph: sparse graph where edges are undirected but known to be of causal nature; is returned as list where
                  entries are tuples of two nodes that are connected by an edge
    full_results: the results of all conditional independence tests
    lamb_cond: array of optimal lambda values for each size of conditional sets
    rejects_opts: rejection rate with optimal lambda
    """
    n_nodes, n_samples, n_preds = X_array.shape

    # generate fully-connected graph
    sparse_graph = nx.Graph(combinations_tuple(range(n_nodes), 2))

    rejects = []
    p_values = []
    full_results = []
    sepsets_results = {key: [] for key in list(sparse_graph.edges())}

    lamb_cond = np.zeros(n_nodes - 2)
    rejects_opts = np.zeros(n_nodes - 2)

    # iterate over each possible size of the conditional set
    s = -1
    while np.max(sparse_graph.degree) - 1 > s:
        s += 1

        for x in range(n_nodes):
            neigh_x = list(sparse_graph.neighbors(x))
            if len(neigh_x) < s - 1:
                continue
            for y in neigh_x:
                neigh_x_wo_y = [v for v in neigh_x if v != y]

                for cond_set in combinations_tuple(neigh_x_wo_y, s):
                    if l_cond[s-1] == 0:
                        if find_lambda:
                            # find optimal lambda for conditional set of various sizes
                            l_cond[s-1], r_opts[s-1] = opt_lambda(X_array[x], X_array[y], X_array[list(cond_set)].reshape(len(list(cond_set)), n_samples, n_preds),
                                                                  lambs, n_pretests, n_perms, n_steps, alpha, K='K_ID')

                        else:
                            # choose optimal lambda from conditional independence test experiments
                            lamb_opts = pickle.load(open('results/conditional/lambs_opt_conditional.pkl', 'rb'))
                            #lamb_opts = pickle.load(open('lambs_opt_conditional.pkl', 'rb'))
                            if 0 < n_samples <= 100:
                                n_s = 100
                            elif 100 < n_samples <= 200:
                                n_s = 200
                            elif 200 < n_samples <= 300:
                                n_s = 300
                            else:
                                n_s = 300
                            l_cond[s-1] = lamb_opts[s][n_s] + 0.1

                    if s == 0:
                        # perform marginal independence test if conditional set is empty
                        reject, p_value, _ = marginal_indep_test(X_array[x], X_array[y], n_perms, alpha, make_K, biased=True)
                    else:
                        # perform conditional independence test if conditional set is not empty
                        reject, p_value, _ = cond_indep_test(X_array[x], X_array[y],
                                                             X_array[list(cond_set)].reshape(len(list(cond_set)), n_samples, n_preds),
                                                             l_cond[s-1], alpha, n_perms, n_steps, make_K, pretest=False)
                    rejects.append(reject)
                    p_values.append(p_value)

                    if analyse:
                        print('Conditional independence test between:', x, 'and', y,
                              'given', cond_set, 'with p-value:', p_value)

                    # save all results
                    full_results.append([(x, y), reject, p_value, cond_set])

        lamb_cond = l_cond
        rejects_opts = r_opts

    # loop over all edges and check if edge can be deleted where conditional independence is found
    for edge in sparse_graph.edges():
        edge_res = {}
        edge_results = {}
        for i_s in range(n_nodes - 2 + 1):
            edge_res[i_s] = [r for r in full_results if set(r[0]) == set(edge) if len(r[3]) == i_s]
            edge_results[i_s] = min(edge_res[i_s], key=lambda x_: x_[2])

        for e_r in edge_results.values():
            if e_r[1] == 0:
                if sparse_graph.has_edge(edge[0], edge[1]):
                    sparse_graph.remove_edge(edge[0], edge[1])
                for node in list(e_r[3]):
                    if node not in sepsets_results[edge]:
                        sepsets_results[edge].append(node)

    return sparse_graph, full_results, sepsets_results, lamb_cond, rejects_opts



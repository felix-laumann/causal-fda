import numpy as np
import networkx as nx
import pickle
from itertools import permutations, chain, combinations


# UTIL FUNCTIONS TO GENERATE DAGS
def n_DAGs(n):
    """
    Computation of the number of DAGs over all descents for a given number of n nodes,
    according to Archer et al. (2020), Corollary 11 (https://arxiv.org/pdf/1909.01550.pdf)

    Inputs:
    n: number of nodes (maximum 5)

    Returns:
    a_n: total number of DAGs
    """
    if n==1:
        a_n = 1
    elif n==2:
        a_n = 3
    elif n==3:
        a_n = 25
    elif n==4:
        a_n = 543
    elif n==5:
        a_n = 29281
    else:
        raise ValueError('Generating DAGs with more than five nodes is not implemented due to computational reasons. '
                         'We recommend to partially direct your graph first.')

    return a_n


# UTIL FUNCTIONS TO GENERATE PARTIALLY DIRECTED GRAPHS
def combinations_tuple(iterable, r):
    """
    Examples:
    combinations('ABCD', 2) --> AB AC AD BC BD CD
    combinations(range(4), 3) --> 012 013 023 123

    Inputs:
    iterable: sequence to be iterated
    r: length of each combination

    Returns:
    tuple of combinations
    """
    pool = tuple(iterable)
    n = len(pool)
    if r > n:
        return
    indices = list(range(r))
    yield tuple(pool[i] for i in indices)
    while True:
        for i in reversed(range(r)):
            if indices[i] != i + n - r:
                break
        else:
            return
        indices[i] += 1
        for j in range(i+1, r):
            indices[j] = indices[j-1] + 1
        yield tuple(pool[i] for i in indices)


def product(pool_0, pool_1):
    result = [[x, y]+[z] for x, y in pool_0 for z in pool_1]
    for prod in result:
        yield tuple(prod)


def conditions(n_nodes):
    """
    Generate list of tuples where first entry is first node, second entry is second node of edge, and the third entry of
    the tuple is the conditional set.

    Inputs:
    n_nodes: number of variables/nodes in the graph (no maximum)

    Returns:
    edges_conditions: list of tuples

    """
    # generate edges of a fully-connected undirected graph
    edges = tuple(combinations_tuple(range(n_nodes), 2))

    # generate conditional sets for all edges, including empty set
    _conditions = []
    for i in range(n_nodes):
        _conditions.extend(tuple(combinations_tuple(range(n_nodes), i)))

    edges_cond = list(product(edges, _conditions))

    # delete combinations where one of the two nodes forming the edge are also present in the conditional set
    edges_conditions = [e_c for e_c in edges_cond if all([e_c[0] not in e_c[2], e_c[1] not in e_c[2]])]

    return edges_conditions


def find_unshielded_triples(adj_mat, check=True):
    """
    Find the list of unshielded triples i--j--k in adjacency matrix as (i, j, k)
    """
    if check:
        assert np.all(adj_mat == adj_mat.T)  # the adjacency matrix must be symmetric
    unsh_trip = []

    if np.any(adj_mat != 0):  # if at least one edge, find all unshielded triples in adj_mat
        xy = np.transpose(np.nonzero(adj_mat))  # find (x,y) indices in adjacency matrix of edges
        for i in range(xy.shape[0]):
            x, y = xy[i][0], xy[i][1]
            # return the unique values in np.nonzero(adj_mat[y, :])[0] that are not in x
            all_z = np.setdiff1d(np.nonzero(adj_mat[y, :])[0], x)
            for z in all_z:
                if adj_mat[x, z] == 0 and adj_mat[z, x] == 0:
                    unsh_trip.append([xy[i][0], xy[i][1], z])

        # delete duplicates
        if len(unsh_trip) > 0:
            l = len(unsh_trip)
            delete_dup = np.full(l, False)
            for i in range(l):
                if unsh_trip[i][0] > unsh_trip[i][2]:
                    delete_dup[i] = True

            unsh_trip = [e for i_e, e in enumerate(unsh_trip) if delete_dup[i_e] == False]

    return unsh_trip


def sort_dict_ascending(d, descending=False):
    """
    Sort dict (dictionary) by its value in ascending order
    """
    dict_list = sorted(d.items(), key=lambda x: x[1], reverse=descending)
    return {dict_list[i][0]: dict_list[i][1] for i in range(len(dict_list))}


def cond_set_init(sparse_graph, full_results, sepsets_results):
    """
    Function to direct unshielded triples based on separation sets

    Inputs:
    sparse_graph: graph with undirected edges in form of list of tuples where each tuple is an edge
    full_results: the complete results with edge, rejection, p-value, full result and separation sets

    Returns:
    pd_graph_init: partially directed graph based on Phase I of the PC algorithm where unshielded triples are directed
    """
    pd_graph_init = {key: [] for key in list(sparse_graph.nodes())}
    pd_graph_init_dict = {}
    poss_orients = []  # possible orientations

    adj_mat = nx.to_numpy_array(sparse_graph, nodelist=sparse_graph.nodes())
    # find unshielded triples
    un_triples = [(i, j, k) for (i, j, k) in find_unshielded_triples(adj_mat)]

    for (x, y, z) in un_triples:
        # find possible separation sets
        poss_sepsets = [conds for edge, conds in sepsets_results.items() if edge == (x, z)]
        # define possible orientations
        if all(y not in sepset for sepset in poss_sepsets):
            poss_orients.append((x, y, z))    # x --> y <-- z

    for (x, y, z) in poss_orients:
        p_cond = [r[2] for r in full_results if r[0] == (x, z)]
        if p_cond:
            pd_graph_init_dict[(x, y, z)] = max(p_cond)

    pd_graph_init_sort = sort_dict_ascending(pd_graph_init_dict)

    for (x, y, z) in pd_graph_init_sort.keys():
        pd_graph_init[y].extend([x, z])

    if not poss_orients:
        for edge in sparse_graph.edges():
            # leave edge undirected (saved as double edge)
            pd_graph_init[edge[0]].append(edge[1])
            pd_graph_init[edge[1]].append(edge[0])

    return pd_graph_init


def max_p_init(sparse_graph, full_results):
    """
    Function to direct unshielded triples based on the separating/conditional set with the highest p-value as described
    in Ramsey, J. (2016): https://arxiv.org/abs/1610.00378#

    Inputs:
    sparse_graph: graph with undirected edges in form of list of tuples where each tuple is an edge
    full_results: the complete results with edge, rejection, p-value and conditional set

    Returns:
    pd_graph_init: partially directed graph based on Phase I of the PC algorithm where unshielded triples are directed
    """
    pd_graph_init = {}

    # Phase I: iterate over all edges and find unshielded triples (0 - 1 - 2)
    if len(sparse_graph.edges) == 1:
        # leave edge undirected (saved as double edge)
        pd_graph_init[list(sparse_graph.edges)[0][0]] = []
        pd_graph_init[list(sparse_graph.edges)[0][0]].append(list(sparse_graph.edges)[0][1])
        pd_graph_init[list(sparse_graph.edges)[0][1]] = []
        pd_graph_init[list(sparse_graph.edges)[0][1]].append(list(sparse_graph.edges)[0][0])

    for edge in sparse_graph.edges:
        max_p = 0
        max_sep_set = []
        # iterate over all separating conditional sets
        rows = [result for result in full_results if result[0] == edge]
        for row in rows:
            # find maximum p-value
            if row[2] > max_p:
                max_p = row[2]
                max_sep_set = row[3]

        # direct the edges to the separating conditional set with the highest p-value
        for max_sep_var in max_sep_set:
            pd_graph_init[max_sep_var] = [edge[0], edge[1]]

    return pd_graph_init


def both_edges(G, i, j):
    return G.has_edge(i, j) and G.has_edge(j, i)


def any_edge(G, i, j):
    return G.has_edge(i, j) or G.has_edge(j, i)


def find_triangles(adj_mat):
    """
    Return the list of triangles i o-o j o-o k o-o i in adj_mat as (i, j, k) [with symmetry]
    """
    return [(pair[0][0], pair[0][1], pair[1][1]) for pair in permutations(adj_mat, 2)
            if pair[0][1] == pair[1][0] and pair[0][0] != pair[1][1] and (pair[0][0], pair[1][1]) in adj_mat]


def find_kites(adj_mat):
    """
    Return the list of non-ambiguous kites i o-o j o-o l o-o k o-o i o-o l in adj_mat \
    (where j and k are non-adjacent) as (i, j, k, l) [with asymmetry j < k]
    """
    return [(pair[0][0], pair[0][1], pair[1][1], pair[0][2]) for pair in permutations(find_triangles(adj_mat), 2)
            if pair[0][0] == pair[1][0] and pair[0][2] == pair[1][2]
            and pair[0][1] < pair[1][1] and adj_mat[pair[0][1], pair[1][1]] == 0]


def Meek_rules(pd_graph_init):
    """
    Implementation of Meek's orientation rules

    Inputs:
    pd_graph_init: partially directed graph based on Meek_init

    Returns:
    pd_graph: partially directed graph after Meek's orientation rules were applied

    """
    # define a graph of initial parents and descendants where first entry in edge is parent and second is descendant
    pd_graph = nx.MultiDiGraph()
    for desc, parents in pd_graph_init.items():
        pd_graph.add_node(desc)
        if parents:
            for i in range(len(parents)):
                pd_graph.add_edge(parents[i], desc)

    adj_mat = nx.to_numpy_array(pd_graph, nodelist=pd_graph.nodes())
    un_triples = find_unshielded_triples(adj_mat, check=False)
    tris = find_triangles(adj_mat)
    kites = find_kites(adj_mat)

    loop = True
    while loop:
        loop = False
        # Rule 1
        for (i, j, k) in un_triples:
            if (i in pd_graph.predecessors(j)) and (both_edges(pd_graph, j, k)):
                if k in pd_graph.predecessors(j):
                    continue
                pd_graph.remove_edge(k, j)
                loop = True

        # Rule 2
        for (i, j, k) in tris:
            if (i in pd_graph.predecessors(j)) and (j in pd_graph.predecessors(k)) and (both_edges(pd_graph, i, k)):
                if k in pd_graph.predecessors(i):
                    continue
                pd_graph.remove_edge(k, i)
                loop = True

        # Rule 3 and 4
        for (i, j, k, l) in kites:
            if (both_edges(pd_graph, i, j)) and (both_edges(pd_graph, i, k)) and (j in pd_graph.predecessors(l)) and (k in pd_graph.predecessors(l)) and (both_edges(pd_graph, i, l)):
                if l in pd_graph.predecessors(i):
                    continue
                pd_graph.remove_edge(l, i)
                loop = True

    return pd_graph

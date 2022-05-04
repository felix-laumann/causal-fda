import numpy as np
import networkx as nx


# UTIL FUNCTIONS TO GENERATE DAGS
def q_binom(n, q):
    n_q = 0
    for _j in range(n-1):
        for _i in range(_j):
            n_q += np.power(q, _i)
    return n_q


def n_DAGs_u(n, u, y):
    """
    Computation of the number of DAGs with u descents for a given number of n nodes,
    according to Archer et al. (2020), Corollary 11 (https://arxiv.org/pdf/1909.01550.pdf)

    Inputs:
    n: number of nodes
    u: number of descents
    y: free parameter (default: 1)

    Returns:
    a_n_u: number of DAGs with u descents
    """
    q = (1+u*y)/(1+y)

    a_i = np.zeros(n)
    a_i[-1] = 1
    for i in range(n-1):
        ni_q = q_binom(n, q) / (q_binom(i, q) * q_binom((n-i), q))
        a_i[i] = np.power(-1, n-i-1) * ni_q * np.power(1+y, i*(n-i)) * a_i[i-1]
    a_n_u = np.sum(a_i)
    return a_n_u


def n_DAGs(n, y=1):
    """
    Computation of the number of DAGs over all descents for a given number of n nodes,
    according to Archer et al. (2020), Corollary 11 (https://arxiv.org/pdf/1909.01550.pdf)

    Inputs:
    n: number of nodes (maximum 6)
    y: free parameter (default: 1)

    Returns:
    a_n: total number of DAGs
    """
    if n==1:
        u_range = range(1)
    elif n==2:
        u_range = range(2)
    elif n==3:
        u_range = range(4)
    elif n==4:
        u_range = range(7)
    elif n==5:
        u_range = range(11)
    elif n==6:
        u_range = range(16)
    else:
        raise ValueError('Generating DAGs with more than 6 nodes is not implemented due to computational reasons. '
                         'We recommend to partially direct your graph first with')

    a_n_u = np.zeros(max(u_range)+1)
    for u in u_range:
        a_n_u[u] = n_DAGs_u(n, u, y)
    a_n = np.sum(a_n_u)

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
    Generate list of tuples where first entry is first node and second entry is second node of edge; the third entry of
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


def Meek_init(sparse_graph):
    pd_graph_init = {}

    # Phase I: iterate over all edges and find unshielded triples (0 - 1 - 2)
    for edge_0 in sparse_graph:
        for edge_1 in sparse_graph:
            if edge_0 == edge_1:
                continue
            # form unshielded triples
            elif edge_0[0] in edge_1:
                for edge_2 in sparse_graph:
                    if (edge_0[1], edge_1[1]) == edge_2:
                        continue
                    else:
                        # direct edges 0 -> 1 <- 2
                        pd_graph_init[edge_1[0]] = []
                        pd_graph_init[edge_1[0]].append(tuple((edge_0[1], edge_1[1])))

            elif edge_0[1] in edge_1:
                for edge_2 in sparse_graph:
                    if (edge_0[0], edge_1[1]) == edge_2:
                        continue
                    else:
                        # direct edges 0 -> 1 <- 2
                        pd_graph_init[edge_1[0]] = []
                        pd_graph_init[edge_1[0]].append(tuple((edge_0[0], edge_1[1])))

            else:
                # leave edge undirected
                pd_graph_init[edge_0[0]] = []
                pd_graph_init[edge_0[0]].append(edge_0[1])
                pd_graph_init[edge_0[1]] = []
                pd_graph_init[edge_0[1]].append(edge_0[0])

    return pd_graph_init


def both_edges(G, i, j):
    return G.has_edge(i, j) and G.has_edge(j, i)


def any_edge(G, i, j):
    return G.has_edge(i, j) or G.has_edge(j, i)


def Meek_rules(pd_graph_init):
    """
    Implementation of Meek's orientation rules

    Inputs:
    pd_graph_init: partially directed graph based on Meek_init

    Returns:
    pd_graph: partially directed graph after Meek's orientation rules were applied

    """
    # define a graph of initial parents and descendants
    pd_graph = nx.MultiDiGraph()
    for desc, parents in pd_graph_init:
        if len(list(parents)) > 1:
            for i in range(len(list(parents))):
                pd_graph.add_edge(parents[i], desc)
        else:
            pd_graph.add_edge(parents, desc)

    # Phase II
    for (i, j) in combinations_tuple(pd_graph.number_of_nodes(), 2):
        # Rule 1: if k -> i and i - j and k and j are not adjacent, then i -> j
        if both_edges(pd_graph, i, j):
            # look at all the predecessors of i
            for k in pd_graph.predecessors(i):
                # skip if there is an arrow i -> k
                if pd_graph.has_edge(i, k):
                    continue
                # skip if k and j are adjacent
                if any_edge(pd_graph, k, j):
                    continue
                # make i - j into i -> j
                pd_graph.remove_edge(j, i)
                break

        # Rule 2: orient i - j into i -> j whenever there is a chain i -> k -> j
        if both_edges(pd_graph, i, j):
            # find nodes k where k is i -> k
            succs_i = set()
            for k in pd_graph.successors(i):
                if not pd_graph.has_edge(k, i):
                    succs_i.add(k)
            # find nodes j where j is k -> j
            preds_j = set()
            for k in pd_graph.predecessors(j):
                if not pd_graph.has_edge(j, k):
                    preds_j.add(k)
            # check if there is any node k where i -> k -> j
            if len(succs_i & preds_j) > 0:
                # make i - j into i -> j
                pd_graph.remove_edge(j, i)

        # Rule 3: orient i - j into i -> j whenever there are two chains i -k -> j and i - l -> j such that
        # k and l are not adjacent
        if both_edges(pd_graph, i, j):
            # find nodes k where i - k
            adj_i = set()
            for k in pd_graph.successors(i):
                if pd_graph.has_edge(k, i):
                    adj_i.add(k)
            # for all the pairs of nodes in adj_i
            for (k, l) in combinations_tuple(adj_i, 2):
                # skip if k and l are adjacent
                if any_edge(pd_graph, k, l):
                    continue
                # skip if not k -> j
                if pd_graph.has_edge(j, k) or (not pd_graph.has_edge(k, j)):
                    continue
                # skip if not l -> j
                if pd_graph.has_edge(j, l) or (not pd_graph.has_edge(l, j)):
                    continue
                # make i - j into i -> j.
                pd_graph.remove_edge(j, i)
                break

        # Rule 4: orient i - j into i -> j whenever there are two chains i - k -> l and k -> l -> j
        # such that k and j are not adjacent; the edge i - l can optionally exist
        # Rule 4 is not necessary with PC algorithm

    return pd_graph

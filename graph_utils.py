import numpy as np
import networkx as nx


# UTIL FUNCTIONS TO GENERATE DAGS
def n_DAGs(n):
    """
    Computation of the number of DAGs over all descents for a given number of n nodes,
    according to Archer et al. (2020), Corollary 11 (https://arxiv.org/pdf/1909.01550.pdf)

    Inputs:
    n: number of nodes (maximum 6)

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
    elif n==6:
        a_n = 3781503
    else:
        raise ValueError('Generating DAGs with more than six nodes is not implemented due to computational reasons. '
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


def Meek_init(sparse_graph):
    """
    Function to direct unshielded triples

    Inputs:
    sparse_graph: graph with undirected edges in form of list of tuples where each tuple is an edge

    Returns:
    pd_graph_init: partially directed graph based on Phase I of the PC algorithm where unshielded triples are directed
    """
    pd_graph_init = {}

    # Phase I: iterate over all edges and find unshielded triples (0 - 1 - 2)
    if len(sparse_graph) == 1:
        # leave edge undirected (saved as double edge)
        pd_graph_init[sparse_graph[0][0]] = []
        pd_graph_init[sparse_graph[0][0]].append(sparse_graph[0][1])
        pd_graph_init[sparse_graph[0][1]] = []
        pd_graph_init[sparse_graph[0][1]].append(sparse_graph[0][0])

    for edge_i in sparse_graph:
        for edge_j in sparse_graph:
            if edge_i == edge_j:
                continue
            # form unshielded triples
            elif edge_i[0] == edge_j[0]:
                for edge_k in sparse_graph:
                    if (edge_i[1], edge_j[1]) == edge_k:    # checking whether edge 0 - 2 exists
                        continue
                    else:
                        # direct edges 0 -> 1 <- 2
                        pd_graph_init[edge_j[0]] = []
                        pd_graph_init[edge_j[0]].extend([edge_i[1], edge_j[1]])

            elif edge_i[0] == edge_j[1]:
                for edge_k in sparse_graph:
                    if (edge_i[1], edge_j[0]) == edge_k:    # checking whether edge 0 - 2 exists
                        continue
                    else:
                        # direct edges 0 -> 1 <- 2
                        pd_graph_init[edge_j[1]] = []
                        pd_graph_init[edge_j[1]].extend([edge_i[1], edge_j[0]])

            elif edge_i[1] == edge_j[0]:
                for edge_k in sparse_graph:
                    if (edge_i[0], edge_j[1]) == edge_k:    # checking whether edge 0 - 2 exists
                        continue
                    else:
                        # direct edges 0 -> 1 <- 2
                        pd_graph_init[edge_j[0]] = []
                        pd_graph_init[edge_j[0]].extend([edge_i[0], edge_j[1]])

            elif edge_i[1] == edge_j[1]:
                for edge_k in sparse_graph:
                    if (edge_i[0], edge_j[0]) == edge_k:    # checking whether edge 0 - 2 exists
                        continue
                    else:
                        # direct edges 0 -> 1 <- 2
                        pd_graph_init[edge_j[1]] = []
                        pd_graph_init[edge_j[1]].extend([edge_i[0], edge_j[0]])

            else:
                # leave edge undirected (saved as double edge)
                pd_graph_init[edge_i[0]] = []
                pd_graph_init[edge_i[0]].extend(edge_i[1])
                pd_graph_init[edge_i[1]] = []
                pd_graph_init[edge_i[1]].extend(edge_i[0])

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
    # define a graph of initial parents and descendants where first entry in edge is parent and second is descendant
    pd_graph = nx.MultiDiGraph()
    for desc, parents in pd_graph_init.items():
        for i in range(len(parents)):
            pd_graph.add_edge(parents[i], desc)

    # Phase II
    for (i, j) in list(combinations_tuple(range(pd_graph.number_of_nodes()), 2)):
        # Rule 1: if k -> i and i - j and k and j are not adjacent, then i -> j
        if both_edges(pd_graph, i, j):
            # look at all the parents (or predecessors) of i
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

        # Rule 2: orient i - j into i -> j whenever there is a chain i -> k -> j, and i and j are adjacent
        if both_edges(pd_graph, i, j):
            # find nodes k where k is descendant (or successor) of i, i -> k
            succs_i = set()
            for k in pd_graph.successors(i):
                if not pd_graph.has_edge(k, i):
                    succs_i.add(k)
            # find nodes k where k is parent (or predecessor) of j, k -> j
            preds_j = set()
            for k in pd_graph.predecessors(j):
                if not pd_graph.has_edge(j, k):
                    preds_j.add(k)
            # check if there is any node k where i -> k -> j
            if len(succs_i & preds_j) > 0:
                # make i - j into i -> j
                pd_graph.remove_edge(j, i)

        # Rule 3: orient i - j into i -> j whenever there are two chains i - k -> j and i - l -> j such that
        # k and l are not adjacent
        if both_edges(pd_graph, i, j):
            # find nodes k where i - k
            adj_i = set()
            for k in pd_graph.successors(i):
                if pd_graph.has_edge(k, i):    # finds all nodes k and l that are adjacent to i
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
                # make i - j into i -> j
                pd_graph.remove_edge(j, i)
                break

        # Rule 4: orient i - j into i -> j whenever there are two chains i - k -> l and k -> l -> j
        # such that k and j are not adjacent; the edge i - l can optionally exist
        if both_edges(pd_graph, i, j):
            # find nodes k where i - k
            adj_i = set()
            for k in pd_graph.successors(i):
                if pd_graph.has_edge(k, i):
                    adj_i.add(k)
            # for all the pairs of nodes in adj_i
            for (k, l) in combinations_tuple(adj_i, 2):
                # skip if k and j are adjacent
                if any_edge(pd_graph, k, j):
                    continue
                # skip if not k -> l
                if pd_graph.has_edge(l, k) or (not pd_graph.has_edge(k, l)):
                    continue
                # skip if not l -> j
                if pd_graph.has_edge(j, l) or (not pd_graph.has_edge(l, j)):
                    continue
                # make i - j into i -> j.
                pd_graph.remove_edge(j, i)
                break

    return pd_graph

from z3 import *
from igraph import *
import numpy as np


### Helper functions ###
########################

# Does not work when the valuations can be negative
# This is because a boolean array is used to determine
# whether a item should count or not. This is done by
# multipliyng it with values. But a zero resulting for
# this multiplicaiton will be more than any potential
# negative value
def max_in_product_array(d_i_j, v_i):
    product = [a*b for a, b in zip(d_i_j, v_i)]
    m = product[0]
    for v in product[1:]:
        m = If(v > m, v, m)
    return m


def max_in_product_array_bool(d_i_j, v_i):
    product = [If(a, 1, 0)*b for a, b in zip(d_i_j, v_i)]
    m = product[0]
    for v in product[1:]:
        m = If(v > m, v, m)
    return m


def get_edge_conflicts(G, A, n):
    conflicts = []
    # Enforce the conflicts from the graph
    for e in G.get_edgelist():
        g = e[0]
        h = e[1]

        for i in range(n):
            # Make sure that a single agent only can get one of two conflicting items
            conflicts.append((If(A[i][g], 1, 0) + If(A[i][h], 1, 0)) <= 1)

    return And([conflict for conflict in conflicts])

# TODO:: lage en test som sjekker ate denne funker riktig
def get_edge_connectivity_formulas(G, A, n, m):

    items_are_connected_to_something = []
    for agent in range(n):
        for i in range(m):
            item_is_connected_to_something = []
            for j in range(m):
                item_is_connected_to_something.append(
                    If(
                        And(A[agent][i]),
                        Or(bool(G[i][j]), bool(G[j][i])),
                        True
                    ))
            items_are_connected_to_something.append(
                Or(item_is_connected_to_something))

    return And(items_are_connected_to_something)


def get_edge_conflicts_adjacency_matrix(G, A, n, m):
    formulas = []

    # Enforce the conflicts from the graph
    # Only looking at the lower half if the adjacency matrix due to it being symmetric.
    # It is symmetric because conclift graphs are undirected
    for row in range(m):
        for col in range(row):
            for i in range(n):
                formulas.append(
                    If(
                        # If: there is an edge
                        G[row][col],

                        # Then: make sure that the same agent dont get the items in both ends
                        ((If(A[i][row], 1, 0) + If(A[i][col], 1, 0)) <= 1),

                        # Else: Do whatever
                        True
                    )
                )

    return And(formulas)


def get_edge_conflicts_adjacency_matrix_unknown_agents(G, A, m):
    formulas = []

    # Enforce the conflicts from the graph
    # Only looking at the lower half if the adjacency matrix due to it being symmetric.
    # It is symmetric because conflict graphs are undirected
    for row in range(m):
        for col in range(row):
            for i in range(m):
                formulas.append(
                    If(
                        # If: there is an edge
                        G[row][col],

                        # Then: make sure that the same agent dont get the items in both ends
                        ((If(A[i][row], 1, 0) + If(A[i][col], 1, 0)) <= 1),

                        # Else: Do whatever
                        True
                    )
                )

    return And(formulas)


def get_max_degree_less_than_agents(G, n, m):
    formulas = []

    for i in range(m):
        num_edges_for_item = 0
        for j in range(m):
            if j > i:
                num_edges_for_item = num_edges_for_item + \
                    If(G[j][i], 1, 0)
            else:
                num_edges_for_item = num_edges_for_item + \
                    If(G[i][j], 1, 0)

        formulas.append(num_edges_for_item < n)

    return And(formulas)


"""As a adjacency matrix for an undirected graph is symmetric, the program
really only have to worry about one side of the diagonal.
This function will also make sure that the diagonal is zero, meaning there
will be no self-loops."""


def get_upper_half_zero(G, m):
    formulas = []

    for row in range(m):
        for col in range(m):
            if col >= row:
                formulas.append(G[row][col] == False)

    return And(formulas)


def get_formula_for_correct_removing_of_items(A, D, n, m):
    formulas = []

    for i in range(n):
        for j in range(n):
            # Make sure that when checking for EF1, each agent is only allowed togive away one item to make another agent not jealous
            formulas.append(Sum([If(D[i][j][g], 1, 0) for g in range(m)]) <= 1)

            for g in range(m):
                # Make sure only items that are actuallallocated to the agent in question, is dropped
                formulas.append(If(D[i][j][g], 1, 0) <= If(A[j][g], 1, 0))

    return And([formula for formula in formulas])


def get_formula_for_one_item_to_one_agent(A, n, m):
    formulas = []
    # Each item allocated to at exactly one agent
    for g in range(m):
        formulas.append(Sum([If(A[i][g], 1, 0) for i in range(n)]) == 1)

    return And([formula for formula in formulas])


def get_formula_for_one_item_to_one_agent_uknown_agents(A, n, m):
    formulas = []
    # Each item allocated to at exactly one agent
    for g in range(m):
        formulas.append(Sum([If(i < n, If(A[i][g], 1, 0), 0)
                        for i in range(m)]) == 1)

    return And([formula for formula in formulas])


def get_formula_for_ensuring_ef1(A, V, n, m):
    formulas = []

    for i in range(n):
        for j in range(n):

            if i == j:
                continue

            # Check that there is no envy once an item is possibly dropped
            formulas.append(Sum([V[i][g] * If(A[i][g], 1, 0) for g in range(m)]) >=
                            Sum([V[i][g] * If(A[j][g], 1, 0) for g in range(m)]) - max_in_product_array_bool(A[j], V[i]))

    return And([formula for formula in formulas])


def get_formula_for_ensuring_ef1_unknown_agents(A, V, n, m):
    formulas = []

    for i in range(m):
        for j in range(m):

            if i == j:
                continue

            # Check that there is no envy once an item is possibly dropped
            formulas.append(If(And(i < n, j < n), Sum([V[i][g] * If(A[i][g], 1, 0) for g in range(m)]) >=
                            Sum([V[i][g] * If(A[j][g], 1, 0) for g in range(m)]) - max_in_product_array_bool(A[j], V[i]), True))

    return And(formulas)

################################################################


def is_ef1_possible(n, m, V):
    s = Solver()

    # A  keeps track of the allocated items
    A = [[Bool("a_%s_%s" % (i+1, j+1)) for j in range(m)]
         for i in range(n)]

    # D keepstrack of items that are dropped when checking for Ef1
    D = [[[Bool("d_%s_%s_%s" % (k+1, j+1, i+1)) for i in range(m)]
          for j in range(n)] for k in range(n)]

    s.add(get_formula_for_one_item_to_one_agent(A, n, m))
    s.add(get_formula_for_correct_removing_of_items(A, D, n, m))
    s.add(get_formula_for_ensuring_ef1(A, V, n, m))

    return s.check() == sat


def is_ef1_with_conflicts_possible(n, m, V, G):
    s = Solver()

    # Make sure that the number of nodes in the graph matches the number of items and the valuation function
    assert m == G.vcount(), "The number of items do not match the size of the graph"
    assert m == V[0].size, "The number of items do not match the valuation function"

    # A  keeps track of the allocated items
    A = [[Bool("a_%s_%s" % (i+1, j+1)) for j in range(m)]
         for i in range(n)]

    # TODO: remove all use of D?
    # D keepstrack of items that are dropped when checking for Ef1
    D = [[[Bool("d_%s_%s_%s" % (k+1, j+1, i+1)) for i in range(m)]
          for j in range(n)] for k in range(n)]

    s.add(get_formula_for_one_item_to_one_agent(A, n, m))
    s.add(get_formula_for_correct_removing_of_items(A, D, n, m))
    s.add(get_formula_for_ensuring_ef1(A, V, n, m))
    s.add(get_edge_conflicts(G, A, n))

    return s.check() == sat


def is_ef1_with_connectivity_possible(n, m, V, G_graph):
    s = Solver()

    # Make sure that the number of nodes in the graph matches the number of items and the valuation function
    assert m == G_graph.vcount(), "The number of items do not match the size of the graph"
    assert m == V[0].size, "The number of items do not match the valuation function"

    # A  keeps track of the allocated items
    A = [[Bool("a_%s_%s" % (i+1, j+1)) for j in range(m)]
         for i in range(n)]

    G = G_graph.get_adjacency()

    s.add(get_formula_for_one_item_to_one_agent(A, n, m))
    s.add(get_formula_for_ensuring_ef1_outer_good(A, G, V, n, m))
    s.add(get_edge_connectivity_formulas(G, A, n, m))

    return s.check() == sat


def find_valuation_function_with_no_ef1(n, m, G):
    s = Solver()

    # Make sure that the number of nodes in the graph matches the number of items and the valuation function
    assert m == G.vcount(), "The number of items do not match the size of the graph"

    # A  keeps track of the allocated items
    A = [[Bool("a_%s_%s" % (i+1, j+1)) for j in range(m)]
         for i in range(n)]

    # Valuations
    V = [[Int("v_agent%s_item%s" % (i, j)) for j in range(m)]
         for i in range(n)]

    # Make sure all values are non-negative
    for i in range(n):
        for j in range(m):
            s.add(V[i][j] >= 0)

    s.add(ForAll(
        [a for aa in A for a in aa],
        Implies(

            And(
                get_formula_for_one_item_to_one_agent(
                    [[a for a in aa] for aa in A], n, m),
                get_edge_conflicts(
                    G, [[a for a in aa] for aa in A], n)),

            Not(
                get_formula_for_ensuring_ef1(
                    [[a for a in aa] for aa in A], V, n, m)
            )
        )
    )
    )

    print(s.check())
    valuation_function = []
    is_sat = s.check()

    if(is_sat == sat):

        m = s.model()
        tuples = sorted([(d, m[d]) for d in m], key=lambda x: str(x[0]))
        valuation_function = [d[1] for d in tuples]

        # counter = 0
        # while s.check() == sat and counter < 15:
        #     counter = counter+1
        #     print(s.model())
        #     # prevent next model from using the same assignment as a previous model
        #     s.add(Or([(v != s.model()[v]) for vv in V for v in vv]))

    print()
    print(valuation_function)
    print()
    return (is_sat == sat, valuation_function)


def find_valuation_function_and_graph_with_no_ef1(n, m):
    s = Solver()

    # A  keeps track of the allocated items
    A = [[Bool("a_%s_%s" % (i+1, j+1)) for j in range(m)]
         for i in range(n)]

    # Valuations
    V = [[Int("v_agent%s_item%s" % (i, j)) for j in range(m)]
         for i in range(n)]

    # Adjacency matrux for conlfict graph
    G = [[Bool("g_row%s_col%s" % (i, j)) for j in range(m)]  # TODO ikke hardkode dette 2-tallet
         for i in range(m)]

    # Make sure all values are non-negative
    for i in range(n):
        for j in range(m):
            s.add(V[i][j] >= 0)

    s.add(get_upper_half_zero(G, m))
    s.add(get_max_degree_less_than_agents(G, n, m))

    s.add(ForAll(
        [a for aa in A for a in aa],
        Implies(

            And(
                get_formula_for_one_item_to_one_agent(
                    [[a for a in aa] for aa in A], n, m),
                get_edge_conflicts_adjacency_matrix(
                    G, [[a for a in aa] for aa in A], n, m)
            ),

            Not(
                get_formula_for_ensuring_ef1(
                    [[a for a in aa] for aa in A], V, n, m)
            )
        )
    ))

    print(s.check())
    valuation_function = []
    discovered_graph = []
    is_sat = s.check()
    matrix = [[]]
    if(is_sat == sat):

        mod = s.model()
        print(mod)
        tuples = sorted([(d, mod[d]) for d in mod], key=lambda x: str(x[0]))
        valuation_function = [d[1] for d in tuples[(m*m):len(tuples)]]
        discovered_graph = [d[1]
                            for d in tuples[0:(m*m)]]  # TODO fjerne hardkodet 2

        # make graph array into incidence matrix
        # looks weird because z3 numbers cannot be used direclty as numbers, you have to convert them to longs
        matrix = [[is_true(edge) for edge in discovered_graph[i:i+m]]
                  for i in range(0, len(discovered_graph), m)]  # TODO fjerne hardkodet 2

    print()
    print("valuation_function", valuation_function)
    print("discovered_graph:", matrix)

    graph = Graph.Adjacency(matrix, mode="max")

    return (is_sat == sat, valuation_function, graph)


def find_valuation_function_and_graph_and_agents_with_no_ef1(m):
    s = Solver()

    n = Int("n")

    # A  keeps track of the allocated items
    A = [[Bool("a_%s_%s" % (i+1, j+1)) for j in range(m)]
         for i in range(m)]

    # Valuations
    V = [[Int("v_agent%s_item%s" % (i, j)) for j in range(m)]
         for i in range(m)]

    # Adjacency matrux for conlfict graph
    G = [[Bool("g_row%s_col%s" % (i, j)) for j in range(m)]  # TODO ikke hardkode dette 2-tallet
         for i in range(m)]

    # Make sure all values are non-negative
    for i in range(m):
        for j in range(m):
            s.add(V[i][j] >= 0)

    s.add(n < m)
    # TODO: make the number of agents larger than the largest connected component of the graph
    s.add(get_upper_half_zero(G, m))
    s.add(get_max_degree_less_than_agents(G, n, m))

    s.add(ForAll(
        [a for aa in A for a in aa],
        Implies(

            And(
                get_formula_for_one_item_to_one_agent_uknown_agents(
                    [[a for a in aa] for aa in A], n, m),
                get_edge_conflicts_adjacency_matrix_unknown_agents(
                    G, [[a for a in aa] for aa in A], m)
            ),

            Not(
                get_formula_for_ensuring_ef1_unknown_agents(
                    [[a for a in aa] for aa in A], V, n, m)
            )
        )
    ))

    print(s.check())
    valuation_function = []
    discovered_graph = []
    is_sat = s.check()
    matrix = [[]]
    n_int = 0
    if(is_sat == sat):

        mod = s.model()
        n_int = mod[n].as_long()
        print(mod)
        tuples = sorted([(d, mod[d]) for d in mod], key=lambda x: str(x[0]))
        print([d[1] for d in tuples[(m*m):(len(tuples))]])

        # plus one because n is now a part of the answer
        valuation_function = [d[1] for d in tuples[(m*m+1):(m*m+n_int*m+1)]]
        discovered_graph = [d[1]
                            for d in tuples[0:(m*m)]]

        # make graph array into incidence matrix
        matrix = [[is_true(edge) for edge in discovered_graph[i:i+m]]
                  for i in range(0, len(discovered_graph), m)]

    print()
    print("valuation_function", valuation_function)
    print("discovered_graph:", matrix)

    graph = Graph.Adjacency(matrix, mode="max")

    return (is_sat == sat, valuation_function, graph, n_int)

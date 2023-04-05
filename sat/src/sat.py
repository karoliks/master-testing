from z3 import *
from igraph import *
import numpy as np

from helpers import get_edge_conflicts, get_edge_conflicts_adjacency_matrix, get_edge_conflicts_adjacency_matrix_unknown_agents, get_edge_conflicts_path_array, get_formula_for_correct_removing_of_items, get_formula_for_ensuring_ef1, get_formula_for_ensuring_ef1_unknown_agents, get_formula_for_one_item_to_one_agent, get_formula_for_one_item_to_one_agent_uknown_agents, get_formula_for_path, get_max_degree_less_than_agents, get_total_edges, get_upper_half_zero


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


def get_mms_for_this_agent(this_agent, n, m, V, G):
    formulas = []
    # TODO hvilket tall? og vanlig variabel eller z3 variabel?
    mms = Real("mms_%s" % this_agent)

    # A  keeps track of the allocated items
    A = [[Bool("a_%s_%s_mms_calculation_%s" % (i+1, j+1, this_agent)) for j in range(m)]  # TODO må jeg ha denne? virker som det blir mye vfor z3 å tenke på
         for i in range(n)]

    for i in range(n):
        formulas.append(mms <= Sum(
            [If(A[i][g], V[this_agent][g], 0) for g in range(m)]))  # look at this_agents values, because this is from her point of view

    opt = Optimize()
    opt.add(And(formulas))
    opt.add(get_formula_for_one_item_to_one_agent(A, n, m))
    opt.add(get_edge_conflicts(G, A, n))
    opt.maximize(mms)
    print(opt.check())
    mod = opt.model()
    print(mod)
    print(mod[mms])

    return mod[mms]


def get_value_to_optimize(n, m, scaled_V, G, A):
    formulas = []
    # TODO hvilket tall? og vanlig variabel eller z3 variabel?
    lowest_scaled_bundle_value = Real("lowest_scaled_bundle_value")

    for i in range(n):
        formulas.append(lowest_scaled_bundle_value <= Sum(
            [If(A[i][g], scaled_V[i][g], 0) for g in range(m)]))  # look at this_agents values, because this is from her point of view

    return And(formulas), lowest_scaled_bundle_value

# TODO skrive ferdig


def maximin_shares(n, m, V, G):
    s = Solver()

    # Make sure that the number of nodes in the graph matches the number of items and the valuation function
    assert m == G.vcount(), "The number of items do not match the size of the graph"
    assert m == V[0].size, "The number of items do not match the valuation function"

    individual_mms = [Real("mms_agent_%s" % (i+1)) for i in range(n)]
    alpha_mms_agents = [Real("alpha_agent_%s" % (i+1)) for i in range(n)]

    for i in range(n):
        individual_mms[i] = get_mms_for_this_agent(i, n, m, V, G)



    # A  keeps track of the allocated items
    A = [[Bool("a_%s_%s" % (i+1, j+1)) for j in range(m)]
         for i in range(n)]

    for i in range(n):
        alpha_mms_agents[i] = If(individual_mms[i] > 0, Sum([If(
            A[i][g], V[i][g], 0) for g in range(m)]) / individual_mms[i], 1)

    opt = Optimize()

    for i in range(n):
        opt.add_soft(individual_mms[i] <= Sum(
            [If(A[i][g], V[i][g], 0) for g in range(m)]))
        # TODO maximere bundle-verdiene i tillegg?

    opt.add(get_formula_for_one_item_to_one_agent(A, n, m))
    opt.add(get_edge_conflicts(G, A, n))


    print(opt.check())
    mod = opt.model()
    print(mod)

    for i in range(n):
        print("hei")
        print(mod.eval(alpha_mms_agents[i]))

    return


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


def find_valuation_function_and_graph_and_agents_with_no_ef1_binary_vals(m):
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

    # Forece the values to be binary
    for i in range(m):
        for j in range(m):
            s.add(Or(V[i][j] == 1, V[i][j] == 0))

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


def find_valuation_function_and_graph_and_agents_with_no_ef1_ternary_vals(m):
    s = Solver()

    n = Int("n")
    p = Int("p")

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

    # Forece the values to be ternary
    s.add(And(Not(p == 0), Not(p == 1), p > 0))
    for i in range(m):
        for j in range(m):
            s.add(Or(V[i][j] == 1, V[i][j] == 0, V[i][j] == p))

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


def matrix_path(m):
    s = Solver()
    # Adjacency matrux for conlfict graph
    G = [[Bool("g_row%s_col%s" % (i, j)) for j in range(m)]  # TODO ikke hardkode dette 2-tallet
         for i in range(m)]
    s.add(get_formula_for_path(G, m))
    s.add(get_upper_half_zero(G, m))

    is_sat = s.check()
    print(is_sat)
    matrix = [[]]
    if(is_sat == sat):

        mod = s.model()
        print(mod)

        print(mod.eval(get_total_edges(G, m)))

        tuples = sorted([(d, mod[d]) for d in mod], key=lambda x: str(x[0]))
        print([d[1] for d in tuples[(m*m):(len(tuples))]])

        discovered_graph = [d[1]
                            for d in tuples[0:(m*m)]]

        # make graph array into incidence matrix
        matrix = [[is_true(edge) for edge in discovered_graph[i:i+m]]
                  for i in range(0, len(discovered_graph), m)]

    print()
    print("discovered_graph:", matrix)

    graph = Graph.Adjacency(matrix, mode="max")

    return (is_sat == sat, graph)

# TODO:når jeg ikke har et -varianel t antall ting så er det vel ingengrunn til å lete etter en graf? kanvel bare gi den inn?


def find_valuation_function_and_graph_and_agents_with_no_ef1_only_paths(m):
    s = Solver()
    print()
    print("m:", m)

    n = Int("n")

    # A  keeps track of the allocated items
    A = [[Bool("a_%s_%s" % (i+1, j+1)) for j in range(m)]
         for i in range(m)]

    # Valuations
    V = [[Int("v_agent%s_item%s" % (i, j)) for j in range(m)]
         for i in range(m)]

    # Make sure all values are non-negative
    for i in range(m):
        for j in range(m):
            s.add(V[i][j] >= 0)

    s.add(n < m)
    # max degree of graph should be less than the number of agents (path has max degree equal to two)
    s.add(2 < n)
   
    s.add(ForAll(
        [a for aa in A for a in aa],
        Implies(

            And(
                get_formula_for_one_item_to_one_agent_uknown_agents(
                    [[a for a in aa] for aa in A], n, m),
                get_edge_conflicts_path_array(
                    [[a for a in aa] for aa in A], m, n)
            ),

            Not(
                get_formula_for_ensuring_ef1_unknown_agents(
                    [[a for a in aa] for aa in A], V, n, m)
            )
        )
    ))

    print(s.check())
    valuation_function = []
    is_sat = s.check()
    n_int = 0
    if(is_sat == sat):

        mod = s.model()
        n_int = mod[n].as_long()
        print(mod)
        print(n_int)
        tuples = sorted([(d, mod[d]) for d in mod], key=lambda x: str(x[0]))
        print([d[1] for d in tuples[(m*m):(len(tuples))]])

        # plus one because n is now a part of the answer
        valuation_function = [d[1] for d in tuples[(1):(n_int*m+1)]]

    print("valuation_function", valuation_function)

    return (is_sat == sat, valuation_function, n_int)

# TODO:når jeg ikke har et -varianel t antall ting så er det vel ingengrunn til å lete etter en graf? kanvel bare gi den inn?


def find_valuation_function_and_graph_and_agents_with_no_ef1_only_paths_and_cycles(m):
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
    s.add(get_formula_for_path(G, m))
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
        print(n_int)
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


def is_path_always_ef1():
    s = Solver()
    # s.set("smt.string_solver", "seq")  # TODO er denne nyttig her?

    IntSeqSort = SeqSort(IntSort())
    SeqSeqSort = SeqSort(IntSeqSort)

    n = Int("n")
    m = Int("m")

    values = Const("values", IntSeqSort)
    allocation = Const("allocation", IntSeqSort)

    s.add(Length(values) == n*m)
    s.add(Length(allocation) == n*m)

    # dummyIndex = FreshInt('dummyIndex')
    # s.add(ForAll(dummyIndex, Implies(dummyIndex < n,
    #                                  Length(allocation[dummyIndex]) == m)))
    # s.add(Length(allocation[0]) == m)

    s.add(n == 4)
    s.add(m == 6)

    print(s.check())
    if(s.check() == sat):
        print(s.model())
    return True

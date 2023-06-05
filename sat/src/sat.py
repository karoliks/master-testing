from z3 import *
from igraph import *

from helpers import get_edge_conflicts, get_edge_conflicts_adjacency_matrix, get_edge_conflicts_adjacency_matrix_unknown_agents, get_edge_conflicts_int, get_edge_conflicts_path_array, get_formula_for_correct_removing_of_items, get_formula_for_ensuring_ef1, get_formula_for_ensuring_ef1_equal_valuation_functions, get_formula_for_ensuring_ef1_unknown_agents, get_formula_for_ensuring_ef1_unknown_agents_boolean_values, get_formula_for_ensuring_efx, get_formula_for_ensuring_efx_unknown_agents, get_formula_for_one_item_to_one_agent, get_formula_for_one_item_to_one_agent_int, get_formula_for_one_item_to_one_agent_uknown_agents, get_formula_for_path, get_max_degree_less_than_agents, get_mms_for_this_agent, get_mms_for_this_agent_manual_optimization, get_total_edges, get_upper_half_zero




def is_ef1_possible(n, m, V):
    # Convert to python list, in case it is made with numpy
    V = V.tolist()
    s = Solver()

    # A  keeps track of the allocated items
    A = [[Bool("a_%s_%s" % (i+1, j+1)) for j in range(m)]
         for i in range(n)]

    s.add(get_formula_for_one_item_to_one_agent(A, n, m))
    s.add(get_formula_for_ensuring_ef1(A, V, n, m))

    return s.check() == sat


# TODO:: lage en test som sjekker ate denne funker riktig


def get_edge_connectivity_formulas(G, A, n):

    formulas = []
    for agent in range(n):
        # -1 will make the function not care about the missing item
        formulas.append(
            is_graph_connected_after_removing_item(G, A[agent], -1, agent))

    return And(formulas)

# T)D) kanskje ikke bruke nabimatrise, men en fynskjon? https://stackoverflow.com/questions/72507692/z3-connected-components

def is_ef1_with_conflicts_possible(n, m, V, G):
    s = Solver()

    # Make sure that the number of nodes in the graph matches the number of items and the valuation function
    assert m == G.vcount(), "The number of items do not match the size of the graph"
    assert m == V[0].size, "The number of items do not match the valuation function"
    # Convert to python list, in case it is made with numpy
    V = V.tolist()
    # A  keeps track of the allocated items
    A = [[Bool("a_%s_%s" % (i+1, j+1)) for j in range(m)]
         for i in range(n)]

    s.add(get_formula_for_one_item_to_one_agent(A, n, m))
    s.add(get_formula_for_ensuring_ef1(A, V, n, m))
    s.add(get_edge_conflicts(G, A, n))

    return s.check() == sat


def is_efx_with_conflicts_possible(n, m, V, G):
    s = Solver()
    s.set("timeout", 300000)

    # Make sure that the number of nodes in the graph matches the number of items and the valuation function
    assert m == G.vcount(), "The number of items do not match the size of the graph"
    assert m == V[0].size, "The number of items do not match the valuation function"
    # Convert to python list, in case it is made with numpy
    V = V.tolist()
    # A  keeps track of the allocated items
    A = [[Bool("a_%s_%s" % (i+1, j+1)) for j in range(m)]
         for i in range(n)]

    s.add(get_formula_for_one_item_to_one_agent(A, n, m))
    s.add(get_formula_for_ensuring_efx(A, V, n, m))
    s.add(get_edge_conflicts(G, A, n))

    return s.check() 



def maximin_shares(n, m, V, G=None):  # TODO ikke bruke optimization her?

    if G != None:
        # Make sure that the number of nodes in the graph matches the number of items and the valuation function
        assert m == G.vcount(), "The number of items do not match the size of the graph, items: " + \
            str(m)+" nodes: "+str(G.vcount())
    assert m == V[0].size, "The number of items do not match the valuation function"

    # individual_mms = [Real("mms_agent_%s" % (i+1)) for i in range(n)]
    individual_mms = [0 for i in range(n)]
    alpha_mms_agents = [Real("alpha_agent_%s" % (i+1)) for i in range(n)]

    for i in range(n):
        individual_mms[i], result = get_mms_for_this_agent(i, n, m, V, G)
        # print(type(individual_mms[i]))
        if result == unknown or result == unsat:
            return result
    # A  keeps track of the allocated items
    A = [[Bool("a_%s_%s" % (i+1, j+1)) for j in range(m)]
         for i in range(n)]

    for i in range(n):
        alpha_mms_agents[i] = If(individual_mms[i] > 0, Sum([If(
            A[i][g], V[i][g], 0) for g in range(m)]) / individual_mms[i], 1)

    opt = Solver()
    # opt.set("timeout", 300000)  # TODO increase timeout
    

    # min_alpha = Real("min_alpha")

    for i in range(n):
        bundle_value = Sum(
            [If(A[i][g], V[i][g], 0) for g in range(m)])

        opt.add(individual_mms[i] <= bundle_value)
        # opt.add(min_alpha <= bundle_value/individual_mms[i])
        # opt.maximize(bundle_value)
        # TODO maximere bundle-verdiene i tillegg?
    # opt.maximize(min_alpha)

    opt.add(get_formula_for_one_item_to_one_agent(A, n, m))
    if G != None:
        opt.add(get_edge_conflicts(G, A, n))
    is_sat = opt.check()
    print(is_sat)
    mod = opt.model()
    # print(mod)

    for i in range(n):
        print("alpha agent", i)
        print(mod.eval(alpha_mms_agents[i]).as_decimal(3))

    return is_sat == sat


def is_efx_possible(n, m, V):
    s = Solver()

    # A  keeps track of the allocated items
    A = [[Bool("a_%s_%s" % (i+1, j+1)) for j in range(m)]
         for i in range(n)]

    s.add(get_formula_for_one_item_to_one_agent(A, n, m))
    s.add(get_formula_for_ensuring_efx(A, V, n, m))
    s.set("timeout", 10000000)  # TODO timeout is now 10000 seconds.
    
    return s.check()


def maximin_shares_manual_optimization(n, m, V, G):
    s = Solver()

    # Make sure that the number of nodes in the graph matches the number of items and the valuation function
    assert m == G.vcount(), "The number of items do not match the size of the graph, items: " + \
        str(m)+" nodes: "+str(G.vcount())
    assert m == V[0].size, "The number of items do not match the valuation function"

    # individual_mms = [Real("mms_agent_%s" % (i+1)) for i in range(n)]
    individual_mms = [0 for i in range(n)]
    alpha_mms_agents = [Real("alpha_agent_%s" % (i+1)) for i in range(n)]

    for i in range(n):
        individual_mms[i] = get_mms_for_this_agent_manual_optimization(
            i, n, m, V, G)
        # print(type(individual_mms[i]))
        if individual_mms[i] == None:
            return False
    # A  keeps track of the allocated items
    A = [[Bool("a_%s_%s" % (i+1, j+1)) for j in range(m)]
         for i in range(n)]

    for i in range(n):
        alpha_mms_agents[i] = If(individual_mms[i] > 0, Sum([If(
            A[i][g], V[i][g], 0) for g in range(m)]) / individual_mms[i], 1)

    opt = Optimize()
    
    min_alpha = Real("min_alpha")

    for i in range(n):
        bundle_value = Sum(
            [If(A[i][g], V[i][g], 0) for g in range(m)])
        
        # opt.add_soft(individual_mms[i] <= bundle_value)
        opt.add(min_alpha <= bundle_value/individual_mms[i])
        # opt.maximize(bundle_value)
        # TODO maximere bundle-verdiene i tillegg?
    opt.maximize(min_alpha)

    opt.add(get_formula_for_one_item_to_one_agent(A, n, m))
    opt.add(get_edge_conflicts(G, A, n))
    opt.set("timeout", 40000)  # TODO increase timeout

    print(opt.check())
    mod = opt.model()
    # print(mod)

    for i in range(n):
        print("alpha agent", i)
        print(mod.eval(alpha_mms_agents[i]).as_decimal(3))

    return opt.check() == sat


def find_valuation_function_and_agents_with_no_efx(m):
    s = Solver()
    s.set("timeout", 18000000)  # TODO timeout is now 5 hours.
    
    
    n = Int("n")

    # A  keeps track of the allocated items
    A = [[Bool("a_%s_%s" % (i+1, j+1)) for j in range(m)]
         for i in range(m-1)]

    # Valuations
    V = [[Int("v_agent%s_item%s" % (i, j)) for j in range(m)]
         for i in range(m-1)]

    # Make sure all values are non-negative
    for i in range(m-1):
        for j in range(m):
            s.add(V[i][j] >= 0)
    
    # Neccesary restricion because of how the allocation matrix is made
    s.add(n < m)
    # s.add(n <= m)
    s.add(n >= 4)
    # s.add(n >= 2)
    

    s.add(ForAll(
        [a for aa in A for a in aa],
        Implies(

            get_formula_for_one_item_to_one_agent_uknown_agents(
                [[a for a in aa] for aa in A], n, m),

            Not(
                get_formula_for_ensuring_efx_unknown_agents(
                    [[a for a in aa] for aa in A], V, n, m)
            )
        )
    )
    )

    valuation_function = []
    is_sat = s.check()
    print(is_sat)
    n_int = 0

    if(is_sat == sat):

        mod = s.model()
        n_int = mod[n].as_long()
        
        print(mod)
        tuples = sorted([(d, mod[d]) for d in mod], key=lambda x: str(x[0]))
        valuation_function = [d[1] for d in tuples] # TODO ikke sikkert disse henter ut rikitge tall etter at n ble introdusert

        # counter = 0
        # while s.check() == sat and counter < 15:
        #     counter = counter+1
        #     print(s.model())
        #     # prevent next model from using the same assignment as a previous model
        #     s.add(Or([(v != s.model()[v]) for vv in V for v in vv]))

    print()
    print(valuation_function)
    print()
    return (is_sat == sat, valuation_function, n_int)




# def find_valuation_function_and_graph_and_agents_with_no_ef1(m):
#     s = Solver()

#     n = Int("n")

#     # A  keeps track of the allocated items
#     A = [[Bool("a_%s_%s" % (i+1, j+1)) for j in range(m)]
#          for i in range(m)]

#     # Valuations
#     V = [[Int("v_agent%s_item%s" % (i, j)) for j in range(m)]
#          for i in range(m)]

#     # Adjacency matrux for conlfict graph
#     G = [[Bool("g_row%s_col%s" % (i, j)) for j in range(m)]  # TODO ikke hardkode dette 2-tallet
#          for i in range(m)]

#     # Make sure all values are non-negative
#     for i in range(m):
#         for j in range(m):
#             s.add(V[i][j] >= 0)

#     s.add(n < m)
#     # TODO: make the number of agents larger than the largest connected component of the graph
#     s.add(get_upper_half_zero(G, m))
#     s.add(get_max_degree_less_than_agents(G, n, m))

#     s.add(ForAll(
#         [a for aa in A for a in aa],
#         Implies(

#             And(
#                 get_formula_for_one_item_to_one_agent_uknown_agents(
#                     [[a for a in aa] for aa in A], n, m),
#                 get_edge_conflicts_adjacency_matrix_unknown_agents(
#                     G, [[a for a in aa] for aa in A], m)
#             ),

# TODO fix denne er ikke riktig, men er god for å tenke gjennom til den andre funkjsonen


def get_bundle_graph_for_agent(A, G, n, m):
    for agent in range(n):
        for row in range(m):
            for col in range(row):
                is_outer = If(Or(A[agent][row], A[agent][col]))

            # A[n][m]


def get_formula_for_ensuring_ef1_outer_good(A, G, V, n, m):
    formulas = []

    for i in range(n):
        for j in range(n):

            if i == j:
                continue

            # Check that there is no envy once an item is possibly dropped
            # formulas.append(Sum([V[i][g] * If(A[i][g], 1, 0) for g in range(m)]) >=
            #                 Sum([V[i][g] * If(A[j][g], 1, 0) for g in range(m)]) - max_in_product_array_bool_outer_items(A[j], V[i], G, i, j))

            formulas.append(Or([(Sum([V[i][g] * If(A[i][g], 1, 0) for g in range(m)]) >=
                            Sum([V[i][g] * If(A[j][g], 1, 0) for g in range(m)]) -
                            If(
                                And(A[j][potentially_removed_item],
                                    is_graph_connected_after_removing_item(
                                        G, A[j], potentially_removed_item, j)
                                    ),
                                V[i][potentially_removed_item],
                                0)
                            ) for potentially_removed_item in range(m)]))

    return And([formula for formula in formulas])

#             Not(
#                 get_formula_for_ensuring_ef1_unknown_agents(
#                     [[a for a in aa] for aa in A], V, n, m)
#             )
#         )
#     ))

#     is_sat = s.check()
#     print(is_sat)
    
#     valuation_function = []
#     discovered_graph = []
#     matrix = [[]]
#     n_int = 0
#     if(is_sat == sat):

#         mod = s.model()
#         n_int = mod[n].as_long()
#         print(mod)
#         tuples = sorted([(d, mod[d]) for d in mod], key=lambda x: str(x[0]))
#         print([d[1] for d in tuples[(m*m):(len(tuples))]])

#         # plus one because n is now a part of the answer
#         valuation_function = [d[1] for d in tuples[(m*m+1):(m*m+n_int*m+1)]]
#         discovered_graph = [d[1]
#                             for d in tuples[0:(m*m)]]

#         # make graph array into incidence matrix
#         matrix = [[is_true(edge) for edge in discovered_graph[i:i+m]]
#                   for i in range(0, len(discovered_graph), m)]

#     print()
#     print("valuation_function", valuation_function)
#     print("discovered_graph:", matrix)

#     graph = Graph.Adjacency(matrix, mode="max")

#     return (is_sat == sat, valuation_function, graph, n_int)


def find_valuation_function_with_no_efx(n, m):
    s = Solver()

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

            get_formula_for_one_item_to_one_agent(
                [[a for a in aa] for aa in A], n, m),

            Not(
                get_formula_for_ensuring_efx(
                    [[a for a in aa] for aa in A], V, n, m)
            )
        )
    )
    )

    valuation_function = []
    is_sat = s.check()
    print(is_sat)

    if(is_sat == sat):

        m = s.model()
        print(m)
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



def is_graph_connected(graph):
    B = BoolSort()
    s = Solver()
    NodeSort = Datatype('Node')

    vertices = [v for v in range(graph.vcount())]

    # Setup
    #################
    for vertex in vertices:
        NodeSort.declare(str(vertex))
    NodeSort = NodeSort.create()

    vs = {}
    for vertex in vertices:
        vs[vertex] = getattr(NodeSort, str(vertex))

    EdgeConection = Function('EdgeConection',
                             NodeSort,
                             NodeSort, B)
    TC_EdgeConection = TransitiveClosure(EdgeConection)

    # Make edges go both ways (since they are undirected)
    x, y = Consts("x y", NodeSort)
    s.add(ForAll([x, y], Implies(EdgeConection(x, y), EdgeConection(y, x))))

    # Give information about the given graph
    ###########################################
    adjacency = graph.get_adjacency()

    for vertex1 in vertices:
        for vertex2 in vertices:
            if vertex1 == vertex2:
                continue
            # Say where there is and where there is not edges in the graph
            s.add(If(adjacency[vertex1][vertex2] == 1,
                     EdgeConection(vs[vertex1], vs[vertex2]),
                     Not(EdgeConection(vs[vertex1], vs[vertex2]))
                     ))

    # Check connectivity for whole graph by looking at the transitive closure
    for vertex1 in vertices:
        for vertex2 in vertices:
            if vertex1 == vertex2:
                continue
            s.add(TC_EdgeConection(vs[vertex1], vs[vertex2]) == True)

    return s.check() == sat


def is_agent_graph_connected(graph, allocated_goods_to_this_agent, item_to_remove):
    s = Solver()
    s.add(is_graph_connected_after_removing_item(
        graph, allocated_goods_to_this_agent, item_to_remove, 0))
    is_sat = s.check()
    print(is_sat)
    return is_sat == sat


def is_graph_connected_after_removing_item(graph, allocated_goods_to_this_agent, item_to_remove, agent):
    formulas = []
    B = BoolSort()
    NodeSort = Datatype('Node'+str(agent)+str(item_to_remove))

    vertices = [v for v in range(graph.vcount())]

    # Setup
    #################
    for vertex in vertices:
        NodeSort.declare(str(vertex))
    NodeSort = NodeSort.create()

    vs = {}
    for vertex in vertices:
        vs[vertex] = getattr(NodeSort, str(vertex))

    EdgeConection = Function('EdgeConection'+str(agent)+str(item_to_remove),
                             NodeSort,
                             NodeSort, B)
    TC_EdgeConection = TransitiveClosure(EdgeConection)

    # Make edges go both ways (since they are undirected)
    x = Const("x"+str(agent)+str(item_to_remove), NodeSort)
    y = Const("y"+str(agent)+str(item_to_remove), NodeSort)
    formulas.append(ForAll([x, y], Implies(
        EdgeConection(x, y), EdgeConection(y, x))))

    # Give information about the given graph
    ###########################################
    adjacency = graph.get_adjacency()

    for vertex1 in vertices:
        for vertex2 in vertices:
            if vertex1 == vertex2:
                continue

            # Say where there is and where there is not edges in the graph
            formulas.append(If(And(
                allocated_goods_to_this_agent[vertex1],
                allocated_goods_to_this_agent[vertex2],
                vertex1 != item_to_remove,
                vertex2 != item_to_remove,
                adjacency[vertex1][vertex2] == 1),
                EdgeConection(vs[vertex1], vs[vertex2]),
                Not(EdgeConection(vs[vertex1], vs[vertex2]))
            ))

    # Check connectivity for whole graph by looking at the transitive closure
    for vertex1 in vertices:
        for vertex2 in vertices:
            if vertex1 == vertex2:
                continue
            if vertex1 == item_to_remove or vertex2 == item_to_remove:
                continue

            formulas.append(If(And(
                allocated_goods_to_this_agent[vertex1],
                allocated_goods_to_this_agent[vertex2]),
                TC_EdgeConection(
                vs[vertex1], vs[vertex2]) == True,
                True))

    return And(formulas)

# TODO tror ikke dene er helt riktig! Hva om grafen er splittet? Nå sjekker den vel bare om hver node henger sammen med en annen node, men ikke om alle nodene i bundelen hengge sammen.


def is_ef1_with_connectivity_possible(n, m, V, G_graph):
    s = Solver()

    # Make sure that the number of nodes in the graph matches the number of items and the valuation function
    assert m == G_graph.vcount(), "The number of items do not match the size of the graph"
    assert m == V[0].size, "The number of items do not match the valuation function"

    # A  keeps track of the allocated items
    A = [[Bool("a_%s_%s" % (i+1, j+1)) for j in range(m)]
         for i in range(n)]

    G = G_graph  # .get_adjacency()

    s.add(get_formula_for_one_item_to_one_agent(A, n, m))
    s.add(get_formula_for_ensuring_ef1_outer_good(A, G, V, n, m))
    s.add(get_edge_connectivity_formulas(G, A, n))

    is_sat = s.check()
    print(is_sat)
    return is_sat == sat


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
    s.set("timeout", 8600000)  # TODO increase timeout
    

    valuation_function = []
    is_sat = s.check()
    print(is_sat)

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



def find_valuation_function_with_no_ef1_equal_valuation_functions(n, m, G):
    s = Solver()

    # Make sure that the number of nodes in the graph matches the number of items and the valuation function
    assert m == G.vcount(), "The number of items do not match the size of the graph"

    # A  keeps track of the allocated items
    A = [[Bool("a_%s_%s" % (i+1, j+1)) for j in range(m)]
         for i in range(n)]

    # Valuations
    V = [Int("v_item%s" % (j)) for j in range(m)]
         

    # Make sure all values are non-negative
    for i in range(m):
        s.add(V[i] >= 0)

    s.add(ForAll(
        [a for aa in A for a in aa],
        Implies(

            And(
                get_formula_for_one_item_to_one_agent(
                    [[a for a in aa] for aa in A], n, m),
                get_edge_conflicts(
                    G, [[a for a in aa] for aa in A], n)),

            Not(
                get_formula_for_ensuring_ef1_equal_valuation_functions(
                    [[a for a in aa] for aa in A], V, n, m)
            )
        )
    )
    )
    s.set("timeout", 8600000)  # TODO increase timeout
    

    valuation_function = []
    is_sat = s.check()
    print(is_sat)

    if(is_sat == sat):

        model = s.model()
        print(model)
        tuples = sorted([(d, model[d]) for d in model], key=lambda x: str(x[0]))
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
    s.set("timeout", 7200000)
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

    valuation_function = []
    discovered_graph = []
    is_sat = s.check()
    print(is_sat)
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

    # graph = Graph.Adjacency(matrix, mode="max")

    return (is_sat, valuation_function, matrix)


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

    s.add(n < m -1)
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

    is_sat = s.check()
    print(is_sat)
    
    valuation_function = []
    discovered_graph = []
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
         for i in range(m-1)] # todo GJØRE OM TIL M-1?

    # Valuations
    V = [[Int("v_agent%s_item%s" % (i, j)) for j in range(m)]
         for i in range(m-1)]

    # Adjacency matrux for conlfict graph
    G = [[Bool("g_row%s_col%s" % (i, j)) for j in range(m)] 
         for i in range(m)]

    # Forece the values to be binary
    for i in range(m-1):
        for j in range(m):
            s.add(Or(V[i][j] == 1, V[i][j] == 0))

    s.add(n < m -1)
    s.add(n > 2)
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
                # get_formula_for_ensuring_ef1_unknown_agents(
                get_formula_for_ensuring_ef1_unknown_agents_boolean_values(
                    [[a for a in aa] for aa in A], V, n, m)
            )
        )
    ))

    valuation_function = []
    discovered_graph = []
    is_sat = s.check()
    print(is_sat)
    matrix = [[]]
    n_int = 0
    graph = None
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

        graph = Graph.Adjacency(matrix, mode="max")
    print()
    print("valuation_function", valuation_function)
    print("discovered_graph:", matrix)


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

    s.add(n < m -1)
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

    valuation_function = []
    discovered_graph = []
    is_sat = s.check()
    print(is_sat)
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

    s.add(n < m -1)
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

    valuation_function = []
    is_sat = s.check()
    print(is_sat)
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

    s.add(n < m - 1)
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

    valuation_function = []
    discovered_graph = []
    is_sat = s.check()
    print(is_sat)
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
    is_sat = s.check()
    print(is_sat)
    if(is_sat == sat):
        print(s.model())
    return True

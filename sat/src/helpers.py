from z3 import *

### Helper functions ###
########################

def max_in_product_array_bool(d_i_j, v_i, m):
    max_value = 0

    for item in range(0, m):
        
        agent_has_this_item = d_i_j[item]
        value_of_item = v_i[item]
        
        max_value = If(
            And(
                agent_has_this_item,
                value_of_item > max_value
            ),
            value_of_item,
            max_value)
        
    return max_value


def get_edge_connectivity_formulas(G, A, n):

    formulas = []
    for agent in range(n):
        # -1 will make the function not care about the missing item
        formulas.append(
            is_graph_connected_after_removing_item(G, A[agent], -1, agent))

    return And(formulas)

def get_formula_for_ensuring_ef1_outer_good(A, G, V, n, m):
    formulas = []

    for i in range(n):
        for j in range(n):

            if i == j:
                continue

            # Check that there is no envy once an item is possibly dropped

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


def get_edge_conflicts(G, A, n):
    conflicts = []
    # Enforce the conflicts from the graph
    for e in G.get_edgelist():
        g = e[0]
        h = e[1]

        for i in range(n):
            # Make sure that a single agent only can get one of two conflicting items
            conflicts.append(Not(And(A[i][g], A[i][h])))

    return And([conflict for conflict in conflicts])

def get_edge_conflicts_int(G, A, n):
    conflicts = []
    # Enforce the conflicts from the graph
    for e in G.get_edgelist():
        g = e[0]
        h = e[1]

        for i in range(n):
            # Make sure that a single agent only can get one of two conflicting items
            conflicts.append((A[i][g] + A[i][h]) <= 1)

    return And([conflict for conflict in conflicts])


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
                        Not(And(A[i][row], A[i][col])),

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
            for i in range(m-1):
                formulas.append(
                    If(
                        # If: there is an edge
                        G[row][col],

                        # Then: make sure that the same agent dont get the items in both ends
                        Not(And(A[i][row], A[i][col])),

                        # Else: Do whatever
                        True
                    )
                )

    return And(formulas)


def get_edge_conflicts_path_array(A, m, n):
    formulas = []

    for row in range(m):
        for col in range(m-1):
            # It is not allowed to have two items next to each other in the path
            formulas.append(
                Not(
                    And(
                        A[row][col], A[row][col+1]
                    ))

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


# OBSBS does not gurantee a path!! But it does allow forpaths
def get_formula_for_path(G, m):
    formulas = []

    num_total_edges = 0

    for i in range(m):
        num_edges_for_item = 0
        for j in range(m):
            if j > i:
                num_edges_for_item = num_edges_for_item + \
                    If(G[j][i], 1, 0)
            else:
                num_edges_for_item = num_edges_for_item + \
                    If(G[i][j], 1, 0)

        formulas.append(Or(num_edges_for_item == 1, num_edges_for_item == 2))
        num_total_edges = num_total_edges + num_edges_for_item
    # Each edge is counted twice because we look at the perspective of each node
    num_total_edges = num_total_edges / 2
    formulas.append(num_total_edges == m-1)
    return And(formulas)


def get_total_edges(G, m):

    num_total_edges = 0

    for i in range(m):
        num_edges_for_item = 0
        for j in range(m):
            if j > i:
                num_edges_for_item = num_edges_for_item + \
                    If(G[j][i], 1, 0)
            else:
                num_edges_for_item = num_edges_for_item + \
                    If(G[i][j], 1, 0)

        num_total_edges = num_total_edges + num_edges_for_item
    return num_total_edges / 2


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

    return And(formulas)


def get_formula_for_one_item_to_one_agent(A, n, m):
    formulas = []
    # Each item allocated to at exactly one agent
    for g in range(m):
        formulas.append(PbEq([(A[i][g], 1) for i in range(n)], 1))

    return And(formulas)

def get_formula_for_one_item_to_one_agent_int(A, n, m):
    formulas = []
    # Each item allocated to at exactly one agent
    for g in range(m):
        formulas.append(PbEq([(A[i][g], 1) for i in range(n)], 1))

    return And(formulas)


def get_formula_for_one_item_to_one_agent_uknown_agents(A, n, m):
    formulas = []
    # Each item allocated to at exactly one agent
    for g in range(m):
        formulas.append(PbEq(
            [(And(i < n, A[i][g]), 1)
             for i in range(m-1)],
            1))

    return And(formulas)

def get_formula_for_ensuring_ef1(A, V, n, m):
    formulas = []

    for i in range(n):
        for j in range(n):

            if i == j:
                continue

             # Check that there is no envy once an item is possibly dropped
            formulas.append(Sum([If(A[i][g], V[i][g], 0) for g in range(m)]) >=
                            Sum([If(A[j][g], V[i][g], 0) for g in range(m)]) - max_in_product_array_bool(A[j], V[i], m))

    return And(formulas)


def get_formula_for_ensuring_ef1_equal_valuation_functions(A, V, n, m):
    formulas = []

    for i in range(n):
        for j in range(n):

            if i == j:
                continue

             # Check that there is no envy once an item is possibly dropped
            formulas.append(Sum([If(A[i][g], V[g], 0) for g in range(m)]) >=
                            Sum([If(A[j][g], V[g], 0) for g in range(m)]) - max_in_product_array_bool(A[j], V, m))

    return And(formulas)


def get_formula_for_ensuring_ef1_unknown_agents(A, V, n, m):
    formulas = []

    for i in range(m):
        for j in range(m):

            if i == j:
                continue
            
            # # Check that there is no envy once an item is possibly dropped
            formulas.append(If(And(i < n, j < n), Sum([If(A[i][g], V[i][g], 0) for g in range(m)]) >=
                            Sum([If(A[j][g], V[i][g], 0) for g in range(m)]) - max_in_product_array_bool(A[j], V[i], m), True))# TODO kan vel gjøres  enkelere med binære verdier?

    return And(formulas)

def get_formula_for_ensuring_ef1_unknown_agents_boolean_values(A, V, n, m):
    formulas = []

    for i in range(m-1):
        for j in range(m-1):

            if i == j:
                continue
            
            # # Check that there is no envy once an item is possibly dropped
            formulas.append(If(And(i < n, j < n), Sum([If(A[i][g], V[i][g], 0) for g in range(m)]) >= # TODO teste med å gjøre v boolsk og snde det sammen med A?
                            Sum([If(A[j][g], V[i][g], 0) for g in range(m)]) - 1, True))# TODO kan vel gjøres  enkelere med binære verdier?

    return And(formulas)


def min_in_product_array_bool(d_i_j, v_i,m):

    v_i = [If(a, b,b) for a, b in zip(d_i_j, v_i)] # TODO: bedre måte å unngå numpy?

    min_v = -10
    # print(type(min_v), type(v_i[0]))
    for i in range(m):
        min_v = If(
                    And(d_i_j[i],
                        Or(
                            min_v < 0,
                            v_i[i] < min_v),
                        v_i[i] > 0
                        )
                        ,
                    v_i[i], 
                    min_v
                ) # TODO legge til v > 0 for å ikke bruke streng efx?
    min_v = If(min_v< 0,0,min_v) # The value will still be -10 if no items are allocated to the agent
    return min_v


def get_formula_for_ensuring_efx(A, V, n, m):
    formulas = []

    for i in range(n):
        for j in range(n):

            if i == j:
                continue

            # Check that there is no envy once an item is possibly dropped
            formulas.append(Sum([If(A[i][g], V[i][g] , 0) for g in range(m)]) >=
                            Sum([If(A[j][g], V[i][g], 0) for g in range(m)]) - min_in_product_array_bool(A[j], V[i],m))

    return And(formulas)


def get_formula_for_ensuring_efx_unknown_agents(A, V, n, m):
    formulas = []

    for i in range(m-1):
        for j in range(m-1):

            if i == j:
                continue

            # Check that there is no envy once an item is possibly dropped
            formulas.append(If(And(i < n, j < n),Sum([If(A[i][g], V[i][g], 0) for g in range(m)]) >=
                            Sum([If(A[j][g], V[i][g], 0) for g in range(m)]) - min_in_product_array_bool(A[j], V[i],m),True))

    return And(formulas)



def get_mms_for_this_agent_int(this_agent, n, m, V, G):
    formulas = []
    # TODO hvilket tall? og vanlig variabel eller z3 variabel?
    mms = Int("mms_%s" % this_agent)

    # A  keeps track of the allocated items
    A = [[Int("a_%s_%s_mms_calculation_%s" % (i+1, j+1, this_agent)) for j in range(m)]  # TODO må jeg ha denne? virker som det blir mye vfor z3 å tenke på
         for i in range(n)]

    for i in range(n):
        for j in range(m):
            formulas.append(Or(A[i][j] == 0, A[i][j] == 1))

    for i in range(n):
        formulas.append(mms <= Sum(
            [

                # If(A[i][g],
                V[this_agent][g] * A[i][g]
                # , 0)
                for g in range(m)

            ]))  # look at this_agents values, because this is from her point of view

    opt = Optimize()
    opt.set("timeout", 20000)  # TODO increase timeout
    opt.add(And(formulas))
    opt.add(get_formula_for_one_item_to_one_agent_int(A, n, m))
    opt.add(get_edge_conflicts_int(G, A, n))
    opt.maximize(mms)
    res = opt.check()
    if res == unknown:
        print("Unknown, reason: %s" % opt.reason_unknown())
    mod = opt.model()
    # print(mod)
    print(mod[mms])

    return mod[mms]


def get_mms_for_this_agent(this_agent, n, m, V, G):
    set_param("parallel.enable", True)
    opt = Optimize()
    opt.set("timeout", 300000)  # TODO increase timeout

    # TODO hvilket tall? og vanlig variabel eller z3 variabel?
    mms = Int("mms_%s" % this_agent)

    # A  keeps track of the allocated items
    A = [[Bool("a_%s_%s_mms_calculation_%s" % (i+1, j+1, this_agent)) for j in range(m)]  # TODO må jeg ha denne? virker som det blir mye vfor z3 å tenke på
         for i in range(n)]
    # opt.add(mms < np.sum(V[this_agent]))

    for i in range(n):
        other_agents_bundle_value = Int("other_agents_bundle_value_%s" % i)
        opt.add(other_agents_bundle_value == Sum(
            [
                If(A[i][g],
                   V[this_agent][g], 0)
                for g in range(m)
            ]))
        # look at this_agents values, because this is from her point of view
        opt.add(mms <= other_agents_bundle_value)

    opt.add(get_formula_for_one_item_to_one_agent(A, n, m))
    if G != None:
        opt.add(get_edge_conflicts(G, A, n))
    opt.maximize(mms)
    res = opt.check()
    print("mms this agent:", res)
    if res == unknown:
        print("Unknown, reason: %s" % opt.reason_unknown())
        return 0, res
    mod = opt.model()
    # print(mod)
    print(mod[mms])

    return mod[mms], res


def this_agents_bundle_value_from_their_pov(A, this_agent, m, V):
    return Sum([If(A[this_agent][g],
                   V[this_agent][g], 0) for g in range(m)])


def this_agents_bundle_is_the_least_valuable(A, this_agent, n, m, V):
    formulas = []

    this_agents_bundle_value = this_agents_bundle_value_from_their_pov(
        A, this_agent, m, V)

    for i in range(n):
        formulas.append(this_agents_bundle_value <= Sum([If(A[i][g],
                                                            V[this_agent][g], 0) for g in range(m)]))

    return And(formulas)


def get_mms_for_this_agent_quantified(this_agent, n, m, V, G):
    s = Solver()
    # s.set("timeout", 100000)  # TODO increase timeout

    # TODO hvilket tall? og vanlig variabel eller z3 variabel?
    mms = Int("mms_%s" % this_agent)

    # A  keeps track of the allocated items
    A = [[Bool("a_%s_%s_mms_calculation_%s" % (i+1, j+1, this_agent)) for j in range(m)]  # TODO må jeg ha denne? virker som det blir mye vfor z3 å tenke på
         for i in range(n)]
    # opt.add(mms < np.sum(V[this_agent]))
    # for all allocations, mms is greater than or equal to the worst bundle

    s.add(ForAll(
        [a for aa in A for a in aa],
        Implies(

            And(
                get_formula_for_one_item_to_one_agent(
                    [[a for a in aa] for aa in A], n, m),
                get_edge_conflicts(G,
                                   [[a for a in aa] for aa in A], n),
                this_agents_bundle_is_the_least_valuable(
                    [[a for a in aa] for aa in A], this_agent, n, m, V)
            ),
            mms >= this_agents_bundle_value_from_their_pov(
                [[a for a in aa] for aa in A], this_agent, m, V)


        )
    ))
    # opt.maximize(mms)
    print("befrore check")
    res = s.check()
    if res == unknown:
        print("Unknown, reason: %s" % s.reason_unknown())
    mod = s.model()
    # print(mod)
    print(mod[mms])

    return mod[mms]

# TODO sjekk med eksemplene der vi vet at det ikke er mulig at alle får sin mms


def get_mms_for_this_agent_manual_optimization(this_agent, n, m, V, G):
    # opt = Optimize()
    # opt.set("timeout", 100000)  # TODO increase timeout
    s = Solver()

    start_mms = 163

    # TODO hvilket tall? og vanlig variabel eller z3 variabel?
    mms = Int("mms_%s" % this_agent)

    # A  keeps track of the allocated items
    A = [[Bool("a_%s_%s_mms_calculation_%s" % (i+1, j+1, this_agent)) for j in range(m)]  # TODO må jeg ha denne? virker som det blir mye vfor z3 å tenke på
         for i in range(n)]

    # beginning optimazation value
    s.add(mms == start_mms)

    for i in range(n):
        s.add(mms <= Sum(
            [

                If(A[i][g],
                   V[this_agent][g], 0)
                for g in range(m)

            ]))  # look at this_agents values, because this is from her point of view

    s.add(get_formula_for_one_item_to_one_agent(A, n, m))
    s.add(get_edge_conflicts(G, A, n))
    res = s.check()
    best = s.model()
    print(res)
    # while res == sat:
    mod = s.model()
    print(mod[mms])

    res = s.check()

    # print(mod)
    print(mod[mms])

    return best[mms]

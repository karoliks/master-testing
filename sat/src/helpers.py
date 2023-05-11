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

# TODO brukes egentlig denne?
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
            for i in range(m):
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


# TODO er ikke sikker på om dette garanterer sti, men det virker lovende
# TODO OBSBS does not gurantee a path!! But it does allow forpaths
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

# TODO heller bruke sum her siden det erint?
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
            [(And(i < n, If(A[i][g])), 1)
             for i in range(m)],
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


def get_formula_for_ensuring_ef1_unknown_agents(A, V, n, m):
    formulas = []

    for i in range(m):
        for j in range(m):

            if i == j:
                continue

            # Check that there is no envy once an item is possibly dropped
            formulas.append(If(And(i < n, j < n), Sum([V[i][g] * If(A[i][g], 1, 0) for g in range(m)]) >=
                            Sum([V[i][g] * If(A[j][g], 1, 0) for g in range(m)]) - max_in_product_array_bool(A[j], V[i], m), True))

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
    opt.set("timeout", 100000)  # TODO increase timeout

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
    opt.add(get_edge_conflicts(G, A, n))
    opt.maximize(mms)
    res = opt.check()
    print("mms this agent:", res)
    if res == unknown:
        print("Unknown, reason: %s" % opt.reason_unknown())
    mod = opt.model()
    # print(mod)
    print(mod[mms])

    return mod[mms]


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

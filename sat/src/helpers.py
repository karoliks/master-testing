from z3 import *

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
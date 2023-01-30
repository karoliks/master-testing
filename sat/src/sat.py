from z3 import *
import numpy as np


### Helper functions ###
########################

def max_in_product_array(d_i_j, v_i):
    product = [a*b for a, b in zip(d_i_j, v_i)]
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


def get_formula_for_correct_removing_of_items(A, D, n, m):
    formulas = []

    for i in range(n):
        for j in range(n):
            # Make sure that when checking for EF1, each agent is only allowed togive away one item to make another agent not jealous
            # TODO double check that this is correct order
            formulas.append(Sum([If(D[i][j][g], 1, 0) for g in range(n)]) == 1)

            for g in range(m):
                # Make sure only items that are actuallallocated to the agent in question, is dropped
                formulas.append(If(D[i][j][g], 1, 0) <= If(A[j][g], 1, 0))

    return And([formula for formula in formulas])


def get_formula_for_one_item_for_each_agent(A, n, m):
    formulas = []
    # Each item allocated to at exactly one agent
    for g in range(m):
        # TODO double check that this is correct order
        # var <= her. skal det ikke være ==?
        formulas.append(Sum([If(A[i][g], 1, 0) for i in range(n)]) == 1)

    return And([formula for formula in formulas])


def get_formula_for_ensuring_ef1(A, D, V, n, m):
    formulas = []

    for i in range(n):
        for j in range(n):

            if i == j:
                continue

            # Check that there is no envy once an item is possibly dropped
            formulas.append(Sum([If(A[i][g], 1, 0) * V[i][g] for g in range(m)]) >=
                            Sum([V[j][g] * (If(A[j][g], 1, 0) - If(D[j][i][g], 1, 0)) for g in range(m)]))  # TODO dobbelsjekk omindeksenei D er riktige

    return And([formula for formula in formulas])


################################################################


def is_ef1_possible(n, m, V):
    s = Solver()

    # A  keeps track of the allocated items
    A = [[Bool("a_%s_%s" % (i+1, j+1)) for j in range(m)]
         for i in range(n)]

    # D keepstrack of items that are dropped when checking for Ef1
    D = [[[Bool("d_%s_%s_%s" % (i+1, j+1, k+1)) for i in range(m)]
          for j in range(n)] for k in range(n)]

    s.add(get_formula_for_one_item_for_each_agent(A, n, m))
    s.add(get_formula_for_correct_removing_of_items(A, D, n, m))
    s.add(get_formula_for_ensuring_ef1(A, D, V, n, m))

    return s.check() == sat


def is_ef1_with_conflicts_possible(n, m, V, G):
    s = Solver()

    # Make sure that the number of nodes in the graph matches the number of items and the valuation function
    assert m == G.vcount(), "The number of items do not match the size of the graph"
    assert m == V[0].size, "The number of items do not match the valuation function"

    # A  keeps track of the allocated items
    A = [[Bool("a_%s_%s" % (i, j)) for j in range(m)]
         for i in range(n)]

    # D keepstrack of items that are dropped when checking for Ef1
    D = [[[Bool("d_%s_%s_%s" % (i, j, k)) for i in range(m)]
          for j in range(n)] for k in range(n)]

    s.add(get_formula_for_one_item_for_each_agent(A, n, m))
    s.add(get_formula_for_correct_removing_of_items(A, D, n, m))
    s.add(get_formula_for_ensuring_ef1(A, D, V, n, m))
    s.add(get_edge_conflicts(G, A, n))

    return s.check() == sat



    for i in range(n):
        for j in range(n):
            # Make sure that when checking for EF1, each agent is only allowed togive away one item to make another agent not jealous
            # TODO double check that this is correct order
            s.add(Sum([If(D[i][j][g], 1, 0) for g in range(n)]) == 1)

            for g in range(m):
                # Make sure only items that are actuallallocated to the agent in question, is dropped
                s.add(If(D[i][j][g], 1, 0) <= If(A[j][g], 1, 0))

    for i in range(n):
        for j in range(n):

            if i == j:
                continue
            # Check that there is no envy once an item is possibly dropped
            s.add(Sum([V[i][g] * If(A[i][g], 1, 0)
                       for g in range(m)]) >=
                  Sum([V[i][g] * (If(A[j][g], 1, 0) - If(D[j][i][g], 1, 0))  # TODO dobbelsjekk omindeksenei D er riktige
                       for g in range(m)]))

    s.add(get_edge_conflicts(G, A, n))

    # Check if an EF1 allocation can be found
    return s.check() == sat


        for i in range(n):
            # Make sure that a single agent only can get one of two conflicting items
            s.add((If(A[i][g], 1, 0) + If(A[i][h], 1, 0)) <= 1)

    # Check if an EF1 allocation can be found
    print(s.check())
    if(s.check() == sat):
        print(s.model())

    return s.check() == sat

from z3 import *
import numpy as np


def is_ef1_possible(n, m, V):
    s = Solver()

    # A  keeps track of the allocated items
    A = [[Bool("a_%s_%s" % (i+1, j+1)) for j in range(m)]
         for i in range(n)]

    # D keepstrack of items that are dropped when checking for Ef1
    D = [[[Bool("d_%s_%s_%s" % (i+1, j+1, k+1)) for i in range(m)]
          for j in range(n)] for k in range(n)]

    # Each item allocated to at exactly one agent
    for g in range(m):
        # TODO double check that this is correct order
        s.add(Sum([If(A[i][g], 1, 0) for i in range(n)]) == 1)

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
            s.add(Sum([If(A[i][g], 1, 0) * V[i][g] for g in range(m)]) >= 
                  Sum([V[j][g] * (If(A[j][g], 1, 0) - If(D[j][i][g], 1, 0)) for g in range(m)]))  # TODO dobbelsjekk omindeksenei D er riktige

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

    # Each item allocated to at exactly one agent
    for g in range(m):
        # TODO double check that this is correct order
        # var <= her. skal det ikke være ==?
        s.add(Sum([If(A[i][g], 1, 0) for i in range(n)]) == 1)

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

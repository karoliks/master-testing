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
        s.add(Sum([If(A[i][g], 1, 0) for i in range(n)]) <= 1)

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
                  Sum([V[j][g] * (If(A[j][g], 1, 0) - If(D[i][j][g], 1, 0)) for g in range(m)]))

    print(s.check())
    if(s.check() == sat):
        print(s.model())
        
    return s.check() == sat


# TODO: legge inn konflikter
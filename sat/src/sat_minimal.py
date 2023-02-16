from z3 import *
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


def get_edge_conflicts_list(G, A, n):
    conflicts = []
    # Enforce the conflicts from the graph
    for e in G.get_edgelist():
        g = e[0]
        h = e[1]

        for i in range(n):
            # Make sure that a single agent only can get one of two conflicting items
            conflicts.append((If(A[i*n+g], 1, 0) + If(A[i*n+h], 1, 0)) <= 1)

    return And([conflict for conflict in conflicts])


def get_formula_for_one_item_to_one_agent(A, n, m):
    formulas = []
    # Each item allocated to at exactly one agent
    for g in range(m):
        # TODO double check that this is correct order
        # var <= her. skal det ikke være ==?
        formulas.append(Sum([If(A[i][g], 1, 0) for i in range(n)]) == 1)

    return And([formula for formula in formulas])


def get_formula_for_one_item_to_one_agent_list(A, n, m):
    formulas = []
    # Each item allocated to at exactly one agent
    for g in range(m):
        # TODO double check that this is correct order
        # var <= her. skal det ikke være ==?
        formulas.append(Sum([If(A[i*n+g], 1, 0) for i in range(n)]) == 1)

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


def get_formula_for_ensuring_ef1_list_new(A, V, n, m):
    formulas = []

    for i in range(n):
        for j in range(n):

            if i == j:
                continue

            # Check that there is no envy once an item is possibly dropped
            formulas.append(Sum([V[i][g] * If(A[i*n+g], 1, 0) for g in range(m)]) >=
                            Sum([V[i][g] * If(A[j*n+g], 1, 0) for g in range(m)]) - max_in_product_array_bool(A[j*n:(j+1)*n], V[i]))

    return And([formula for formula in formulas])


def get_formula_for_ensuring_ef1_list(A, D, V, n, m):
    formulas = []

    for i in range(n):
        for j in range(n):

            if i == j:
                continue
            # Check that there is no envy once an item is possibly dropped
            formulas.append(
                Sum([
                    If(A[i*n+g], 1, 0) * V[i][g]
                    for g in range(m)
                ])
                >=
                Sum([
                    V[i][g] * (If(A[j*n+g], 1, 0) - If(D[j][i][g], 1, 0))
                    for g in range(m)]))  # TODO dobbelsjekk omindeksenei D er riktige

    return And([formula for formula in formulas])


def find_valuation_function_with_no_ef1_working(n, m, G):
    s = Solver()

    # A  keeps track of the allocated items
    A = [[Bool("a_%s_%s" % (i+1, j+1)) for j in range(m)]
         for i in range(n)]

    # Valuations
    V = [[Int("v_agent%s_item%s" % (i, j)) for j in range(m)]
         for i in range(n)]

    s.add(Exists(
        [a for aa in A for a in aa],
        And(
            get_formula_for_one_item_to_one_agent(
                [[a for a in aa] for aa in A], n, m),
            get_edge_conflicts(G, [[a for a in aa] for aa in A], n))))  # TODO and de over med dette

    print(s.check())
    valuation_function = []
    is_sat = s.check()
    if(s.check() == sat):
        # print(s.model())
        res = s.model()
        m = s.model()
        tuples = sorted([(d, m[d]) for d in m], key=lambda x: str(x[0]))
        valuation_function = [d[1] for d in tuples]

        counter = 0
        while s.check() == sat and counter < 15:
            counter = counter+1
            print(s.model())
            # prevent next model from using the same assignment as a previous model
            s.add(Or([(v != s.model()[v]) for vv in V for v in vv]))

    print()
    print(valuation_function)
    return (is_sat == sat, valuation_function)


def find_valuation_function_with_no_ef1_not_working(n, m, G):
    s = Solver()

    # A  keeps track of the allocated items
    A = [[Bool("a_%s_%s" % (i+1, j+1)) for j in range(m)]
         for i in range(n)]

    # Valuations
    V = [[Int("v_agent%s_item%s" % (i, j)) for j in range(m)]
         for i in range(n)]

    s.add(Exists(
        [a for aa in A for a in aa],
        And(
            get_formula_for_one_item_to_one_agent_list(
                [a for aa in A for a in aa], n, m),
            get_edge_conflicts_list(G, [a for aa in A for a in aa], n))))

    print(s.check())
    valuation_function = []
    is_sat = s.check()
    if(s.check() == sat):
        # print(s.model())
        res = s.model()
        m = s.model()
        tuples = sorted([(d, m[d]) for d in m], key=lambda x: str(x[0]))
        valuation_function = [d[1] for d in tuples]

        counter = 0
        while s.check() == sat and counter < 15:
            counter = counter+1
            print(s.model())
            # prevent next model from using the same assignment as a previous model
            s.add(Or([(v != s.model()[v]) for vv in V for v in vv]))

        print()
        print(valuation_function)
    return (is_sat == sat, valuation_function)

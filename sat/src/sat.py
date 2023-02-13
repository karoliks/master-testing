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


def get_formula_for_correct_removing_of_items(A, D, n, m):
    formulas = []

    for i in range(n):
        for j in range(n):
            # Make sure that when checking for EF1, each agent is only allowed togive away one item to make another agent not jealous
            # TODO double check that this is correct order
            formulas.append(Sum([If(D[i][j][g], 1, 0) for g in range(m)]) <= 1)

            for g in range(m):
                # Make sure only items that are actuallallocated to the agent in question, is dropped
                formulas.append(If(D[i][j][g], 1, 0) <= If(A[j][g], 1, 0))

    return And([formula for formula in formulas])


def get_formula_for_correct_removing_of_items_list(A, D, n, m):
    formulas = []

    for i in range(n):
        for j in range(n):
            # Make sure that when checking for EF1, each agent is only allowed togive away one item to make another agent not jealous
            # TODO double check that this is correct order
            formulas.append(Sum([If(D[i*n+j*n+g], 1, 0)
                            for g in range(m)]) <= 1)

            for g in range(m):
                # Make sure only items that are actuallallocated to the agent in question, is dropped
                formulas.append(If(D[i*n+j*n+g], 1, 0) <= If(A[j*n+g], 1, 0))

    return And([formula for formula in formulas])


def get_formula_for_one_item_for_each_agent(A, n, m):
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


def get_formula_for_ensuring_ef1_old(A, D, V, n, m):
    formulas = []

    for i in range(n):
        for j in range(n):

            if i == j:
                continue

            # Check that there is no envy once an item is possibly dropped
            formulas.append(Sum([If(A[i][g], 1, 0) * V[i][g] for g in range(m)]) >=
                            Sum([V[i][g] * (If(A[j][g], 1, 0) - If(D[j][i][g], 1, 0)) for g in range(m)]))  # TODO dobbelsjekk omindeksenei D er riktige

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


def get_formula_for_ensuring_ef1_list_list(A, D, V, n, m):
    formulas = []

    for i in range(n):
        for j in range(m):
            formulas.append(V[i*n+j] >= 0)

    for i in range(n):
        for j in range(n):

            if i == j:
                continue
            # Check that there is no envy once an item is possibly dropped
            formulas.append(
                Sum([
                    If(A[i*n+g], 1, 0) * V[i*n+g]
                    for g in range(m)
                ])
                >=

                Sum([
                    V[i*n+g] * (If(A[j*n+g], 1, 0) - If(D[j*n+i*n+g], 1, 0))
                    for g in range(m)]))  # TODO dobbelsjekk omindeksenei D er riktige

    return And([formula for formula in formulas])


def get_formula_for_ensuring_ef1_list_old(A, D, V, n, m):
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
                    V[i][g] * (If(A[j*n+g], 1, 0) - If(D[j*n+i*n+g], 1, 0))
                    for g in range(m)]))  # TODO dobbelsjekk omindeksenei D er riktige

    return And([formula for formula in formulas])


def get_formula_for_ensuring_at_least_one_jealous_agent_list(A, D, V, n, m):
    a_jalous_agent = []
    for i in range(n):
        for j in range(n):

            if i == j:
                continue
            # Check that there is at least one evious agent once an item is possibly dropped
            a_jalous_agent.append(Sum([V[i][g] * If(A[i*n+g], 1, 0)
                                       for g in range(m)]) <
                                  Sum([V[i][g] * If(A[j*n+g], 1, 0)   # må fåf den til å skjønne at den beste skal legges bort
                                       # TODO dobbelsjekk om indeksene i D er riktige
                                       for g in range(m)]) - max_in_product_array(A[j], V[i]))  # dette er sant når en agent oppfatter at verdien av sin egen bundle er mindre enn en annen sin
            #    for g in range(m)]) - max_in_product_array(D[i][j], V[i]))  # dette er sant når en agent oppfatter at verdien av sin egen bundle er mindre enn en annen sin

    return Or([a_jalous_agent[i] for i in range(len(a_jalous_agent))])


def get_formula_for_ensuring_at_least_one_jealous_agent(A, D, V, n, m):
    a_jalous_agent = []
    DroppedItems = [[[Bool("dI_%s_%s_%s" % (i, j, k)) for i in range(m)]
                     for j in range(n)] for k in range(n)]

    for i in range(n):
        for j in range(n):

            if i == j:
                continue
            # Check that there is at least one evious agent once an item is possibly dropped
            a_jalous_agent.append(Sum([V[i][g] * If(A[i][g], 1, 0)
                                       for g in range(m)]) <
                                  Sum([V[i][g] * If(A[j][g], 1, 0)   # må fåf den til å skjønne at den beste skal legges bort
                                       # TODO dobbelsjekk om indeksene i D er riktige
                                       for g in range(m)]) - max_in_product_array(A[j], V[i]))  # dette er sant når en agent oppfatter at verdien av sin egen bundle er mindre enn en annen sin

    return Or([a_jalous_agent[i] for i in range(len(a_jalous_agent))])


def more_than_100(V, n):
    return And(V[1][2] > 100)


def less_than_100(V, n):
    return And(V[1][2] < 100)
################################################################


def is_ef1_possible(n, m, V):
    s = Solver()

    # A  keeps track of the allocated items
    A = [[Bool("a_%s_%s" % (i+1, j+1)) for j in range(m)]
         for i in range(n)]

    # D keepstrack of items that are dropped when checking for Ef1
    D = [[[Bool("d_%s_%s_%s" % (k+1, j+1, i+1)) for i in range(m)]
          for j in range(n)] for k in range(n)]

    s.add(get_formula_for_one_item_for_each_agent(A, n, m))
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

    # D keepstrack of items that are dropped when checking for Ef1
    D = [[[Bool("d_%s_%s_%s" % (k+1, j+1, i+1)) for i in range(m)]
          for j in range(n)] for k in range(n)]

    s.add(get_formula_for_one_item_for_each_agent(A, n, m))
    s.add(get_formula_for_correct_removing_of_items(A, D, n, m))
    s.add(get_formula_for_ensuring_ef1(A, V, n, m))
    s.add(get_edge_conflicts(G, A, n))
    
    # s.add(ForAll(
    #     [a for aa in A for a in aa] + [d for ddd in D for dd in ddd for d in dd], 
    #            Implies(
    #             And(
    #                 get_formula_for_one_item_for_each_agent_list([a for aa in A for a in aa], n, m),
    #                 get_edge_conflicts_list(G, [a for aa in A for a in aa], n),
    #                 get_formula_for_correct_removing_of_items_list([a for aa in A for a in aa], [d for ddd in D for dd in ddd for d in dd], n, m)),
                
    #                 Not(get_formula_for_ensuring_ef1_list_old([a for aa in A for a in aa],  [d for ddd in D for dd in ddd for d in dd], V, n, m),
    #            )))) # TODO and de over med dette
    # s.add(
    #     ForAll( 
    #            [a for aa in A for a in aa] + [d for ddd in D for dd in ddd for d in dd], 
    #            Implies(True,Not(get_formula_for_ensuring_ef1_list_old([a for aa in A for a in aa],  [d for ddd in D for dd in ddd for d in dd], V, n, m)))))
                    
    return s.check() == sat


def find_valuation_function_with_no_ef1(n, m, G):
    s = Solver()

    # Make sure that the number of nodes in the graph matches the number of items and the valuation function
    assert m == G.vcount(), "The number of items do not match the size of the graph"

    # A  keeps track of the allocated items
    A = [[Bool("a_%s_%s" % (i+1, j+1)) for j in range(m)]
         for i in range(n)]

    # D keepstrack of items that are dropped when checking for Ef1
    D = [[[Bool("d_%s_%s_%s" % (k+1, j+1, i+1)) for i in range(m)]
          for j in range(n)] for k in range(n)]

    # Valuations
    V = [[Int("v_agent%s_item%s" % (i, j)) for j in range(m)]
         for i in range(n)]

    # V = np.array([[0., 0., 0.], [0., 0., 0.]])

    # Make sure all values are non-negative
    for i in range(n):
        for j in range(m):
            s.add(V[i][j] >= 0)
    # for i in range(n):
    #     s.add(Sum(V[i]) > 0)
        # s.add(Sum([v for vv in V for v in vv]) > 4)
    # s.add(get_formula_for_one_item_for_each_agent(A, n, m))
    # s.add(get_formula_for_correct_removing_of_items(A, D, n, m))
    # s.add(get_edge_conflicts(G, A, n))
    # s.add(Not(
    #     Exists([a for aa in A for a in aa] + [d for ddd in D for dd in ddd for d in dd], 
    #            And(
    #             get_formula_for_ensuring_ef1_list_new(
    #                 [a for aa in A for a in aa], V, n, m),
    #             get_formula_for_one_item_for_each_agent_list(
    #                 [a for aa in A for a in aa], n, m),
    #             get_formula_for_correct_removing_of_items_list(
    #                 [a for aa in A for a in aa], [d for ddd in D for dd in ddd for d in dd], n, m),
    #             get_edge_conflicts_list(
    #                 G, [a for aa in A for a in aa], n)
    #            ))))  # TODO and de over med dette
    
    # s.add(ForAll(
    #     [a for aa in A for a in aa] + [d for ddd in D for dd in ddd for d in dd],
    #            Implies(
    #             And(
    #                 get_formula_for_one_item_for_each_agent_list([a for aa in A for a in aa], n, m),
    #                 get_edge_conflicts_list(G, [a for aa in A for a in aa], n),
    #                 get_formula_for_correct_removing_of_items_list([a for aa in A for a in aa], [d for ddd in D for dd in ddd for d in dd], n, m)),

    #                 Not(get_formula_for_ensuring_ef1_list_old([a for aa in A for a in aa],  [d for ddd in D for dd in ddd for d in dd], V, n, m),
    #            )))) # TODO and de over med dette
    # s.add(ForAll(
    # [a for aa in A for a in aa] + [d for ddd in D for dd in ddd for d in dd],

    # And(
    #     get_formula_for_one_item_for_each_agent_list(
    #         [a for aa in A for a in aa], n, m),
    #     get_edge_conflicts_list(G, [a for aa in A for a in aa], n),
    #     get_formula_for_correct_removing_of_items_list([a for aa in A for a in aa], [d for ddd in D for dd in ddd for d in dd], n, m))))

    # TODO nøste D inni ? https://stackoverflow.com/questions/66322928/define-quantifier-variable-issue-in-forall-in-z3py
    # s.add(ForAll(
    #     [a for aa in A for a in aa] + [d for ddd in D for dd in ddd for d in dd],
    #            Implies(
    #             And(
    #                 get_formula_for_one_item_for_each_agent_list([a for aa in A for a in aa], n, m),
    #                 get_edge_conflicts_list(G, [a for aa in A for a in aa], n),
    #             ),
    #         And(
    #                 get_formula_for_correct_removing_of_items_list([a for aa in A for a in aa], [d for ddd in D for dd in ddd for d in dd], n, m)),  # TODO: flytte denne sånn at den er sammen med EFI-kravet?
    #         Not(get_formula_for_ensuring_ef1_list_old([a for aa in A for a in aa],  [
    #             d for ddd in D for dd in ddd for d in dd], V, n, m))

    #     )))  # TODO and de over med dette

    # s.add(ForAll(
    #     [a for aa in A for a in aa] + [d for ddd in D for dd in ddd for d in dd],
    #     Implies(And(
    #             get_formula_for_correct_removing_of_items_list([a for aa in A for a in aa], [
    #                                                            d for ddd in D for dd in ddd for d in dd], n, m),  # TODO: flytte denne sånn at den er sammen med EFI-kravet?
    #             get_formula_for_one_item_for_each_agent_list(
    #                 [a for aa in A for a in aa], n, m),
    #             get_edge_conflicts_list(G, [a for aa in A for a in aa], n)),
    #             Not(get_formula_for_ensuring_ef1_list_old([a for aa in A for a in aa],  [
    #                 d for ddd in D for dd in ddd for d in dd], V, n, m))
    #             )

    # ))  # TODO and de over med dette
    # s.add(ForAll(
    #     [a for aa in A for a in aa] + [d for ddd in D for dd in ddd for d in dd],
    #     Implies(And(
    #             get_formula_for_correct_removing_of_items_list([a for aa in A for a in aa], [
    #                                                            d for ddd in D for dd in ddd for d in dd], n, m),  # TODO: flytte denne sånn at den er sammen med EFI-kravet?
    #             get_formula_for_one_item_for_each_agent_list(
    #                 [a for aa in A for a in aa], n, m),
    #             get_edge_conflicts_list(G, [a for aa in A for a in aa], n)),
    #             And([v == 1 for vv in V for v in vv]))

    # ))  # TODO and de over med dette

    s.add(Exists(
        [a for aa in A for a in aa],
        And(
            # get_formula_for_correct_removing_of_items_list([a for aa in A for a in aa], [
            #     d for ddd in D for dd in ddd for d in dd], n, m),  # TODO: flytte denne sånn at den er sammen med EFI-kravet?
            get_formula_for_one_item_to_one_agent_list(
                [a for aa in A for a in aa], n, m),
            get_edge_conflicts_list(G, [a for aa in A for a in aa], n),
            get_formula_for_ensuring_ef1_list_new([a for aa in A for a in aa],
                                                  V, n, m))

    ))  # TODO and de over med dette
    # s.add(ForAll(
    #     [a for aa in A for a in aa] + [d for ddd in D for dd in ddd for d in dd],
    #     If(And(
    #         get_formula_for_correct_removing_of_items_list([a for aa in A for a in aa], [
    #             d for ddd in D for dd in ddd for d in dd], n, m),  # TODO: flytte denne sånn at den er sammen med EFI-kravet?
    #         get_formula_for_one_item_for_each_agent_list(
    #             [a for aa in A for a in aa], n, m),
    #         get_edge_conflicts_list(G, [a for aa in A for a in aa], n)),
    #        Not(get_formula_for_ensuring_ef1_list_old([a for aa in A for a in aa],  [
    #            d for ddd in D for dd in ddd for d in dd], V, n, m)), False)

    # ))  # TODO and de over med dette

    # s.add(Exists([v for vv in V for v in vv], ForAll(
    #     [a for aa in A for a in aa] + [d for ddd in D for dd in ddd for d in dd],  Implies(And(
    #         get_formula_for_correct_removing_of_items_list([a for aa in A for a in aa], [
    #             d for ddd in D for dd in ddd for d in dd], n, m),  # TODO: flytte denne sånn at den er sammen med EFI-kravet?
    #         get_formula_for_one_item_for_each_agent_list(
    #             [a for aa in A for a in aa], n, m),
    #         get_edge_conflicts_list(G, [a for aa in A for a in aa], n),
    #         get_formula_for_correct_removing_of_items_list([a for aa in A for a in aa], [
    #             d for ddd in D for dd in ddd for d in dd], n, m)),  # TODO: flytte denne sånn at den er sammen med EFI-kravet?

    #         Not(get_formula_for_ensuring_ef1_list_list([a for aa in A for a in aa],  [
    #             d for ddd in D for dd in ddd for d in dd], [v for vv in V for v in vv], n, m)))
    # )))  # TODO and de over med dette

    # s.add(
    #     ForAll(
    #         [a for aa in A for a in aa],
    #         Not(Exists([d for ddd in D for dd in ddd for d in dd],
    #                    Implies(
    #             And(
    #                 get_formula_for_correct_removing_of_items_list([a for aa in A for a in aa], [
    #                     d for ddd in D for dd in ddd for d in dd], n, m),  # TODO: flytte denne sånn at den er sammen med EFI-kravet?
    #                 get_formula_for_one_item_for_each_agent_list(
    #                     [a for aa in A for a in aa], n, m),
    #                 get_edge_conflicts_list(G, [a for aa in A for a in aa], n)),
    #             get_formula_for_ensuring_ef1_list_new(
    #                 [a for aa in A for a in aa],  V, n, m)
    #         )

    #         ))))  # TODO and de over med dette
    
    # s.add(ForAll(
    #     [a for aa in A for a in aa] + [d for ddd in D for dd in ddd for d in dd], 
              
    #             And(True,
    #                 get_formula_for_one_item_for_each_agent_list([a for aa in A for a in aa], n, m),
    #                 get_edge_conflicts_list(G, [a for aa in A for a in aa], n),
    #                 get_formula_for_correct_removing_of_items_list([a for aa in A for a in aa], [d for ddd in D for dd in ddd for d in dd], n, m))        )) # TODO and de over med dette
    
    # s.add(Not(Exists(
    #     [a for aa in A for a in aa] + [d for ddd in D for dd in ddd for d in dd],
    #     Implies(
    #         And(
    #             get_formula_for_one_item_for_each_agent_list(
    #                 [a for aa in A for a in aa], n, m),
    #             get_edge_conflicts_list(G, [a for aa in A for a in aa], n),
    #             get_formula_for_correct_removing_of_items_list([a for aa in A for a in aa], [d for ddd in D for dd in ddd for d in dd], n, m)),
                
    #         get_formula_for_ensuring_ef1_list_old([a for aa in A for a in aa],  [
    #                                               d for ddd in D for dd in ddd for d in dd], V, n, m),
    #     ))))  # TODO and de over med dette
    
    # s.add(ForAll(
    #     [a for aa in A for a in aa] + [d for ddd in D for dd in ddd for d in dd], 
    #            Not(And(
    #             get_formula_for_ensuring_ef1_list_old(
    #                 [a for aa in A for a in aa],  [d for ddd in D for dd in ddd for d in dd], V, n, m),
    #             get_formula_for_one_item_for_each_agent_list(
    #                 [a for aa in A for a in aa], n, m),
    #             get_edge_conflicts_list(
    #                 G, [a for aa in A for a in aa], n),
    #             get_formula_for_correct_removing_of_items_list(
    #                 [a for aa in A for a in aa], [d for ddd in D for dd in ddd for d in dd], n, m),
    #            )))) # TODO and de over med dette

    # s.add(
    #     ForAll([a for aa in A for a in aa] + [d for ddd in D for dd in ddd for d in dd], 
    #            And(
    #            Not(And(get_formula_for_ensuring_ef1_list_new(
    #                [a for aa in A for a in aa], V, n, m)),
    #            get_formula_for_correct_removing_of_items_list(
    #                [a for aa in A for a in aa], [d for ddd in D for dd in ddd for d in dd], n, m)),
    #            get_formula_for_one_item_for_each_agent_list(
    #                [a for aa in A for a in aa], n, m),
    #            get_edge_conflicts_list(
    #                G, [a for aa in A for a in aa], n)
    #            )))  # TODO and de over med dette


    print(s.check())
    valuation_function = []
    is_sat = s.check()
    if(s.check() == sat):
        # print(s.model())
        res = s.model()
        m=s.model()
        tuples = sorted ([(d, m[d]) for d in m], key = lambda x: str(x[0]))
        valuation_function = [d[1] for d in tuples]

        counter = 0
        while s.check() == sat and counter < 15:
            counter = counter+1
            print(s.model())
            # prevent next model from using the same assignment as a previous model
            s.add(Or([(v != s.model()[v]) for vv in V for v in vv]))
        """
        print(res.evaluate(get_formula_for_ensuring_ef1_list_new(
            [a for aa in A for a in aa], V, n, m)))

        for i in range(n):
            for j in range(n):

                if i == j:
                    continue

                print()
                print("========== Agent %s ===========" % i)

                # print(
                #     [res.evaluate(If(a, 1, 0)*b) for a, b in zip(D[j][i], V[i])])
                print("Agent %s sine verdier: %s" % (i,
                                                     [res.eval(a) for a in V[i]]))
                # Check that there is at least one evious agent once an item is possibly dropped
                print("Should agent %s be jealous og agent %s? %s vs %s" % (i, j, res.evaluate(Sum([V[i][g] * If(A[i][g], 1, 0)
                                                                                                   for g in range(m)])),
                                                                            res.evaluate(Sum([V[i][g] * If(A[j][g], 1, 0)   # må fåf den til å skjønne at den beste skal legges bort
                                                                                              # TODO dobbelsjekk om indeksene i D er riktige
                                                                                              for g in range(m)]) - max_in_product_array(A[j], V[i]))  # Skal det geller være allokert i A?
                                                                            #   for g in range(m)]) - max_in_product_array(D[i][j], V[i]))  # Skal det geller være allokert i A?
                                                                            ))  # dette er sant når en agent oppfatter at verdien av sin egen bundle er mindre enn en annen sin
                product = [a*b for a, b in zip(D[i][j], V[i])]
                print("(med D) maks produkt for agent %s mot agent %s ble %s" %
                      (i, j, res.evaluate(max_in_product_array(D[i][j], V[i]))))
                print("(med A) maks produkt for agent %s mot agent %s ble %s" %
                      (i, j, res.evaluate(max_in_product_array(A[j], V[i]))))
                print("produkt-zippet blir det %s mot agent %s ble %s" %
                      (i, j, [res.evaluate(p) for p in product]))"""

    print()
    return (is_sat == sat, valuation_function)


def find_valuation_function_with_no_ef1_v3(n, m, G):
    s = Solver()

    # Make sure that the number of nodes in the graph matches the number of items and the valuation function
    assert m == G.vcount(), "The number of items do not match the size of the graph"

    # A  keeps track of the allocated items
    A = [[Bool("a_%s_%s" % (i, j)) for j in range(m)]
         for i in range(n)]

    # D keepstrack of items that are dropped when checking for Ef1
    D = [[[Bool("d_%s_%s_%s" % (i, j, k)) for i in range(m)]
          for j in range(n)] for k in range(n)]

    # Valuations
    V = [[Int("v_agent%s_item%s" % (i, j)) for j in range(m)]
         for i in range(n)]

    # Make sure all values are non-negative
    for i in range(n):
        for j in range(m):
            s.add(V[i][j] >= 0)

    s.add(get_formula_for_one_item_for_each_agent(A, n, m))
    s.add(get_formula_for_correct_removing_of_items(A, D, n, m))
    s.add(get_edge_conflicts(G, A, n))
    s.add(get_formula_for_ensuring_at_least_one_jealous_agent(A, D, V, n, m))

    # Check if an EF1 allocation can be found
    if(s.check() == sat):
        print(s.model())
        res = s.model()

        for i in range(n):
            for j in range(n):

                if i == j:
                    continue

                # Check that there is at least one evious agent once an item is possibly dropped
                print("Should agent %s be jealous og agent %s? %s vs %s" % (i, j, res.evaluate(Sum([V[i][g] * If(A[i][g], 1, 0)
                                                                                                   for g in range(m)])),
                                                                            res.evaluate(Sum([V[i][g] * If(A[j][g], 1, 0)   # må fåf den til å skjønne at den beste skal legges bort
                                                                                              # TODO dobbelsjekk om indeksene i D er riktige
                                                                                              for g in range(m)]) - max_in_product_array(D[i][j], V[i]))
                                                                            ))  # dette er sant når en agent oppfatter at verdien av sin egen bundle er mindre enn en annen sin
                print("maks produkt for agent %s mot agent %s ble %s" %
                      (i, j, res.evaluate(max_in_product_array(D[i][j], V[i]))))

                print(
                    [res.evaluate(If(a, 1, 0)*b) for a, b in zip(D[j][i], V[i])])
                print("Agent %s sine verdier: %s" % (i,
                                                     [res.eval(a) for a in V[i]]))

        while s.check() == sat:

            print(s.model())
            # prevent next model from using the same assignment as a previous model
            s.add(Or(V != [res.eval(a) for a in V[i]], b != s.model()[b]))

        return s.check() == sat


def find_valuation_function_with_no_ef1_v2(n, m, G):
    s = Solver()

    # Make sure that the number of nodes in the graph matches the number of items and the valuation function
    assert m == G.vcount(), "The number of items do not match the size of the graph"

    # A  keeps track of the allocated items
    A = [[Bool("a_%s_%s" % (i, j)) for j in range(m)]
         for i in range(n)]

    # D keepstrack of items that are dropped when checking for Ef1
    D = [[[Bool("d_%s_%s_%s" % (i, j, k)) for i in range(m)]
          for j in range(n)] for k in range(n)]

    # Valuations
    V = [[Int("v_agent%s_item%s" % (i, j)) for j in range(m)]
         for i in range(n)]

    # Make sure all values are non-negative
    for i in range(n):
        for j in range(m):
            s.add(V[i][j] >= 0)

    o = Optimize()
    epsilon = Real("epsilon")
    o.maximize(epsilon)

    s.add(get_formula_for_one_item_for_each_agent(A, n, m))
    s.add(get_formula_for_correct_removing_of_items(A, D, n, m))

    # Make sure that there are no possible ways to allocate the items in a fair (ef1) manner
    # is_possible = False
    # is_possible_and_list = []
    # for i in range(n):
    #     for j in range(n):

    #         if i == j:
    #             continue
    #         print(V[i][0])
    #         # Check that there is no envy once an item is possibly dropped
    #         is_possible_and_list.append(Sum([V[i][g] * If(A[i][g], 1, 0)
    #                                          for g in range(m)]) >=
    #                                     Sum([V[i][g] * (If(A[j][g], 1, 0) - If(D[j][i][g], 1, 0))  # TODO dobbelsjekk omindeksenei D er riktige
    #                                          for g in range(m)]))
    # is_possible = And([is_possible_and_list[i]
    #                   for i in range(len(is_possible_and_list))])
    # s.add(Not(is_possible))

    a_jalous_agent = []
    for i in range(n):
        for j in range(n):

            if i == j:
                continue
            # Check that there is at least one evious agent once an item is possibly dropped
            a_jalous_agent.append(Sum([V[i][g] * If(A[i][g], 1, 0)
                                       for g in range(m)]) <
                                  Sum([V[i][g] * If(A[j][g], 1, 0)   # må fåf den til å skjønne at den beste skal legges bort
                                       # TODO dobbelsjekk om indeksene i D er riktige
                                       for g in range(m)]) - max_in_product_array(D[j][i], V[i]))  # dette er sant når en agent oppfatter at verdien av sin egen bundle er mindre enn en annen sin
            s.add(Sum([V[i][g] * If(A[i][g], 1, 0)
                       for g in range(m)]) + epsilon >=
                  Sum([V[i][g] * If(A[j][g], 1, 0)   # må fåf den til å skjønne at den beste skal legges bort
                       # TODO dobbelsjekk om indeksene i D er riktige
                       for g in range(m)]) - max_in_product_array(D[j][i], V[i]))  # dette er sant når en agent oppfatter at verdien av sin egen bundle er mindre enn en annen sin

    # s.add(Or([False for i in range(len(a_jalous_agent))]))
    s.add(Or([a_jalous_agent[i] for i in range(len(a_jalous_agent))]))
    # s.add(Not(And([a_jalous_agent[i] for i in range(len(a_jalous_agent))])))

    s.add(get_edge_conflicts(G, A, n))

    # Check if an EF1 allocation can be found
    print(s.check())
    if(s.check() == sat):
        print(s.model())
        res = s.model()
        print("a_jealous_agent = %s" % [res.evaluate(Sum([V[i][g] * If(A[i][g], 1, 0)
                                                          for g in range(m)]) <
                                                     Sum([V[i][g] * If(A[j][g], 1, 0)   # må fåf den til å skjønne at den beste skal legges bort
                                                          # TODO dobbelsjekk om indeksene i D er riktige

                                                          for g in range(m)]) - max_in_product_array(D[j][i], V[i]))
                                        for i in range(n) for j in range(n)])
        print("values to be jealous for = %s vs %s" % ([res.evaluate(Sum([V[i][g] * If(A[i][g], 1, 0)
                                                                         for g in range(m) for i in range(n)])
                                                                     )], [res.evaluate(Sum([V[i][g] * If(A[j][g], 1, 0)   # må fåf den til å skjønne at den beste skal legges bort
                                                                                            # TODO dobbelsjekk om indeksene i D er riktige

                                                                                            for g in range(m)]) - max_in_product_array(D[j][i], V[i]))]))
        print("Or(a_jealous_agent) = %s" %
              res.evaluate(Or([a_jalous_agent[i] for i in range(len(a_jalous_agent))])))
        print("Or(a_jealous_agent) = %s" %
              [res.evaluate(a_jalous_agent[i]) for i in range(len(a_jalous_agent))])

        for i in range(n):
            for j in range(n):

                if i == j:
                    continue

                # Check that there is at least one evious agent once an item is possibly dropped
                print("Should agent %s be jealous og agent %s? %s vs %s" % (i, j, res.evaluate(Sum([V[i][g] * If(A[i][g], 1, 0)
                                                                                                   for g in range(m)])),
                                                                            res.evaluate(Sum([V[i][g] * If(A[j][g], 1, 0)   # må fåf den til å skjønne at den beste skal legges bort
                                                                                              # TODO dobbelsjekk om indeksene i D er riktige
                                                                                              for g in range(m)]) - max_in_product_array(D[j][i], V[i]))
                                                                            ))  # dette er sant når en agent oppfatter at verdien av sin egen bundle er mindre enn en annen sin
                print("maks produkt for agent %s mot agent %s ble %s" %
                      (i, j, res.evaluate(max_in_product_array(D[j][i], V[i]))))

                print(
                    [res.evaluate(If(a, 1, 0)*b) for a, b in zip(D[j][i], V[i])])
                print("Agent %s sine verdier: %s" % (i,
                                                     [res.eval(a) for a in V[i]]))

        return s.check() == sat

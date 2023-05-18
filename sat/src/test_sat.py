import numpy as np
from igraph import *
from random import randint, random
import time
import csv

from sat import find_valuation_function_and_graph_and_agents_with_no_ef1, find_valuation_function_and_graph_and_agents_with_no_ef1_binary_vals, find_valuation_function_and_graph_and_agents_with_no_ef1_only_paths, find_valuation_function_and_graph_and_agents_with_no_ef1_only_paths_and_cycles, find_valuation_function_and_graph_and_agents_with_no_ef1_ternary_vals, find_valuation_function_and_graph_with_no_ef1, find_valuation_function_with_no_ef1, find_valuation_function_with_no_efx, get_mms_for_this_agent, is_ef1_possible, is_ef1_with_conflicts_possible, is_efx_possible, is_path_always_ef1, matrix_path, maximin_shares, maximin_shares_manual_optimization


def test_sum():
    assert sum([1, 2, 3]) == 6, "Should be 6"


def test_efx_no_conflicts_1():
    n = 2
    m = 3
    V = np.ones((n, m))

    assert is_efx_possible(
        n, m, V) == True, "EFX should be possible when the valuations are identical"


def test_efx_no_conflicts_2():
    n = 2
    m = 3
    V = np.array([[1., 2., 0.], [3., 4., 0.]])
    
    assert is_efx_possible(
        n, m, V) == True, "EFX should be possible when there are only two agents"


def test_efx_no_conflicts_csv():
    n = 7
    m = 2
    V = np.random.rand(n, m)
    
    times = []
    agents = []
    items = []
    timed_out_counter = 0
    for i in range(30):
        n = randint(2, 5)#50)
        m = n*4#randint(n*2, n*4)
        print("iteration:",i,"n:",n,"m:", m)

        V = np.random.randint(100, size=(n, m)).astype(float)

        st = time.time()
        assert is_efx_possible(
                n, m, V) == True, "EFX isnt possible?!"
        
        et = time.time()

        elapsed_time = et - st
        print("elapsed_time", elapsed_time)

        times.append(elapsed_time)
        agents.append(n)
        items.append(m)
        
    rows=zip(times,agents,items)

    with open("efx_no_conflicts.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(("times", "agents", "items"))
        for row in rows:
            writer.writerow(row)
    


def test_discover_bad_valuation_functions_efx_1():
    n = 2
    m = 6

    assert find_valuation_function_with_no_efx(
        n, m)[0] == False, "Should not be able to find a valuation function with no EFX when there are two agents"


def test_ef1_no_conflicts_1():
    n = 2
    m = 3
    V = np.ones((n, m))

    assert is_ef1_possible(
        n, m, V) == True, "EF1 should be possible when there are no conflicts"


def test_ef1_no_conflicts_2():
    n = 7
    m = 2
    V = np.random.rand(n, m)

    assert is_ef1_possible(
        n, m, V) == True, "EF1 should be possible when there are no conflicts"
    
def test_ef1_no_conflicts_csv():    
    times = []
    agents = []
    items = []
    timed_out_counter = 0
    for i in range(100):
        n = randint(2, 10)#50)
        m = randint(n*2, n*4)
        print("iteration:",i,"n:",n,"m:", m)

        V = np.random.rand(n, m)
        
        for i in range(n):
            V[i] = np.round(1000 * V[i] / sum(V[i]) )

        # print(V)
        st = time.time()
        assert is_ef1_possible(
                n, m, V) == True, "EF1 isnt possible?!"
        
        et = time.time()

        elapsed_time = et - st
        print("elapsed_time", elapsed_time)

        times.append(elapsed_time)
        agents.append(n)
        items.append(m)
        
    rows=zip(times,agents,items)

    with open("ef1_no_conflicts.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(("times", "agents", "items"))
        for row in rows:
            writer.writerow(row)
    


def test_ef1_with_conflicts():
    n = 2
    m = 3

    V_1 = np.array([[1., 0., 1.], [1., 0., 1.]])
    V_2 = np.array([[1., 1., 1.], [1., 1., 1.]])

    path = Graph.Ring(n=3, circular=False)
    plot(path, target='path.pdf')

    assert is_ef1_with_conflicts_possible(
        n, m, V_1, path) == False, "EF1 should not be possible in this case"
    assert is_ef1_with_conflicts_possible(
        n, m, V_2, path) == True, "EF1 should be possible in this case"


def test_mms_with_conflicts():
    n = 2
    m = 3

    V_1 = np.array([[3., 1., 2.], [4., 4., 5.]])
    V_2 = np.array([[2., 1.], [1., 2.]])

    path = Graph.Ring(n=3, circular=False)
    plot(path, target='path.pdf')
    empty_graph_3 = Graph(3)
    empty_graph_2 = Graph(2)

    # get_mms_for_this_agent(
    #    0., n, m, V_1, path)
    # get_mms_for_this_agent(
    #    1., n, m, V_1, path)

    maximin_shares(n, m, V_1, empty_graph_3)
    print()
    maximin_shares(n, 2, V_2, empty_graph_2)


def test_mms_with_conflicts_random():

    # n = randint(2, 10)
    n = 6
    m = randint(n*2, n*4)
    p = random()

    has_found_ok_graph = False
    while not has_found_ok_graph:
        V = np.random.randint(100, size=(n, m)).astype(float)
        print(V)
        graph = Graph.Erdos_Renyi(n=m, p=0.2, directed=False)
        print("adjacency matrix:\n", graph.get_adjacency())

        plot(graph, target='Barabasi.pdf')
        max_degree = max(Graph.degree(graph))
        print("n:", n, "m:", m, "max deg:", max_degree)
        if max_degree >= n:
            continue
        has_found_ok_graph = True
        maximin_shares(n, m, V, graph)


def test_mms_with_conflicts_unknown():

    n = 8
    m = 26

    V = np.array([[89., 32., 16., 32., 13., 65.,  1., 60., 59., 73., 61., 30., 87., 98.,  3., 90., 92., 26.,
                   26., 80., 55., 12., 37., 33., 37., 99.],
                  [92., 38., 40., 13., 63., 97., 69., 76., 93., 55., 40., 74., 53., 35., 89.,  1., 45., 78.,
                   75., 56., 88., 57., 71., 38., 69.,  1.],
                  [90., 42., 28.,  4., 88., 11., 27., 77., 50., 73.,  4., 92., 53., 96., 45., 67., 95., 94.,
                   13., 60., 24., 73., 13., 58., 41., 14.],
                  [12., 27., 18.,  6., 69., 65., 83.,  4., 87., 86., 96., 49., 30.,  9., 11., 83., 12.,  4.,
                   49., 27., 51.,  6., 36., 11., 70., 39.],
                  [41., 41., 27., 91., 22., 39., 68., 52., 10., 89., 12., 30., 70., 92., 75., 37., 58., 37.,
                   37., 63., 77., 77., 13.,  8., 26., 12.],
                  [2., 24., 63., 59.,  2., 74., 29., 45., 65., 84., 12.,  0., 98., 30., 89., 59., 83., 34.,
                   34., 77., 13., 19., 24.,  0., 15., 45.],
                  [93., 58., 51., 62., 21.,  5., 55., 22., 38., 81.,  0., 93., 30., 84., 47., 93., 94., 75.,
                   9., 85.,  7., 14., 10., 60., 14., 27.],
                  [32., 88., 63., 39., 21., 49., 25., 24.,  2.,  0., 81.,  8., 19., 11., 62., 91., 77., 80.,
                   58., 71., 64., 25., 47.,  7., 16., 17.]])
    graph = Graph.Adjacency(np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                      [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                          0., 0., 1., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0.],
                                      [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                          0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                                      [0., 1., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 1.,
                                          0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                      [0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 1., 0., 0.,
                                          0., 1., 0., 1., 0., 0., 0., 1., 0., 0., 1., 0., 0.],
                                      [0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0.,
                                      0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                      [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0.,
                                      1., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0.],
                                      [0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
                                      0., 0., 0., 0., 0., 1., 1., 1., 0., 1., 0., 0., 1.],
                                      [0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 1., 1.,
                                      0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
                                      [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
                                      1., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0.],
                                      [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0.,
                                      0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
                                      [0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 1., 0., 0.,
                                      0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 1.],
                                      [0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
                                      0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 0.],
                                      [0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0.,
                                      0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
                                      [0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0.,
                                      1., 0., 1., 0., 0., 1., 0., 1., 1., 0., 0., 0., 0.],
                                      [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                      0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0.],
                                      [0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0.,
                                      0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
                                      [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0.,
                                      0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
                                      [0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 0., 0., 0.,
                                      0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
                                      [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0.,
                                      0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                      [0., 1., 1., 0., 1., 0., 1., 1., 0., 0., 0., 0., 0.,
                                      0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                      [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                      0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
                                      [0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.,
                                      0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1.],
                                      [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 1.,
                                      1., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0.],
                                      [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
                                      0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                      [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0.]]), ADJ_UNDIRECTED)
    plot(graph, target='Barabasi.pdf')
    max_degree = max(Graph.degree(graph))
    print("n:", n, "m:", m, "max deg:", max_degree)
    st = time.time()

    maximin_shares(n, m, V, graph)
    et = time.time()

    print("elapsed time:", et - st)


def test_mms_with_conflicts_manual_optimization():

    n = 8
    m = 26

    V = np.array([[89., 32., 16., 32., 13., 65.,  1., 60., 59., 73., 61., 30., 87., 98.,  3., 90., 92., 26.,
                   26., 80., 55., 12., 37., 33., 37., 99.],
                  [92., 38., 40., 13., 63., 97., 69., 76., 93., 55., 40., 74., 53., 35., 89.,  1., 45., 78.,
                   75., 56., 88., 57., 71., 38., 69.,  1.],
                  [90., 42., 28.,  4., 88., 11., 27., 77., 50., 73.,  4., 92., 53., 96., 45., 67., 95., 94.,
                   13., 60., 24., 73., 13., 58., 41., 14.],
                  [12., 27., 18.,  6., 69., 65., 83.,  4., 87., 86., 96., 49., 30.,  9., 11., 83., 12.,  4.,
                   49., 27., 51.,  6., 36., 11., 70., 39.],
                  [41., 41., 27., 91., 22., 39., 68., 52., 10., 89., 12., 30., 70., 92., 75., 37., 58., 37.,
                   37., 63., 77., 77., 13.,  8., 26., 12.],
                  [2., 24., 63., 59.,  2., 74., 29., 45., 65., 84., 12.,  0., 98., 30., 89., 59., 83., 34.,
                   34., 77., 13., 19., 24.,  0., 15., 45.],
                  [93., 58., 51., 62., 21.,  5., 55., 22., 38., 81.,  0., 93., 30., 84., 47., 93., 94., 75.,
                   9., 85.,  7., 14., 10., 60., 14., 27.],
                  [32., 88., 63., 39., 21., 49., 25., 24.,  2.,  0., 81.,  8., 19., 11., 62., 91., 77., 80.,
                   58., 71., 64., 25., 47.,  7., 16., 17.]])
    graph = Graph.Adjacency(np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                      [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                          0., 0., 1., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0.],
                                      [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                          0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                                      [0., 1., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 1.,
                                          0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                      [0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 1., 0., 0.,
                                          0., 1., 0., 1., 0., 0., 0., 1., 0., 0., 1., 0., 0.],
                                      [0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0.,
                                      0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                      [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0.,
                                      1., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0.],
                                      [0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
                                      0., 0., 0., 0., 0., 1., 1., 1., 0., 1., 0., 0., 1.],
                                      [0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 1., 1.,
                                      0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
                                      [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
                                      1., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0.],
                                      [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0.,
                                      0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
                                      [0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 1., 0., 0.,
                                      0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 1.],
                                      [0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
                                      0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 0.],
                                      [0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0.,
                                      0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
                                      [0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0.,
                                      1., 0., 1., 0., 0., 1., 0., 1., 1., 0., 0., 0., 0.],
                                      [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                      0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0.],
                                      [0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0.,
                                      0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
                                      [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0.,
                                      0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
                                      [0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 0., 0., 0.,
                                      0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
                                      [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0.,
                                      0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                      [0., 1., 1., 0., 1., 0., 1., 1., 0., 0., 0., 0., 0.,
                                      0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                      [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                      0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
                                      [0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.,
                                      0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1.],
                                      [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 1.,
                                      1., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0.],
                                      [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
                                      0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                      [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0.]]), ADJ_UNDIRECTED)
    plot(graph, target='Barabasi.pdf')
    max_degree = max(Graph.degree(graph))
    print("n:", n, "m:", m, "max deg:", max_degree)
    st = time.time()

    maximin_shares_manual_optimization(n, m, V, graph)
    et = time.time()

    print("elapsed time:", et - st)


def test_discover_bad_valuation_functions():
    n = 4
    m = 5

    path = Graph.Ring(n=m, circular=False)
    plot(path, target='path.pdf')

    assert find_valuation_function_with_no_ef1(
        n, m, path)[0] == True, "Could not find a desired valuation function"


def test_send_valuations_for_checking():
    n = 2
    m = 3

    path = Graph.Ring(n=3, circular=False)
    plot(path, target='path.pdf')

    V = find_valuation_function_with_no_ef1(
        n, m, path)[1]
    V = np.array([V[0:3], V[3:6]])

    assert is_ef1_with_conflicts_possible(
        n, m, V, path) == False, "The program was not able to discover a set of valuation functions were EF1 is not possible"


def test_send_valuations_for_checking_bipartite_minus_edge():
    p = 4
    n = 5
    m = p*2

    graph = Graph.Full_Bipartite(4, 4)
    edges = graph.get_edgelist()
    graph.delete_edges([edges[0]])
    plot(graph, target='bipartite.pdf')

    V = find_valuation_function_with_no_ef1(
        n, m, graph)[1]
    V = np.array([V[0:8],
                  V[8:16],
                  V[16:24],
                  V[24:32],
                  V[32:40],
                  ])

    assert is_ef1_with_conflicts_possible(
        n, m, V, graph) == False, "The program was not able to discover a set of valuation functions were EF1 is not possible"


def test_discover_valuations_and_graph():
    p = 3
    n = 4
    m = p*2

    result, V, graph = find_valuation_function_and_graph_with_no_ef1(
        n, m)

    plot(graph, target='from_z3.pdf')
    V = np.array([[agent_vals for agent_vals in V[i:i+m]]
                  for i in range(0, len(V), m)])

    assert is_ef1_with_conflicts_possible(
        n, m, V, graph) == False, "The program was not able to discover a set of valuation functions were EF1 is not possible"


def test_discover_valuations_and_graph_and_agents():
    p = 3
    m = p*2
    st = time.time()


    result, V, graph, n = find_valuation_function_and_graph_and_agents_with_no_ef1(
        m)
    et = time.time()
    print("elapsed time:", et - st)

    plot(graph, target='from_z3.pdf')
    V = np.array([[agent_vals for agent_vals in V[i:i+m]]
                  for i in range(0, len(V), m)])

    assert is_ef1_with_conflicts_possible(
        n, m, V, graph) == False, "The program was not able to discover a set of valuation functions were EF1 is not possible"


def test_discover_valuations_and_graph_and_agents_only_paths_and_cycles():
    for m in range(4, 60):

        result, V, graph, n = find_valuation_function_and_graph_and_agents_with_no_ef1_only_paths_and_cycles(
            m)

    # plot(graph, target='from_z3.pdf', vertex_label=range(m), vertex_size=32,
    #      vertex_color='#bcf6f7')
    # V = np.array([[agent_vals for agent_vals in V[i:i+m]]
    #               for i in range(0, len(V), m)])

    # assert is_ef1_with_conflicts_possible(
    #     n, m, V, graph) == False, "The program was not able to discover a set of valuation functions were EF1 is not possible"


def test_discover_valuations_and_graph_and_agents_only_paths():
    for m in range(3, 60):

        result, V, n = find_valuation_function_and_graph_and_agents_with_no_ef1_only_paths(
            m)

        # V = np.array([[agent_vals for agent_vals in V[i:i+m]]
        #               for i in range(0, len(V), m)])

        # path = Graph.Ring(n=m, circular=False)

        # assert is_ef1_with_conflicts_possible(
        #     n, m, V, path) == False, "The program was not able to discover a set of valuation functions were EF1 is not possible"
    # plot(graph, target='from_z3.pdf', vertex_label=range(m), vertex_size=32,
    #      vertex_color='#bcf6f7')


def test_discover_valuations_and_graph_and_agents_binary_vals():
    # p = 3
    m =  7
    st = time.time()

    result, V, graph, n = find_valuation_function_and_graph_and_agents_with_no_ef1_binary_vals(
        m)
    et = time.time()
    print("elapsed time:", et - st)
    
    plot(graph, target='from_z3.pdf')
    V = np.array([[agent_vals for agent_vals in V[i:i+m]]
                  for i in range(0, len(V), m)])

    assert is_ef1_with_conflicts_possible(
        n, m, V, graph) == False, "The program was not able to discover a set of valuation functions were EF1 is not possible"


def test_discover_valuations_and_graph_and_agents_ternary_vals():
    p = 3
    m = 3

    result, V, graph, n = find_valuation_function_and_graph_and_agents_with_no_ef1_ternary_vals(
        m)

    plot(graph, target='from_z3.pdf')
    V = np.array([[agent_vals for agent_vals in V[i:i+m]]
                  for i in range(0, len(V), m)])

    assert is_ef1_with_conflicts_possible(
        n, m, V, graph) == False, "The program was not able to discover a set of valuation functions were EF1 is not possible"


def test_path():
    m = 5

    result, graph = matrix_path(m)

    plot(graph, target='maybe_path.pdf')


def test_is_path_always_ef1():
    print(is_path_always_ef1())


if __name__ == "__main__":
    test_sum()
    test_efx_no_conflicts_1()
    test_efx_no_conflicts_2()
    test_discover_bad_valuation_functions_efx_1()
    test_ef1_no_conflicts_1()
    test_ef1_no_conflicts_2()
    test_ef1_with_conflicts()
    # test_discover_bad_valuation_functions()
    # test_send_valuations_for_checking()
    # test_send_valuations_for_checking_bipartite_minus_edge()
    # test_discover_valuations_and_graph()
    # test_discover_valuations_and_graph_and_agents()
    # test_discover_valuations_and_graph_and_agents_only_paths()
    # test_discover_valuations_and_graph_and_agents_ternary_vals()
    # test_path()
    # test_is_path_always_ef1()
    # test_discover_valuations_and_graph_and_agents_only_paths_and_cycles()
    # test_mms_with_conflicts()
    test_mms_with_conflicts_random()
    # test_mms_with_conflicts_unknown()
    # test_mms_with_conflicts_manual_optimization()
    print("Everything passed")

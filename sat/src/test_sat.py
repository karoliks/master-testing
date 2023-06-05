import numpy as np
from igraph import *
from random import randint, random
import time
import csv
from z3 import unsat, unknown, sat

from sat import find_valuation_function_and_agents_with_no_efx, find_valuation_function_and_graph_and_agents_with_no_ef1, find_valuation_function_and_graph_and_agents_with_no_ef1_binary_vals, find_valuation_function_and_graph_and_agents_with_no_ef1_only_paths, find_valuation_function_and_graph_and_agents_with_no_ef1_only_paths_and_cycles, find_valuation_function_and_graph_and_agents_with_no_ef1_ternary_vals, find_valuation_function_and_graph_with_no_ef1, find_valuation_function_with_no_ef1, find_valuation_function_with_no_ef1_equal_valuation_functions, find_valuation_function_with_no_efx, get_mms_for_this_agent, is_ef1_possible, is_ef1_with_conflicts_possible, is_efx_possible, is_efx_with_conflicts_possible, is_path_always_ef1, matrix_path, maximin_shares, maximin_shares_manual_optimization, is_agent_graph_connected, is_ef1_with_connectivity_possible, is_graph_connected


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

    
    times = []
    agents = []
    items = []
    results = []
    timed_out_counter = 0
    for i in range(100):
        n = randint(2, 8)#50)
        m = randint(n*2, n*4)
        print("iteration:",i,"n:",n,"m:", m)

        V = np.random.rand(n, m)
        
        for i in range(n):
            V[i] = np.round(1000 * V[i] / sum(V[i]) )

        # print(V)
        st = time.time()
        result = is_efx_possible(
                n, m, V)
        assert result != unsat, "EFX isnt possible?!"
        
        et = time.time()

        elapsed_time = et - st
        print("elapsed_time", elapsed_time)

        times.append(elapsed_time)
        agents.append(n)
        items.append(m)
        results.append(result)
        
    rows=zip(times,agents,items, results)

    with open("efx_no_conflicts.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(("times", "agents", "items", "results"))
        for row in rows:
            writer.writerow(row)

    

def test_mms_no_conflicts_csv():

    
    times = []
    agents = []
    items = []
    results = []
    timed_out_counter = 0
    for i in range(100):
        n = randint(2, 8)#50)
        m = randint(n*2, n*4)
        print("iteration:",i,"n:",n,"m:", m)
        dummy_graph = Graph.Ring(n=m, circular=False)
        

        V = np.random.rand(n, m)
        
        for i in range(n):
            V[i] = np.round(1000 * V[i] / sum(V[i]) )

        # print(V)
        st = time.time()
        result = maximin_shares(
                n, m, V, None)
        # assert result != unsat, "MMS isnt possible?!"
        
        et = time.time()

        elapsed_time = et - st
        print("elapsed_time", elapsed_time)

        times.append(elapsed_time)
        agents.append(n)
        items.append(m)
        results.append(result)
        if result == unknown:
            timed_out_counter = timed_out_counter + 1
            
        
    rows=zip(times,agents,items, results)
    print(timed_out_counter)

    with open("mms_no_conflicts.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(("times", "agents", "items", "results"))
        for row in rows:
            writer.writerow(row)

  
def test_mms_with_conflicts_csv():

    
    times = []
    agents = []
    items = []
    results = []
    timed_out_counter = 0
    for i in range(100):
        n = randint(2, 10)#50)
        m = randint(n*2, n*4)
        print("iteration:",i,"n:",n,"m:", m)
        max_degree = n
        graph = None
        
        
        while max_degree >= n or max_degree == 0:
            p = random()
            
            print("finding new graph, p:", p)
            
            graph = Graph.Erdos_Renyi(n=m, p=p, directed=False)
            max_degree = max(Graph.degree(graph))
        print("found graph")

        plot(graph, target='Erdos_Renyi.pdf')
        V = np.random.rand(n, m)
        
        for i in range(n):
            V[i] = np.round(1000 * V[i] / sum(V[i]) )

        # print(V)
        st = time.time()
        result = maximin_shares(
                n, m, V, graph)
        # assert result != unsat, "MMS isnt possible?!"
        
        et = time.time()

        elapsed_time = et - st
        print("elapsed_time", elapsed_time)

        times.append(elapsed_time)
        agents.append(n)
        items.append(m)
        results.append(result)
        if result == unknown:
            timed_out_counter = timed_out_counter + 1
            
        
    rows=zip(times,agents,items, results)
    print(timed_out_counter)

    with open("mms_with_conflicts.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(("times", "agents", "items", "results"))
        for row in rows:
            writer.writerow(row)

    
def test_ef1_with_conflicts_csv():

    
    times = []
    agents = []
    items = []
    results = []
    timed_out_counter = 0
    for i in range(100):
        n = randint(2, 10)#50)
        m = randint(n*2, n*4)
        print("iteration:",i,"n:",n,"m:", m)
        max_degree = n
        graph = None
        
        
        while max_degree >= n or max_degree == 0:
            p = random()
            
            print("finding new graph, p:", p)
            
            graph = Graph.Erdos_Renyi(n=m, p=p, directed=False)
            max_degree = max(Graph.degree(graph))
        print("found graph")

        plot(graph, target='Erdos_Renyi.pdf')
        V = np.random.rand(n, m)
        
        for i in range(n):
            V[i] = np.round(1000 * V[i] / sum(V[i]) )

        # print(V)
        st = time.time()
        result = is_ef1_with_conflicts_possible(
                n, m, V, graph)
        # assert result != unsat, "MMS isnt possible?!"
        
        et = time.time()

        elapsed_time = et - st
        print("elapsed_time", elapsed_time)

        times.append(elapsed_time)
        agents.append(n)
        items.append(m)
        results.append(result)
        if result == unknown:
            timed_out_counter = timed_out_counter + 1
            
        
    rows=zip(times,agents,items, results)
    print(timed_out_counter)

    with open("ef1_with_conflicts.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(("times", "agents", "items", "results"))
        for row in rows:
            writer.writerow(row)

  
def test_efx_with_conflicts_csv():

    
    times = []
    agents = []
    items = []
    results = []
    timed_out_counter = 0
    for i in range(100):
        n = randint(2, 10)#50)
        m = randint(n*2, n*4)
        print("iteration:",i,"n:",n,"m:", m)
        max_degree = n
        graph = None
        
        
        while max_degree >= n or max_degree == 0:
            p = random()
            
            print("finding new graph, p:", p)
            
            graph = Graph.Erdos_Renyi(n=m, p=p, directed=False)
            max_degree = max(Graph.degree(graph))
        print("found graph")

        plot(graph, target='Erdos_Renyi.pdf')
        V = np.random.rand(n, m)
        
        for i in range(n):
            V[i] = np.round(1000 * V[i] / sum(V[i]) )

        # print(V)
        st = time.time()
        result = is_efx_with_conflicts_possible(
                n, m, V, graph)
        # assert result != unsat, "MMS isnt possible?!"
        
        et = time.time()

        elapsed_time = et - st
        print("elapsed_time", elapsed_time)

        times.append(elapsed_time)
        agents.append(n)
        items.append(m)
        results.append(result)
        if result == unknown:
            timed_out_counter = timed_out_counter + 1
            
        
    rows=zip(times,agents,items, results)
    print(timed_out_counter)

    with open("efx_with_conflicts.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(("times", "agents", "items", "results"))
        for row in rows:
            writer.writerow(row)

      

def test_discover_bad_valuation_functions_efx_1():
    n = 2
    m = 6

    assert find_valuation_function_with_no_efx(
        n, m)[0] == False, "Should not be able to find a valuation function with no EFX when there are two agents"


def test_discover_bad_valuation_functions_efx_2():
    n = 4
    m = 6
    
    st = time.time()
     
  
       
    print("Starting test_discover_bad_valuation_functions_efx_2")
    
    assert find_valuation_function_with_no_efx(
        n, m)[0] == False, "Should not be able to find a valuation function with no EFX when there are two agents"
    
    et = time.time()
    elapsed_time = et - st
    print("elapsed_time", elapsed_time)


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


def test_ef1_with_connectivity_when_it_exists():
    n = 2
    m = 6

    V = np.array([[1., 3., 2., 1., 3., 1.], [1., 3., 2., 1., 3., 1.]])

    path = Graph.Ring(n=6, circular=False)
    plot(path, target='path.pdf')

    assert is_ef1_with_connectivity_possible(
        n, m, V, path) == True, "EF1 should be possible in this case (connected bundle, path)"


def test_ef1_with_connectivity_when_it_exists_2():
    # Almost envy-free allocations with connected bundles: When items are arranged on a path, we
    # prove that connected EF1 allocations exist when there are two, three, or four agents

    n = 2
    m = 3

    V = np.array([[1., 0., 1.], [1., 0., 1.]])

    path = Graph.Ring(n=3, circular=False)
    plot(path, target='path.pdf')

    assert is_ef1_with_connectivity_possible(
        n, m, V, path) == True, "EF1 should be possible in this case (connected bundle, path)"

# TODO usikker på om denne er rett
def test_ef1_with_connectivity_when_it_does_not_exist():
    n = 2
    m = 4

    V = np.array([[1., 1., 1., 1.], [1., 1., 1., 1.]])

    star = Graph.Star(n=4)
    plot(star, target='star.pdf')

    assert is_ef1_with_connectivity_possible(
        n, m, V, star) == False, "EF1 should not be possible in this case (connected bundle, star)"

# From: The Price of Connectivity in Fair Division
# We now address the case of three agents. Assume that G is neither a path nor a star with three edges. Suppose first that there is a
# vertex v with degree at least 4. Consider three agents who have identical utilities with
# value 1 on v and four of its neighbors, and 0 on all other vertices. In any connected allocation,
# an agent who does not get v receives value at most 1, while the bundle of the agent who gets v
# has value at least 3 to her. Hence, the allocation is not EF1.


def test_ef1_with_connectivity_when_it_does_not_exist2():
    n = 3
    m = 5

    V = np.array(
        [[1., 1., 1., 1., 1.], [1., 1., 1., 1., 1.],  [1., 1., 1., 1., 1.]])

    star = Graph.Star(n=5)
    plot(star, target='star.pdf', vertex_label=range(m), vertex_size=32,
         vertex_color='#bcf6f7')

    assert is_ef1_with_connectivity_possible(
        n, m, V, star) == False, "EF1 should not be possible in this case (connected bundle, star)"


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
    plot(graph, target='bipartite_minus_edge.pdf')
    print("Starting")
    st = time.time()

   
    V = find_valuation_function_with_no_ef1(
        n, m, graph)[1] 
    et = time.time()

    print("elapsed time:", et - st)
    V = np.array([V[0:8],
                  V[8:16],
                  V[16:24],
                  V[24:32],
                  V[32:40],
                  ])

    assert is_ef1_with_conflicts_possible(
        n, m, V, graph) == False, "The program was not able to discover a set of valuation functions were EF1 is not possible"

def test_send_valuations_for_checking_bipartite_minus_edge_equal_valuations():
    p = 4
    n = 5
    m = p*2

    graph = Graph.Full_Bipartite(4, 4)
    edges = graph.get_edgelist()
    graph.delete_edges([edges[0]])
    plot(graph, target='bipartite_minus_edge.pdf')
    print("Starting test_send_valuations_for_checking_bipartite_minus_edge_equal_valuations")
    st = time.time()

   
    V = find_valuation_function_with_no_ef1_equal_valuation_functions(
        n, m, graph)[1] 
    et = time.time()

    print("elapsed time:", et - st)
    V = np.array([V[0:8],
                  V[0:8],
                  V[0:8],
                  V[0:8],
                  V[0:8],
                  ])

    assert is_ef1_with_conflicts_possible(
        n, m, V, graph) == False, "The program was not able to discover a set of valuation functions were EF1 is not possible"

def test_send_valuations_for_checking_bipartite_minus_edges():
    p = 4
    n = 5
    m = p*2

    graph = Graph.Full_Bipartite(4, 4)
    edges = graph.get_edgelist()
    print(edges)
    graph.delete_edges([edges[0]])
    graph.delete_edges([edges[4]])
    plot(graph, target='bipartite_minus_edges.pdf')
    print("Starting")
    st = time.time()

   
    V = find_valuation_function_with_no_ef1(
        n, m, graph)[1] 
    et = time.time()

    print("elapsed time:", et - st)
    V = np.array([V[0:8],
                  V[8:16],
                  V[16:24],
                  V[24:32],
                  V[32:40],
                  ])

    assert is_ef1_with_conflicts_possible(
        n, m, V, graph) == False, "The program was not able to discover a set of valuation functions were EF1 is not possible"

def test_send_valuations_for_checking_bipartite_6i_minus_edge():
    p = 3
    n = 4
    m = p*2

    graph = Graph.Full_Bipartite(p, p)
    edges = graph.get_edgelist()
    print(edges)
    graph.delete_edges([edges[0]])
    plot(graph, target='bipartite_minus_edge_6i.pdf')
    print("Starting")
    st = time.time()

   
    V = find_valuation_function_with_no_ef1(
        n, m, graph)[1] 
    et = time.time()

    print("elapsed time:", et - st)
    V = np.array([[agent_vals for agent_vals in V[i:i+m]]
                  for i in range(0, len(V), m)])

    assert is_ef1_with_conflicts_possible(
        n, m, V, graph) == False, "The program was not able to discover a set of valuation functions were EF1 is not possible"
def test_send_valuations_for_checking_bipartite_10i_minus_edge():
    p = 5
    n = p+1
    m = p*2

    graph = Graph.Full_Bipartite(p, p)
    edges = graph.get_edgelist()
    print(edges)
    graph.delete_edges([edges[0]])
    plot(graph, target='bipartite_minus_edge_6i.pdf')
    print("Starting")
    st = time.time()

   
    V = find_valuation_function_with_no_ef1(
        n, m, graph)[1] 
    et = time.time()

    print("elapsed time:", et - st)
    V = np.array([[agent_vals for agent_vals in V[i:i+m]]
                  for i in range(0, len(V), m)])

    assert is_ef1_with_conflicts_possible(
        n, m, V, graph) == False, "The program was not able to discover a set of valuation functions were EF1 is not possible"

def test_send_valuations_for_checking_bipartite_10i_minus_edge():
    p = 5
    n = p+1
    m = p*2

    graph = Graph.Full_Bipartite(p, p)
    edges = graph.get_edgelist()
    print(edges)
    graph.delete_edges([edges[0]])
    plot(graph, target='bipartite_minus_edge_6i.pdf')
    print("Starting")
    st = time.time()

   
    V = find_valuation_function_with_no_ef1(
        n, m, graph)[1] 
    et = time.time()

    print("elapsed time:", et - st)
    V = np.array([[agent_vals for agent_vals in V[i:i+m]]
                  for i in range(0, len(V), m)])

    assert is_ef1_with_conflicts_possible(
        n, m, V, graph) == False, "The program was not able to discover a set of valuation functions were EF1 is not possible"



def test_send_valuations_for_checking_hummel():
    n = 6
    m = 3 + n-1

    graph = Graph.Full_Bipartite(3, n-1)
    plot(graph, target='bipartite_hummel1.pdf', vertex_label=range(m), vertex_size=32,
         vertex_color='#bcf6f7')
    print("Starting")
    st = time.time()

   
    V = find_valuation_function_with_no_ef1(
        n, m, graph)[1] 
    et = time.time()

    print("elapsed time:", et - st)
    V = np.array([[agent_vals for agent_vals in V[i:i+m]]
                  for i in range(0, len(V), m)])
    print(V)

    assert is_ef1_with_conflicts_possible(
        n, m, V, graph) == False, "The program was not able to discover a set of valuation functions were EF1 is not possible"

def test_discover_bad_valuation_functions_csv():    
    times = []
    agents = []
    items = []
    timed_out_counter = 0
    for i in range(4,7):
        n = i
        p = i-1
        m = p*2
        graph = Graph.Full_Bipartite(p, p)
        
        print("iteration:",i,"n:",n,"m:", m)

    
        st = time.time()
        V = find_valuation_function_with_no_ef1(
        n, m, graph)[1]
        et = time.time()

        # assert is_ef1_with_conflicts_possible(
        #         n, m, V, graph) == False, "The program was not able to discover a set of valuation functions were EF1 is not possible"
        
        elapsed_time = et - st
        print("elapsed_time", elapsed_time)

        times.append(elapsed_time)
        agents.append(n)
        items.append(m)
        
    rows=zip(times,agents,items)

    with open("cb_discover_bad_valuation_functions.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(("times", "agents", "items"))
        for row in rows:
            writer.writerow(row)
    


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


def test_discover_valuations_and_graph_csv():
    
    times = []
    agents = []
    items = []
    results = []
    graphs = []
    timed_out_counter = 0
    for i in range(4,5):
        n = i
        m = n-1 + n-1
        
        print("iteration:",i,"n:",n,"m:", m)

    
        st = time.time()
        result, V, graph = find_valuation_function_and_graph_with_no_ef1(
        n, m)
        et = time.time()

        # assert is_ef1_with_conflicts_possible(
        #         n, m, V, graph) == False, "The program was not able to discover a set of valuation functions were EF1 is not possible"
        
        elapsed_time = et - st
        print("elapsed_time", elapsed_time)

        times.append(elapsed_time)
        agents.append(n)
        items.append(m)
        results.append(result)
        graphs.append(graph)
        
    rows=zip(times,agents,items, results)
    print("finished")

    with open("discover_bad_valuation_functions_and_graph_kn1n1.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(("times", "agents", "items", "results"))
        for row in rows:
            writer.writerow(row)
    

def test_discover_valuations_and_graph_and_agents():
    p = 3
    m = 9
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


def test_is_graph_connected():
    graph = Graph.Ring(n=3, circular=False)
    edges = graph.get_edgelist()
    assert is_graph_connected(
        graph) == True, "The path graph is connected, so the answer should be true."
    graph.delete_edges([edges[0]])
    plot(graph, target='not_connected.pdf')

    assert is_graph_connected(
        graph) == False, "The graph is not a connected component, so the answer should be false."

    graph = Graph.Full_Bipartite(3, 4)
    edges = graph.get_edgelist()
    assert is_graph_connected(
        graph) == True, "The path graph is connected, so the answer should be true."
    plot(graph, target='not_connected_full_bipartie.pdf')
    graph.delete_edges([edges[0]])
    assert is_graph_connected(
        graph) == True, "The graph is a connected component, so the answer should be true."


def test_is_agent_graph_connected():
    graph = Graph.Ring(n=3, circular=False)
    allocated = [True, True, True]

    assert is_agent_graph_connected(
        graph, allocated, -1) == True, "The agent graph is connected, so the answer should be true."

    assert is_agent_graph_connected(
        graph, allocated, 1) == False, "The agent graph is not connected, so the answer should be false."

    allocated = [True, False, True]

    assert is_agent_graph_connected(
        graph, allocated, -1) == False, "The agent graph is not a connected component, so the answer should be false."

    assert is_agent_graph_connected(
        graph, allocated, 0) == True, "The agent graph is a connected component, so the answer should be true."



def test_discover_valuations_and_graph_hummel_csv():
    
    times = []
    agents = []
    items = []
    valuation_functions = []
    results = []
    graphs = []
    timed_out_counter = 0
    for i in range(4,10):
        n = i
        m = 3 + n-1
        
        print("iteration:",i,"n:",n,"m:", m)

    
        st = time.time()
        result, V, matrix = find_valuation_function_and_graph_with_no_ef1(
        n, m)
        et = time.time()

        # assert is_ef1_with_conflicts_possible(
        #         n, m, V, graph) == False, "The program was not able to discover a set of valuation functions were EF1 is not possible"
        
        elapsed_time = et - st
        print("elapsed_time", elapsed_time)
        if result == sat:
            graph = Graph.Adjacency(matrix)
            plot(graph, target='from_z3.pdf', vertex_label=range(m), vertex_size=32,
                vertex_color='#bcf6f7')
            

        times.append(elapsed_time)
        agents.append(n)
        items.append(m)
        results.append(result)
        graphs.append()
        valuation_functions.append(matrix)
        
        
    rows=zip(times,agents,items, results)
    print("finished")

    with open("discover_bad_valuation_functions_and_graph_hummel.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(("times", "agents", "items", "results", "vs"))

        for row in rows:
            writer.writerow(row)
    



def test_discover_valuations_and_graph__knn_csv():
    
    times = []
    agents = []
    items = []
    results = []
    valuation_functions = []
    timed_out_counter = 0
    for i in range(4,10):
        n = i
        m = n-1 + n-1
        
        print("iteration:",i,"n:",n,"m:", m)

    
        st = time.time()
        result, V, graph = find_valuation_function_and_graph_with_no_ef1(
        n, m)
        et = time.time()
        if result == sat:
            graph = Graph.Adjacency(graph)
            plot(graph, target='from_z3.pdf', vertex_label=range(m), vertex_size=32,
                vertex_color='#bcf6f7')
            

        # assert is_ef1_with_conflicts_possible(
        #         n, m, V, graph) == False, "The program was not able to discover a set of valuation functions were EF1 is not possible"
        
        elapsed_time = et - st
        print("elapsed_time", elapsed_time)

        times.append(elapsed_time)
        agents.append(n)
        items.append(m)
        results.append(result)
        valuation_functions.append(V)
        
    rows=zip(times,agents,items, results, valuation_functions)
    print("finished")

    with open("discover_bad_valuation_functions_and_graph_knn.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(("times", "agents", "items", "results", "vs"))
        for row in rows:
            writer.writerow(row)
    



def test_discover_valuations_and_agents_efx():
    # p = 3
    m = 7
    st = time.time()


    result, V, n = find_valuation_function_and_agents_with_no_efx(
        m)
    et = time.time()
    print("elapsed time:", et - st)

    V = np.array([[agent_vals for agent_vals in V[i:i+m]]
                  for i in range(0, len(V), m)])
    
    print(n, m, V)
    print(is_efx_possible(
        n, m, V))

    assert is_efx_possible(
        n, m, V) == False, "The program was not able to discover a set of valuation functions were EF1 is not possible"


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

def test_discover_valuations_and_graph_and_agents_only_paths_csv():
    times = []
    results = []
    items = []
    for i in range(3,8):
        
        print("iteration:",i,"m:", i)

    
        st = time.time()
        result, V, n = find_valuation_function_and_graph_and_agents_with_no_ef1_only_paths(
            i)
        et = time.time()

        # assert is_ef1_with_conflicts_possible(
        #         n, m, V, graph) == False, "The program was not able to discover a set of valuation functions were EF1 is not possible"
        
        elapsed_time = et - st
        print("elapsed_time", elapsed_time)

        times.append(elapsed_time)
        results.append(result)
        items.append(i)
        
    rows=zip(items,times,results)

    with open("path_not_ef1.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(("items","times", "results"))
        for row in rows:
            writer.writerow(row)



def test_discover_valuations_and_agents_efx_csv():
    times = []
    results = []
    items = []
    for m in range(1,10):
        
        print("m:", m)

    
        st = time.time()
        result, V, n = find_valuation_function_and_agents_with_no_efx(
            m)
        et = time.time()

        # assert is_ef1_with_conflicts_possible(
        #         n, m, V, graph) == False, "The program was not able to discover a set of valuation functions were EF1 is not possible"
        
        elapsed_time = et - st
        print("elapsed_time", elapsed_time)

        times.append(elapsed_time)
        results.append(result)
        items.append(m)
        
    rows=zip(items,times,results)

    with open("not_efx.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(("items","times", "results"))
        for row in rows:
            writer.writerow(row)



def test_discover_valuations_and_graph_and_agents_binary_vals():
    # p = 3
    m =  8
    st = time.time()

    result, V, graph, n = find_valuation_function_and_graph_and_agents_with_no_ef1_binary_vals(
        m)
    et = time.time()
    print("elapsed time:", et - st)
    
    plot(graph, target='from_z3_binary_'+str(m)+'_items.pdf')
    V = np.array([[agent_vals for agent_vals in V[i:i+m]]
                  for i in range(0, len(V), m)])

    assert is_ef1_with_conflicts_possible(
        n, m, V, graph) == False, "The program was not able to discover a set of valuation functions were EF1 is not possible"

def test_discover_valuations_and_graph_and_agents_binary_vals_csv():
    times = []
    results = []
    items = []
    for m in range(3,8):
        
        print("m:", m)

    
        st = time.time()
        result, V, g,n = find_valuation_function_and_graph_and_agents_with_no_ef1_binary_vals(
            m)
        et = time.time()

        # assert is_ef1_with_conflicts_possible(
        #         n, m, V, graph) == False, "The program was not able to discover a set of valuation functions were EF1 is not possible"
        
        elapsed_time = et - st
        print("elapsed_time", elapsed_time)

        times.append(elapsed_time)
        results.append(result)
        items.append(m)
        
    rows=zip(items,times,results)

    with open("binary_not_ef1.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(("items","times", "results"))
        for row in rows:
            writer.writerow(row)



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
    # test_sum()
    # test_efx_no_conflicts_1()
    # test_efx_no_conflicts_2()
    # test_discover_bad_valuation_functions_efx_1()
    # test_ef1_no_conflicts_1()
    # test_ef1_no_conflicts_2()
    # test_ef1_with_conflicts()
    # test_discover_bad_valuation_functions()
    # test_discover_valuations_and_graph_and_agents()
    # test_discover_bad_valuation_functions()
    # test_send_valuations_for_checking()
    # test_send_valuations_for_checking_bipartite_minus_edges()
    # test_send_valuations_for_checking_bipartite_minus_edge_equal_valuations()
    # test_send_valuations_for_checking_bipartite_6i_minus_edge()
    # test_send_valuations_for_checking_bipartite_10i_minus_edge()
    # test_send_valuations_for_checking_bipartite_minus_edge()
    # test_discover_valuations_and_graph()
    # test_discover_valuations_and_graph_and_agents()
    test_is_agent_graph_connected()
    test_ef1_with_connectivity_when_it_exists()
    test_ef1_with_connectivity_when_it_exists_2()
    test_ef1_with_connectivity_when_it_does_not_exist2()
    # test_is_graph_connected()

    # test_discover_valuations_and_graph_and_agents_only_paths()
    # test_discover_valuations_and_graph_and_agents_ternary_vals()
    # test_path()
    # test_is_path_always_ef1()
    # test_discover_valuations_and_graph_and_agents_only_paths_and_cycles()
    # test_mms_with_conflicts()
    # test_mms_with_conflicts_random()
    # test_mms_with_conflicts_unknown()
    # test_mms_with_conflicts_manual_optimization()
    # test_discover_valuations_and_graph_and_agents_binary_vals()
    # test_ef1_no_conflicts_csv()
    # test_efx_no_conflicts_csv() 
    # test_discover_valuations_and_agents_efx_csv()
    # test_discover_bad_valuation_functions_csv()
    # test_discover_valuations_and_graph_and_agents_only_paths_csv()
    # test_discover_valuations_and_graph_and_agents_binary_vals_csv()
    # test_send_valuations_for_checking_bipartite_minus_edge()
    # test_discover_valuations_and_agents_efx()
    # test_send_valuations_for_checking_bipartite_stray_nodes()
    # test_send_valuations_for_checking_hummel()
    # test_discover_bad_valuation_functions_efx_2()
    # test_discover_valuations_and_agents_efx()
    # test_mms_no_conflicts_csv()
    # test_mms_with_conflicts_csv()
    # test_send_valuations_for_checking_hummel()
    # test_discover_valuations_and_graph_csv()
    # test_ef1_with_conflicts_csv()
    
    print("Everything passed")

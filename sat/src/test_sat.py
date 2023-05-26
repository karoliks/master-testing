import numpy as np
from igraph import *

from sat import find_valuation_function_and_graph_and_agents_with_no_ef1, find_valuation_function_and_graph_with_no_ef1, find_valuation_function_with_no_ef1, is_ef1_possible, is_ef1_with_conflicts_possible, is_ef1_with_connectivity_possible, is_graph_connected


def test_sum():
    assert sum([1, 2, 3]) == 6, "Should be 6"


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


def test_discover_bad_valuation_functions():
    n = 2
    m = 3

    path = Graph.Ring(n=3, circular=False)
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

    result, V, graph, n = find_valuation_function_and_graph_and_agents_with_no_ef1(
        m)

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


if __name__ == "__main__":
    test_sum()
    # test_ef1_no_conflicts_1()
    # test_ef1_no_conflicts_2()
    # test_ef1_with_conflicts()
    # test_discover_bad_valuation_functions()
    # test_send_valuations_for_checking()
    # test_send_valuations_for_checking_bipartite_minus_edge()
    # test_discover_valuations_and_graph()
    # test_discover_valuations_and_graph_and_agents()
    test_ef1_with_connectivity_when_it_exists()
    test_ef1_with_connectivity_when_it_exists_2()
    test_ef1_with_connectivity_when_it_does_not_exist2()
    test_is_graph_connected()

    print("Everything passed")

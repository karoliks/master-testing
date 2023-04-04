import numpy as np
from igraph import *

from sat import find_valuation_function_and_graph_and_agents_with_no_ef1, find_valuation_function_and_graph_and_agents_with_no_ef1_binary_vals, find_valuation_function_and_graph_and_agents_with_no_ef1_only_paths, find_valuation_function_and_graph_and_agents_with_no_ef1_only_paths_and_cycles, find_valuation_function_and_graph_and_agents_with_no_ef1_ternary_vals, find_valuation_function_and_graph_with_no_ef1, find_valuation_function_with_no_ef1, get_mms_for_this_agent, is_ef1_possible, is_ef1_with_conflicts_possible, is_path_always_ef1, matrix_path


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


def test_mms_with_conflicts():
    n = 2
    m = 3

    V_1 = np.array([[1., 0., 1.], [1., 0., 1.]])
    V_2 = np.array([[1., 1., 1.], [1., 1., 1.]])

    path = Graph.Ring(n=3, circular=False)
    plot(path, target='path.pdf')

    get_mms_for_this_agent(
        0, n, m, V_1, path)
    get_mms_for_this_agent(
        1, n, m, V_2, path)


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
    p = 3
    m = 6

    result, V, graph, n = find_valuation_function_and_graph_and_agents_with_no_ef1_binary_vals(
        m)

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
    # test_sum()
    # test_ef1_no_conflicts_1()
    # test_ef1_no_conflicts_2()
    # test_ef1_with_conflicts()
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
    test_mms_with_conflicts()
    print("Everything passed")

import numpy as np
from igraph import *

from sat import find_valuation_function_with_no_ef1, is_ef1_possible, is_ef1_with_conflicts_possible


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
    # V = np.ones((n, m))
    V_1 = np.array([[1., 0., 1.], [1., 0., 1.]])
    V_2 = np.array([[1., 1., 1.], [1., 1., 1.]])
    # G = ig.Graph.Ring(n=3, circular=False)
    path = Graph.Ring(n=3, circular=False)
    plot(path, target='path.pdf')
    # path.get_edgelist()
    assert is_ef1_with_conflicts_possible(
        n, m, V_1, path) == False, "EF1 should not be possible in this case"
    assert is_ef1_with_conflicts_possible(
        n, m, V_2, path) == True, "EF1 should be possible in this case"


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


if __name__ == "__main__":
    test_sum()
    test_ef1_no_conflicts_1()
    test_ef1_no_conflicts_2()
    test_ef1_with_conflicts()
    test_discover_bad_valuation_functions()
    test_send_valuations_for_checking()
    print("Everything passed")

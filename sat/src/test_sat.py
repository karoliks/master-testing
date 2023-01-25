import numpy as np
from igraph import *

from sat import is_ef1_possible, is_ef1_with_conflicts_possible


def test_sum():
    assert sum([1, 2, 3]) == 6, "Should be 6"


def test_ef1_no_conflicts():
    n = 2
    m = 3
    V = np.ones((n, m))

    assert is_ef1_possible(
        n, m, V) == True, "EF1 should be possible when there are no conflicts"


def test_ef1_with_conflicts():
    n = 2
    m = 3
    # V = np.ones((n, m))
    V = np.array([[1., 0., 1.], [1., 0., 1.]])
    # G = ig.Graph.Ring(n=3, circular=False)
    path = Graph.Ring(n=3, circular=False)
    plot(path, target='path.pdf')
    # path.get_edgelist()
    print(V)
    assert is_ef1_with_conflicts_possible(
        n, m, V, path) == False, "EF1 should not be possible in this case"


if __name__ == "__main__":
    test_sum()
    test_ef1_no_conflicts()
    test_ef1_with_conflicts()
    print("Everything passed")

import numpy as np
from igraph import *

from sat_minimal import find_valuation_function_with_no_ef1_not_working, find_valuation_function_with_no_ef1_working


def test_sum():
    assert sum([1, 2, 3]) == 6, "Should be 6"


def test_discover_bad_valuation_functions():
    n = 2
    m = 3

    path = Graph.Ring(n=3, circular=False)
    plot(path, target='path.pdf')
    assert find_valuation_function_with_no_ef1_working(
        n, m, path)[0] == True, "Could not find a desired valuation function"
    assert find_valuation_function_with_no_ef1_not_working(
        n, m, path)[0] == True, "Could not find a desired valuation function"


if __name__ == "__main__":
    test_sum()
    test_discover_bad_valuation_functions()
    print("Everything passed")

import numpy as np

from sat import is_ef1_possible


def test_sum():
    assert sum([1, 2, 3]) == 6, "Should be 6"


def test_ef1_no_conflicts():
    n = 2
    m = 3
    V = np.ones((n, m))

    assert is_ef1_possible(
        n, m, V) == True, "EF1 should be possible when there are no conlficts"


if __name__ == "__main__":
    test_sum()
    test_ef1_no_conflicts()
    print("Everything passed")

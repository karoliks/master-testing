import time
import numpy as np
from random import randint
from igraph import *
import matplotlib.pyplot as plt

from sat import maximin_shares


def barabasi_experiment():

    times = []
    agents = []
    items = []
    timed_out_counter = 0
    discarded_graph_counter = 0
    for i in range(100):
        n = randint(2, 10)
        m = randint(n*2, n*4)
        k = randint(1, n-1)

        V = np.random.randint(100, size=(n, m)).astype(float)
        # print(V)
        graph = Graph.Barabasi(m, k)
        plot(graph, target='Barabasi.pdf')
        max_degree = max(Graph.degree(graph))
        print("i:", i, "n:", n, "m:", m, "max deg:", max_degree)

        if max_degree >= n:
            discarded_graph_counter = discarded_graph_counter + 1
            continue
        st = time.time()

        if not maximin_shares(n, m, V, graph):
            timed_out_counter = timed_out_counter + 1
        et = time.time()

        elapsed_time = et - st
        print(elapsed_time)

        times.append(elapsed_time)
        agents.append(n)
        items.append(m)

    print("timed_out_counter", timed_out_counter)
    print("discarded_graph_counter", discarded_graph_counter)

    # plotting the points
    plt.plot(agents, times, "o")

    # naming the x axis
    plt.xlabel('agents')
    # naming the y axis
    plt.ylabel('execution time (seconds)')

    # giving a title to my graph
    plt.title('barabasi_mms_z3')
    plt.savefig('barabasi_mms_z3.pdf')

    # function to show the plot
    plt.show()


if __name__ == "__main__":

    barabasi_experiment()

    print("Everything passed")

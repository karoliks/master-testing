import time
import numpy as np
from random import randint, random
from igraph import *
import matplotlib.pyplot as plt

from sat import maximin_shares
# from julia.api import Julia

# TODO: snu om og sette antall agenter etter vi vet max degree?


def barabasi_experiment():

    times = []
    agents = []
    items = []
    timed_out_counter = 0
    discarded_graph_counter = 0
    for i in range(300):
        n = randint(2, 10)
        m = randint(n*2, n*4)
        k = randint(1, n//2)

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
        print("elapsed_time", elapsed_time)

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
    plt.title('barabasi_mms_z3_agents')
    plt.savefig('barabasi_mms_z3_agents.pdf')

    # function to show the plot
    plt.show()
    
    # plotting the points
    plt.plot(items, times, "o")

    # naming the x axis
    plt.xlabel('items')
    # naming the y axis
    plt.ylabel('execution time (seconds)')

    # giving a title to my graph
    plt.title('barabasi_mms_z3_items')
    plt.savefig('barabasi_mms_z3_items.pdf')

    # function to show the plot
    plt.show()


def erdos_renyi_experiment():

    times = []
    agents = []
    items = []
    timed_out_counter = 0
    discarded_graph_counter = 0
    for i in range(200):
        n = randint(2, 10)
        m = randint(n*2, n*4)
        p = random()

        V = np.random.randint(100, size=(n, m)).astype(float)
        # print(V)
        graph = Graph.Erdos_Renyi(n=m, p=p, directed=False)
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
        print("elapsed_time", elapsed_time)

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
    plt.title('erdos_renyi_mms_z3_agents')
    plt.savefig('erdos_renyi_mms_z3_agents.pdf')

    # function to show the plot
    plt.show()

    # plotting the points
    plt.plot(items, times, "o")

    # naming the x axis
    plt.xlabel('items')
    # naming the y axis
    plt.ylabel('execution time (seconds)')

    # giving a title to my graph
    plt.title('erdos_renyi_mms_z3_items')
    plt.savefig('erdos_renyi_mms_z3_items.pdf')

    # function to show the plot
    plt.show()



if __name__ == "__main__":

    barabasi_experiment()
    # erdos_renyi_experiment()
    # erdos_renyi_experiment_julia_mip()

    print("Experiments completed")

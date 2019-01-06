from Genome import *
from Node import *
from Connection import *
from graphs import *
from neuralNetwork import *
from geneticAlgorithm import *
from utilityFunctions import *
import numpy as np
import pickle

train_X, train_Y, dev_X, dev_Y, test_X, test_Y = getInputs2()


def main():
    filename = 'C:/Users/Sravan Kumar/PycharmProjects/NEAT for forecasting/best.csv'
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, 'r') as f:
        data = f.readlines()
    f.close()

    genome = Genome(2, 1)
    genome.node_list = []
    final_data = []
    for line in data:
        check = line.rstrip('\n').split(" ")
        check = [i for i in check if i != '']
        if len(check) != 0:
            final_data.append(check)

    test_y = np.array(final_data)
    for val in final_data:
        val = val[0].split(",")
        if len(val) == 2:
            new_node = Node(val[1], dont_change=True)
            new_node.node_number = int(val[0])
            genome.node_list.append(new_node)
        elif len(val) == 5:
            new_conn = Connection(int(val[0]), int(val[1]), float(val[2]))
            new_conn.enabled = bool(val[3])
            new_conn.innovation_number = int(val[4])
            genome.connection_map[int(val[4])] = new_conn
    genome.input_numbers = []
    genome.output_numbers = []
    for node in genome.node_list:
        if node.node_type == 'input':
            genome.input_numbers.append(node.node_number)
        elif node.node_type == 'output':
            genome.output_numbers.append(node.node_number)

    # print(genome.return_nodes())
    # print(genome.return_connections())
    # print(genome.input_numbers)
    # print(genome.output_numbers)
    print(train_X.shape, train_Y.shape)
    evaluation(train_X, train_Y, genome, check=True)
    evaluation(train_X, train_Y, genome)
    print(genome.fitness)
    create_graph(genome,100000,0)
    print(test_X.shape, test_Y.shape)
    evaluation(test_X, test_Y, genome, check=True)
    evaluation(test_X, test_Y, genome)


if __name__ == "__main__":
    main()
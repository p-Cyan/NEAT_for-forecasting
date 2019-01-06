import copy

from neuralNetwork import *
from graphs import *
from Genome import *
from retrieveData import *
from sklearn.metrics import mean_squared_error

# required global variables
stdev_weight = 1.5
hidden_node_mutation_probability = 0.1
connection_mutation_probability = 0.1
reset_weight_mutation_probability = 0.03
nudge_weight_mutation_probability = 0.8
enable_connection_mutation_probability = 0.01
disable_connection_mutation_probability = 0.01
recurrent_node_mutation_probability = 0.1
bias_mutation_probability = 0.8

# hidden_node_mutation_probability = 0.7
# connection_mutation_probability = 0.7
# reset_weight_mutation_probability = 0.7
# nudge_weight_mutation_probability = 0.7
# enable_connection_mutation_probability = 0.07
# recurrent_node_mutation_probability = 0.7


def add_hidden_node_mutation(connection, node_list):
    """
    adds a node in between the connection
    :param connection: creates a node in between this connection
    :return: one new node and two new connections in a list or a None if input connection is disabled
    """
    if connection.enabled is False or connection.from_node == connection.to_node:
        return None
    connection.enabled = False
    new_node = Node('hidden', connection.from_node, connection.to_node, node_list)
    # print(new_node.return_attributes())
    new_connection1 = Connection(connection.from_node, new_node.node_number, 1)
    new_connection2 = Connection(new_node.node_number, connection.to_node, connection.weight)
    return [new_node, new_connection1, new_connection2]


def add_connection_mutation(from_node, to_node, connection_map, from_node_type, to_node_type):
    """
    adds a new connection to given geome
    :param from_node: node number from which the connection should start
    :param to_node  : node number where connection should end
    :param connection_map  : map of connections in current genome mapped to their innovation_number
    :return: a new connection instance
    """
    if from_node == to_node:
        return None
    if to_node_type == 'input' or from_node_type == 'output':
        return None
    if (from_node, to_node) in Connection.connection_reserved_list:
        innovation_number = Connection.connection_reserved_list[(from_node, to_node)]
        if innovation_number in connection_map:
            return None
        else:
            return Connection(from_node, to_node)
    else:
        return Connection(from_node, to_node)


def reset_weight_mutation(connection):
    """
    changes weight to another new random value
    :param connection: weight of connection we wish to change
    :return: None
    """
    if connection.enabled is not False:
        connection.weight = np.random.normal(0, stdev_weight)
    return


def nudge_weight_mutation(connection):
    """
    nudges weight by a random value
    :param connection:
    :return: None
    """
    if connection.enabled is not False:
        random_nudge = random.uniform(-1, 1)
        connection.weight += random_nudge
    return


def enable_connection_mutation(connection):
    """

    :param connection:
    :return:
    """
    connection.enabled = True

def disable_connection_mutation(connection):
    """

    :param connection:
    :return:
    """
    connection.enabled = False

def recurrent_node_mutation(node):
    """

    :param node:
    :return:
    """
    if node.node_type == 'input' or node.node_type == 'output':
        return None
    from_node = node.node_number
    to_node = node.node_number
    new_connection = Connection(from_node, to_node)
    return new_connection

def bias_mutation(node):
    """

    :param node:
    :return:
    """
    node.bias += random.uniform(-0.5,0.5)

def nudge_recurrent_mutation(connection):
    """

    :param connection:
    :return:
    """
    if connection.enabled is not False:
        random_nudge = random.uniform(-1, 1)
        connection.weight += random_nudge
    return


def mutation(gene):
    """
    :param gene:
    :return:
    """
    # get a random number between 0 and 1
    # r = random.random()
    r = 0
    if r < hidden_node_mutation_probability:
        random_connection = random.choice(list(gene.connection_map.keys()))
        return_values = add_hidden_node_mutation(gene.connection_map[random_connection], gene.node_list)
        if return_values is not None:
            gene.node_list.append(return_values[0])
            gene.connection_map[return_values[1].return_innovation_number()] = return_values[1]
            gene.connection_map[return_values[2].return_innovation_number()] = return_values[2]
    r = random.random()
    # r = 0
    if r < connection_mutation_probability:
        to_node = random.choice(gene.node_list)
        from_node = random.choice(gene.node_list)
        return_value = add_connection_mutation(from_node.node_number, to_node.node_number, gene.connection_map,
                                               from_node.node_type, to_node.node_type)
        if return_value is not None:
            gene.connection_map[return_value.return_innovation_number()] = return_value
    r = random.random()
    # r = 0
    if r < reset_weight_mutation_probability:
        connection_key = random.choice(list(gene.connection_map.keys()))
        reset_weight_mutation(gene.connection_map[connection_key])

    r = random.random()
    # r = 0
    if r < nudge_weight_mutation_probability:
        connection_key = random.choice(list(gene.connection_map.keys()))
        nudge_weight_mutation(gene.connection_map[connection_key])
    r = random.random()
    # r = 0
    if r < enable_connection_mutation_probability:
        # TODO: make a separate list which always holds all disabled connections
        list_of_disabled_conn = []
        for key in gene.connection_map:
            if gene.connection_map[key].enabled is False:
                list_of_disabled_conn.append(gene.connection_map[key])
        if len(list_of_disabled_conn) != 0:
            random_connection = random.choice(list_of_disabled_conn)
            enable_connection_mutation(random_connection)

    r = random.random()
    # r = 0
    if r < disable_connection_mutation_probability:
        # TODO: make a separate list which always holds all disabled connections
        list_of_enabled_conn = []
        for key in gene.connection_map:
            if gene.connection_map[key].enabled is True:
                list_of_enabled_conn.append(gene.connection_map[key])
        if len(list_of_enabled_conn) != 0:
            random_connection = random.choice(list_of_enabled_conn)
            disable_connection_mutation(random_connection)


    # r = random.random()
    r = 0
    if r < recurrent_node_mutation_probability:
        if 'hidden' in gene.return_node_types():
            random_node = None
            while random_node is None or random_node.node_type != 'hidden':
                random_node = random.choice(gene.node_list)
            result = recurrent_node_mutation(random_node)
            if result is not None and result.return_innovation_number() not in gene.connection_map:
                gene.connection_map[result.return_innovation_number()] = result

    r = random.random()
    r=0
    if r < bias_mutation_probability:
        if 'hidden' in gene.return_node_types():
            random_node = None
            while random_node is None or random_node.node_type != 'hidden':
                random_node = random.choice(gene.node_list)
            bias_mutation(random_node)

    return

# genome = Genome(7,1)
# genome.initialize_connections()
# mutation(genome)
# print(genome.return_nodes())
# print(genome.return_connections())
# print(genome.return_biases())
# mutation(genome)
# print(genome.return_nodes())
# print(genome.return_connections())
# print(genome.return_biases())


def evaluation(data_X,data_Y,genome,check = False):
    eval = NeuralNet(genome)
    eval.find_order()
    # inp = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    # oup = np.array([[0], [1], [1], [0]])
    inp = data_X
    oup = data_Y
    if check ==  True:
        print(inp,oup)
        pre_oup = eval.total_evaluation(inp,oup,check=True)
        print(pre_oup)
        mse = mean_squared_error(oup,pre_oup)
        print("mse=",mse)
    else:
        genome.fitness = eval.total_evaluation(inp, oup)
        genome.calculate_weight_sum()


def crossover(genome1, genome2):
    # print(genome1.fitness,genome2.fitness)
    if genome1.fitness > genome2.fitness:
        parent1 = genome1
        parent2 = genome2
    else:
        parent1 = genome2
        parent2 = genome1
    child = Genome(len(parent1.input_numbers), len(parent1.output_numbers),
                   old_node_list=copy.deepcopy(parent1.node_list))
    parent1_keys = list(parent1.connection_map.keys())
    parent2_keys = list(parent2.connection_map.keys())
    # print(parent1_keys,parent2_keys)
    # print("\t\t\t",parent1.return_nodes())
    # print("\t\t\t",parent1.return_connections())
    # print("\t",parent1.fitness)
    # print("\t\t\t",parent2.return_nodes())
    # print("\t\t\t",parent2.return_connections())
    # print("\t",parent2.fitness)
    for key in parent1_keys:
        if key in parent2_keys and parent2.connection_map[key].enabled is not False:
            r = random.random()
            if r < 0.5:
                # print("1")
                child.connection_map[key] = copy.deepcopy(parent1.connection_map[key])
            else:
                # print("2")
                child.connection_map[key] = copy.deepcopy(parent2.connection_map[key])
        else:
            # print("3")
            child.connection_map[key] = copy.deepcopy(parent1.connection_map[key])
    #
    # print("\t\t\t",child.return_nodes())
    # print("\t\t\t",child.return_connections())
    return child


def tournament_selection(species, k):
    tournament_list = []
    fitness_list = []
    for i in range(k):
        r = random.choice(species.members)
        fitness_list.append(r.fitness)
        tournament_list.append(r)
    max_val = fitness_list.index(max(fitness_list))
    return tournament_list[max_val]



# genome = Genome(2,1)
# conn1 = Connection(1, 3, 0.9477887436554019)
# conn2 = Connection(2, 3, 1.047752258105477)
# genome.connection_map[conn1.innovation_number]= conn1
# genome.connection_map[conn2.innovation_number]= conn2
# print(genome.return_nodes())
# print(genome.return_connections())
# evaluation(genome)
# print(genome.fitness)
#
# child = crossover(genome,genome)
# print(child.return_nodes())
# print(child.return_connections())
# evaluation(child)
# print(child.fitness)
# evaluation(child)
# print(child.fitness)
# evaluation(child)
# print(child.fitness)
# # print(child.input_numbers,child.output_numbers)
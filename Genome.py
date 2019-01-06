from Node import *
from Connection import *


class Genome:
    """
    a genome class , each genome represents a neural network and its required functions in GA
    Attributes:
        node_list (list)        : a list of node objects
        connection_map( (map)   : a map with innovation number of connection as key and the connection itself as value
    Methods:
        initialize_connections(self) : creates connections between all input nodes and all output nodes
        return_nodes(self)           : returns all nodes and node attributes in the genome as a list
        return_connections(self)    : returns all connections and their connection attributes in the genome as a list
    """
    def __init__(self, input_size , output_size, old_node_list = None):
        # list of nodes generated
        self.node_list = []

        # dictionary of connections generated
        self.connection_map = {}

        # keep track of inout and output nodes
        self.input_numbers = []
        self.output_numbers =[]

        # assign input nodes
        if old_node_list is not None:
            self.node_list.extend(old_node_list)
            for node in self.node_list:
                if node.node_type == 'input':
                    self.input_numbers.append(node.node_number)
                elif node.node_type == 'output':
                    self.output_numbers.append(node.node_number)

        else:
            for i in range(input_size):
                self.node_list.append(Node('input'))
                self.input_numbers.append(self.node_list[-1].node_number)

            # assign output nodes
            for i in range(output_size):
                self.node_list.append(Node('output'))
                self.output_numbers.append(self.node_list[-1].node_number)
        self.fitness = 99.99
        self.weight_sum=0
        self.adjusted_fitness = 999.99

    def initialize_connections(self):
        """
        initilaizes connection between all input nodes and all output nodes
        :return: none
        """

        # assign initial connections between all input nodes and output nodes
        for inp_node in self.node_list:
            if inp_node.node_type == 'input':

                for out_node in self.node_list:
                    if out_node.node_type == 'output':
                        new_connection = Connection(inp_node.node_number, out_node.node_number)
                        self.connection_map[new_connection.innovation_number] = new_connection

    def return_nodes(self):
        """
        creates a list of all node attributes of nodes
        :return: this list
        """
        row = []
        for node in self.node_list:
            row.append(node.return_attributes())
        return row

    def return_connections(self):
        """
        creates a list of all connection attributes of the nodes
        :return: the above mentioned list list
        """
        rows = []
        for key in self.connection_map:
            rows.append(self.connection_map[key].return_connection())
        return rows

    def calculate_weight_sum(self):
        """

        :return:
        """
        sum =0
        for key in self.connection_map:
            sum += self.connection_map[key].weight
        self.weight_sum = sum

    def return_node_types(self):
        """

        :return:
        """
        val = []
        for node in self.node_list:
            val.append(node.node_type)
        return val

    def return_biases(self):
        """

        :return:
        """
        val = {}
        for node in self.node_list:
            val[node.node_number]=node.bias
        return val
from Connection import *
import numpy as np

stdev_weight = 0.5


class Node:
    """
    A node class which represents a node in the neural network
    contains all the attributes of nodes necessary to build a neural network
    Attributes:
        node_type   : the type of node this object is
                        possible types are 'input' , 'output' , 'hidden'
        node_number : the innovation number of a node ( aka , when the node was formed universally )

    Methods:
        return_attributes (self) : returns a list of attributes of the node object
    """

    innovation_number = 1
    previous_made_nodes_map = {}

    def __init__(self, node_type, from_node=None, to_node=None , existing_node_list=None , dont_change = False):
        # type of node
        self.node_type = node_type
        if node_type == 'input' or node_type == 'output':
            self.bias = 0
        else:
            self.bias = np.random.normal(0, stdev_weight)
        if dont_change is True:
            return

        # check if 'hidden' type of node
        if node_type == 'hidden':
            # check if node already exists
            if (from_node,to_node) in Node.previous_made_nodes_map:
                for number in Node.previous_made_nodes_map[(from_node,to_node)]:
                    if number not in existing_node_list:
                        self.node_number = number
                        return
                Node.previous_made_nodes_map[(from_node, to_node)].append(Node.innovation_number)
                self.node_number = Node.innovation_number
                Node.innovation_number += 1
            else:
                Node.previous_made_nodes_map[(from_node, to_node)] = [Node.innovation_number]
                self.node_number = Node.innovation_number
                Node.innovation_number += 1
        # if node type is not 'hidden'
        else:
            # innovation number of node
            self.node_number = Node.innovation_number
            Node.innovation_number += 1

    def return_attributes(self):
        row = [self.node_number, self.node_type, self.bias]
        return row

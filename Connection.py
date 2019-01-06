import random
import numpy as np

# required global variables
stdev_weight = 1


class Connection:
    """
    a class which represents connections in a neural network
    Attributes:
        from_node           : connection starts from
        to_node             : connection ends at
        innovation_number   : innovation number of a connection
        enabled             : a boolean which stats if the connection is enabled or disabled
        (static) global_innovation_number           : next possible innovation number
        (static) connection_reserved_list           : list of all possible connections made till now mapped to their innovation numbers
    Methods:
        return_innovation_number(self)  : returns innovation number of current connection
        return_connection(self)         : returns a list containing all attributes of connection

        static methods
        print_connection_reserved_list(self) : prints connection_reserved_list in console
    """

    global_innovation_number = 0

    connection_reserved_list = {}

    def __init__(self, in_node, out_node, weight = None):
        self.from_node = in_node
        self.to_node = out_node
        if weight != None:
            self.weight = weight
        else :
            self.weight = np.random.normal(0, stdev_weight)
        self.enabled = True
        if (self.from_node, self.to_node) not in Connection.connection_reserved_list:
            Connection.global_innovation_number += 1
            self.innovation_number = Connection.global_innovation_number
            Connection.connection_reserved_list[(self.from_node, self.to_node)] = self.innovation_number
        else:
            self.innovation_number = Connection.connection_reserved_list[(self.from_node, self.to_node)]


    def return_innovation_number(self):
        """
        :return: returns innovation_number of the connection
        """
        return self.innovation_number

    def return_connection(self):
        """
        :return: returns a list containing all the data of the node
        """
        rows = [self.from_node, self.to_node, self.weight, self.enabled, self.innovation_number]
        return rows

    @staticmethod
    def print_connection_reserved_list():
        """
        prints reserved_connection_list
        :return: none
        """
        for value in Connection.connection_reserved_list:
            print(value, Connection.connection_reserved_list[value])

import numpy as np
import math
import random
from sklearn.metrics import mean_squared_error
# import warnings
from scipy.special import expit
# warnings.filterwarnings("error")

def sigmoid(x):
    x = expit(x)
    return x


def relu(x):
    return max(0,x)

#TODO def tanh
#TODO def stepfunc

class NeuralNet:
    def __init__(self,genome):
        # print(genome.return_nodes())
        # print(genome.return_connections())
        self.inputs = genome.input_numbers
        self.outputs = genome.output_numbers
        self.order = []
        self.biases = genome.return_biases()
        # self.input = []
        for number in genome.input_numbers:
            self.order.append(number)
        self.data_map_op = {}
        self.data_map_ip = {}
        for key in genome.connection_map:
            from_node = genome.connection_map[key].from_node
            to_node = genome.connection_map[key].to_node
            weight = genome.connection_map[key].weight
            # print(from_node,to_node,weight, genome.connection_map[key].enabled)
            if genome.connection_map[key].enabled is False:
                continue
            if from_node not in self.data_map_op:
                self.data_map_op[from_node]=[(to_node,weight)]
            else:
                self.data_map_op[from_node].append((to_node,weight))
            if to_node not in self.data_map_ip:
                self.data_map_ip[to_node]=[(from_node,weight)]
            else:
                self.data_map_ip[to_node].append((from_node,weight))
        # print(self.data_map_ip)
        # print(self.data_map_op)
        self.evaluations = {}
        for node in genome.node_list:
            self.evaluations[node.node_number] = 0.0
        # print(self.evaluations)

    def find_order(self):
        keys_to_be_checked = self.data_map_op.keys()
        for value in self.order:
            if value in keys_to_be_checked:
                for next in self.data_map_op[value]:
                    if next[0] not in self.order and next[0] not in self.outputs:
                        self.order.append(next[0])
        self.order.extend(self.outputs)
        # print(self.order)

    def evaluate_for_input(self,input):
        for node_number in self.order:
            if node_number in self.inputs:
                self.evaluations[node_number] = input[node_number-1]
            else:
                eval = 0.0
                for (node,weight) in self.data_map_ip[node_number]:
                    eval+=self.evaluations[node]*weight
                # eval += self.biases[node_number]
                if node_number not in self.outputs:
                    eval = sigmoid(eval)
                # eval = relu(eval)
                self.evaluations[node_number] = eval
        output_values = []
        for node in self.outputs:
            output_values.append(self.evaluations[node])
        return output_values

    def total_evaluation(self,inputs,true_output,check = False):
        loss_value = 0.0
        outputs = []
        for index in range(len(inputs)):
            ip = inputs[index]
            output_pred = self.evaluate_for_input(ip)
            outputs.append(output_pred)
            # loss_value += self.cost_function(np.array(output_pred),np.array(true_output[index]))
        if check == True:
            return outputs
        try:
          loss_value = mean_squared_error(true_output,outputs)
        except:
          print("ERROR!",outputs)
        loss_value = 1/(1+loss_value)
        return loss_value


    def cost_function(self,output_pred,true_output):
        # print(output_pred,true_output)
        return np.sum((output_pred-true_output)**2)

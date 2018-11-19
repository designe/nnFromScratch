import numpy as np
import math

# Make Your Own Neural Network

class neuralNetwork:
    def __init__(self, _inodes, _hnodes, _onodes, _lr):
        # input, hidden, output, learning rate

        self.inodes = _inodes
        self.hnodes = _hnodes
        self.onodes = _onodes

        self.weightIH = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.weightHO = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        # define sigmoid function 
        self.activation = lambda x: return (1. / (1. + np.exp(-x)))
        self.relu = lambda x: return (x if x > 0 else 0)

        self.lr = _lr
        pass

    def train(self, inputs_list, target_list):
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        # forward
        hidden_inputs = np.dot(self.weightIH, inputs)
        hidden_outputs = self.activation(hidden_inputs)

        final_inputs = np.dot(self.weightHO, hidden_outputs)
        final_outputs = self.activation(hidden_inputs)        

        # get errors
        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.weightHO, output_errors)

        # Update weight
        self.weightHO += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))
        self.weightIH += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))
        
        pass

    # forward function
    def query(self, inputs_list):
        inputs = np.array(inputs_list, ndmin=2).T

        hidden_inputs = np.dot(self.weightIH, inputs)
        hidden_outputs = self.activation(hidden_inputs)

        final_inputs = np.dot(self.weightHO, hidden_outputs)
        final_outputs = self.activation(hidden_inputs)        

        return final_outputs


# Main Part
    
input_nodes = 3
hidden_nodes = 3
output_nodes = 3

learning_rate = 0.3

n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)


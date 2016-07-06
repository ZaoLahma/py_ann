#!/usr/bin/env python3

from random import uniform
from math import tanh

BIAS = -1

#------------------------------------------------
# The activation function.
def sigmoid(x):
	return tanh(x)

# The derivate of the activation function
def sigmoid_prim(x):
	return 1.0 - x**2
#------------------------------------------------

class Neuron:
	def __init__(self, n_inputs):
		self.weights = [0.0] * (n_inputs + 1) # For bias
		
		self.prev_out = 0.0
		
		for weight_index in range(len(self.weights)):
			self.weights[weight_index] = uniform(-1, 1)
		print("Created neuron with weights: " + str(self.weights))
		
	def feed_forward(self, values):
		val = 0.0
		for weight, value in zip(self.weights, values):
			val += weight * value
		self.prev_output = sigmoid(val)
		return self.prev_output		
		
class NeuronLayer:
	def __init__(self, n_neurons, n_inputs):
		self.n_neurons = n_neurons
		self.neurons = [Neuron( n_inputs ) for _ in range(0,self.n_neurons)]


class NeuralNet:
	def __init__(self, n_inputs, n_outputs, n_neurons_to_hl, n_hidden_layers):
		self.n_inputs = n_inputs
		self.n_outputs = n_outputs
		self.n_hidden_layers = n_hidden_layers
		self.n_neurons_to_hl = n_neurons_to_hl

		self._init_network()

	def _init_network(self):
		if self.n_hidden_layers>0:
			# create the first layer
			self.layers = [NeuronLayer( self.n_neurons_to_hl,self.n_inputs )]

			# create hidden layers
			for _ in range(self.n_hidden_layers):
				self.layers.append(NeuronLayer( self.n_neurons_to_hl,self.n_neurons_to_hl ))

			# hidden-to-output layer
			self.layers += [NeuronLayer( self.n_outputs,self.n_neurons_to_hl )]
		else:
			# If we don't require hidden layers
			self.layers = [NeuronLayer( self.n_outputs,self.n_inputs )]


	def feed_forward(self, inputs ):
		for layer in self.layers:
			outputs = []
			for neuron in layer.neurons:
				outputs.append(neuron.feed_forward(inputs))
			inputs = outputs   
		return outputs
		
		
		
if __name__ == "__main__":
	net = NeuralNet(3, 1, 2, 1)
	
	print("Result: " + str(net.feed_forward([1, 1, 1])))

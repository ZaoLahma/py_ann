#!/usr/bin/env python3

from random import uniform
from math import tanh

BIAS = -1.0

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
		val += self.weights[-1] * BIAS
		self.prev_out = sigmoid(val)
		return self.prev_out	
		
class NeuronLayer:
	def __init__(self, n_neurons, n_inputs):
		self.n_neurons = n_neurons
		self.neurons = []
		self.error = 0.0
		for _ in range(0, self.n_neurons):
			self.neurons.append(Neuron(n_inputs))


class NeuralNet:
	def __init__(self, n_inputs, n_outputs, n_neurons_to_hl):
		self.n_inputs = n_inputs
		self.n_outputs = n_outputs
		self.n_neurons_to_hl = n_neurons_to_hl

		self._init_network()

	def _init_network(self):
		# create the first layer
		self.layers = [NeuronLayer( self.n_neurons_to_hl,self.n_inputs )]

		# create hidden layer
		self.layers.append(NeuronLayer( self.n_neurons_to_hl,self.n_neurons_to_hl ))

		# hidden-to-output layer
		self.layers += [NeuronLayer( self.n_outputs,self.n_neurons_to_hl )]


	def feed_forward(self, inputs):
		for layer in self.layers:
			outputs = []
			for neuron in layer.neurons:
				outputs.append(neuron.feed_forward(inputs))
			inputs = outputs   
		return outputs
		
	def back_prop(self, expected, learning_rate = 0.01):
		# Get the output layer's error, beginning with getting the error from each neuron
		output_layer = self.layers[2]
		output_errors = [0.0] * output_layer.n_neurons
		for n_index in range(output_layer.n_neurons):
			error = output_layer.neurons[n_index].prev_out - expected[n_index]
			print("Relative error: " + str(error))
			output_errors[n_index] = error * sigmoid_prim(output_layer.neurons[n_index].prev_out)
			
		print("Output layer errors: " + str(output_errors))
		
		# Get hidden layer's error
		hidden_layer = self.layers[1]
		hidden_errors = [0.0] * hidden_layer.n_neurons
		for n_index in range(hidden_layer.n_neurons):
			total = 0.0
			for error_index in range(len(output_errors)):
				total += hidden_layer.neurons[n_index].weights[error_index] * output_errors[error_index]
			hidden_errors[n_index] = total * sigmoid_prim(hidden_layer.neurons[n_index].prev_out)
			
		# Get input layer's error
		input_layer = self.layers[0]
		input_errors = [0.0] * input_layer.n_neurons
		for n_index in range(input_layer.n_neurons):
			total = 0.0
			for error_index in range(len(hidden_errors)):
				total += input_layer.neurons[n_index].weights[error_index] * hidden_errors[error_index]
			input_errors[n_index] = total * sigmoid_prim(input_layer.neurons[n_index].prev_out)
			
		# Update the weights of the input layer
		for error_index in range(len(hidden_errors)):
			for update_index in range(input_layer.n_neurons):
				input_layer.neurons[update_index].weights[error_index] = learning_rate * hidden_errors[error_index] * input_layer.neurons[update_index].prev_out
		
		# Update the weights of the hidden layer
		for error_index in range(len(output_errors)):
			for update_index in range(input_layer.n_neurons):
				hidden_layer.neurons[update_index].weights[error_index] = learning_rate * output_errors[error_index] * hidden_layer.neurons[update_index].prev_out
		
		
if __name__ == "__main__":
	net = NeuralNet(3, 1, 3)
	
	for _ in range(10):
		print("Result: " + str(net.feed_forward([1, 1, 1])))
		
		net.back_prop([1])
	
	

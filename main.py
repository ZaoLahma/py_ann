#!/usr/bin/env python3


from random import uniform
from math import exp

#------------------------------------------------
# The activation function.
def sigmoid(x):
	return (1 / (1 + exp(-x)))

# The derivate of the activation function
def sigmoid_prim(x):
	return (x * (1 - x))
#------------------------------------------------

class Neuron():
	def __init__(self, num_inputs):
		self.weights = [0.0] * num_inputs
		self.prev_output = 0.0
		for weight_index in range(len(self.weights)):
			self.weights[weight_index] = uniform(-1, 1)
		print("Created neuron with weights: " + str(self.weights))
		
	def feed_forward(self, values):
		val = 0.0
		for weight, value in zip(self.weights, values):
			val += weight * value
		self.prev_output = sigmoid(val)
		return self.prev_output

class NeuralNet():
	def __init__(self, num_inputs, num_outputs, num_hidden):
		self.inputs = [Neuron(num_inputs)] * num_inputs
		self.hidden = [Neuron(num_inputs)] * num_hidden
		self.outputs = [Neuron(num_hidden)] * num_outputs
		
		self.layer_results = []
		self.layer_result = []
	
	def feed_forward(self, values):
		for neuron in self.inputs:
			self.layer_result.append(neuron.feed_forward(values))
		self.layer_results.append(self.layer_result)
		self.layer_result = []
		
		for neuron in self.hidden:
			self.layer_result.append(neuron.feed_forward(self.layer_results[-1]))
		self.layer_results.append(self.layer_result)
		self.layer_result = []
			
		for neuron in self.outputs:
			self.layer_result.append(neuron.feed_forward(self.layer_results[-1]))
		self.layer_results.append(self.layer_result)
		self.layer_result = []
		
		return self.layer_results[-1]
		
	def back_propagate(self, expected, actual):
		# Get the error from the output neuron(s)
		out_error = [0.0] * len(self.outputs)
		for neuron_index in range(len(self.outputs)):
			error = expected[neuron_index] - actual[neuron_index]
			out_error[neuron_index] = error * sigmoid_prim(self.outputs[neuron_index].prev_output)
		print("Output error: " + str(out_error))
		
		# Get the error from the hidden neuron(s):
		# Multiply the output error with the weight between the neurons
		# Sum the products as the total error being propagated back to the hidden layer
		# The neuron error is given by multiplying the total error with 
		# the derivative of the activation function applied to the original output value
				

def run():
	print("run called")
	net = NeuralNet(4, 1, 1)
	
	res = net.feed_forward([1, 1, 1, 1])
	
	print("Result: " + str(res))
	
	net.back_propagate([1], res)

if __name__ == "__main__":
	run()

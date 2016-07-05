#!/usr/bin/env python3

from random import uniform
from math import exp

def sigmoid(x):
	return (1 / (1 + exp(-x)))
	
def sigmoid_prim(x):
	return (x * (1 - x))

class Neuron:
	def __init__(self, num_weights):
		self.weights = []
		self.output = 0
		for i in range(0, num_weights):
			self.weights.append(uniform(-1, 1))
			print(self.weights[i])
			
	def feed_forward(self, values):
		if len(self.weights) != len(values):
			return None
		
		x = 0
		for weight, value in zip(self.weights, values):
			x += weight*value
		
		self.output = sigmoid(x)	
		return self.output
		
	def learn(self, values, correct_answer):
		learn_speed = 0.01
		result = self.feed_forward(values)
		print("Result: " + str(result))
		error = correct_answer - result
		print("Error: " + str(error))
		index = 0
		for weight, value in zip(self.weights, values):
			self.weights[index] += learn_speed * error * value
			print("Weight: " + str(self.weights[index]))
			index += 1
		return result

class Network:
	def __init__(self, num_hidden_layers, num_inputs, num_outputs):
		self.neuron_layers = []
		#Input layer
		neurons = []
		for i in range(0, num_inputs):
			neurons.append(Neuron(num_inputs))
		self.neuron_layers.append(neurons)
		#Hidden layers
		for i in range(0, num_hidden_layers):
			neurons = []
			for n in range(0, num_inputs):
				neurons.append(Neuron(num_inputs))
			self.neuron_layers.append(neurons)
		#Output layer
		neurons = []
		for i in range(0, num_outputs):
			neurons.append(Neuron(num_inputs))
		self.neuron_layers.append(neurons)
		
	def feed_forward(self, values):
		layer_results = []
		layer_result = []
		for neuron in self.neuron_layers[0]:
			layer_result.append(neuron.feed_forward(values))
		print(str(layer_result))
		layer_results.append(layer_result)
		
		for index in range(1, len(self.neuron_layers)):
			prev_layer_result = layer_result
			layer_result = []
			for neuron in self.neuron_layers[index]:
				layer_result.append(neuron.feed_forward(prev_layer_result))
			print(str(layer_result))
			layer_results.append(layer_result)
			
		return layer_results[-1]
			
	def back_propagate(self, expected, actual):
		output_layer = self.neuron_layers[-1]
		
		# Get error from each neuron in output layer.
		output_error = [0,0] * len(output_layer)
		for error_index in range(0, len(output_layer)):
			error = expected[error_index] - actual[error_index]
			output_error[error_index] = sigmoid_prim(output_layer[error_index].output) * error
		print(str(output_error))
		

if __name__ == "__main__":
	print("Main called")
	network = Network(1, 4, 1)
	res = network.feed_forward([1, 1, 1, 1])
	
	print("Res: " + str(res))
	
	network.back_propagate([1], res)

	#neuron = Neuron(4)	
	#for i in range(0, 1000):
	#	print("Iteration: " + str(i) + ": " + str(neuron.learn([1, 1, 0, 1], 0.5)))

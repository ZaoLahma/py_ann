#!/usr/bin/env python3

from random import uniform
from math import exp

def sigmoid(x):
	return (1 / (1 + exp(-x)))
	
def sigmoid_prim(x):
	return (x * (1 - x))

class Neutron:
	def __init__(self, num_weights):
		self.weights = []
		for i in range(0, num_weights):
			self.weights.append(uniform(-1, 1))
			print(self.weights[i])
			
	def feed_forward(self, values):
		if len(self.weights) != len(values):
			return None
		
		x = 0
		for weight, value in zip(self.weights, values):
			x += weight*value
			
		return sigmoid(x)
		
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
		self.neutron_layers = []
		#Input layer
		neutrons = []		
		for i in range(0, num_inputs):
			neutrons.append(Neutron(num_inputs))
		self.neutron_layers.append(neutrons)
		#Hidden layers
		for i in range(0, num_hidden_layers):
			neutrons = []
			for n in range(0, num_inputs):
				neutrons.append(Neutron(num_inputs))
			self.neutron_layers.append(neutrons)
		#Output layer
		neutrons = []
		for i in range(0, num_outputs):
			neutrons.append(Neutron(num_inputs))
		self.neutron_layers.append(neutrons)
		
	def feed_forward(self, values):
		layer_results = []
		layer_result = []
		for neutron in self.neutron_layers[0]:
			layer_result.append(neutron.feed_forward(values))
		print(str(layer_result))
		layer_results.append(layer_result)
		
		for index in range(1, len(self.neutron_layers)):
			prev_layer_result = layer_result
			layer_result = []
			for neutron in self.neutron_layers[index]:
				layer_result.append(neutron.feed_forward(prev_layer_result))
			print(str(layer_result))
			layer_results.append(layer_result)			

if __name__ == "__main__":
	print("Main called")
	network = Network(1, 4, 1)
	network.feed_forward([1, 1, 1, 1])	

	#neutron = Neutron(4)	
	#for i in range(0, 1000):
	#	print("Iteration: " + str(i) + ": " + str(neutron.learn([1, 1, 0, 1], 0.5)))

#!/usr/bin/env python3


from random import uniform
from math import exp
from math import tanh

#------------------------------------------------
# The activation function.
def sigmoid(x):
	return tanh(x)

# The derivate of the activation function
def sigmoid_prim(x):
	return 1.0 - x**2
#------------------------------------------------


class InputNeuron:
	def __init__(self, num_in):
		self.value = 0.0
		self.weights = []
		for _ in range(num_in + 1):
			self.weights.append(uniform(-1, 1))
			
		print("Created InputNeuron with weights: " + str(self.weights))
		
		
class Neuron:
	def __init__(self, num_in):
		self.weights = []
		self.input_sum = 0.0
		self.output = 0.0
		self.error = 0.0
		for _ in range(num_in + 1):
			self.weights.append(uniform(-1, 1))
			
		print("Created Neuron with weights: " + str(self.weights))		

class OutputNeuron:
	def __init__(self):
		self.input_sum = 0.0
		self.output = 0.0
		self.error = 0.0
		self.target = 0.0
		
		
class NeuralNet:
	def __init__(self, num_in, num_out):
		self.pre_input_layer = []
		for _ in range(num_in):
			self.pre_input_layer.append(InputNeuron(num_in))
		
		self.input_layer = []
		for _ in range(num_in):
			self.input_layer.append(Neuron(num_in))		
		
		self.hidden_layer = []
		for _ in range(4):
			self.hidden_layer.append(Neuron(len(self.input_layer)))
			
		self.output_layer = []
		for _ in range(num_out):
			self.output_layer.append(OutputNeuron())
			
			
			
	def forward_prop(inputs, expected):
		for i in range(len(inputs)):
			self.pre_input_layer[i].value = inputs[i]
			
		for i in range(len(inputs)):
			total = 0.0
			for j in range(len(self.pre_input_layer)):
				total += self.pre_input_layer[j].value * self.pre_input_layer[j].weights[i]
			self.input_layer[i].input_sum = total
			self.input_layer[i].output = sigmoid(total)
			
		for i in range(len(self.hidden_layer)):
			total = 0.0
			for j in range(len(self.input_layer)):
				total += self.input_layer[j].value * self.input_layer[j].weights[i]
			self.hidden_layer[i].input_sum = total
			self.hidden_layer[i].output = sigmoid(total)
			
		for i in range(len(self.output_layer)):
			total = 0.0
			for j in range(len(self.hidden_layer)):
				total += self.hidden_layer[j].value * self.hidden_layer[j].weights[i]
			self.output_layer[i].input_sum = total
			self.output_layer[i].output = sigmoid(total)				
					

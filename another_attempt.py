#!/usr/bin/env python3

from random import uniform
from math import tanh
from math import exp
from itertools import chain

BIAS = 0

#------------------------------------------------
def sigmoid(x):
	return (1 / (1 + exp(-x)))
	
def sigmoid_prim(x):
	return (sigmoid(x) * (1 - x))
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

		# create the first layer
		self.input_layer = NeuronLayer(self.n_neurons_to_hl,self.n_inputs )

		# create hidden layer
		self.hidden_layer = NeuronLayer(self.n_neurons_to_hl,self.n_neurons_to_hl)

		# hidden-to-output layer
		self.output_layer = NeuronLayer(self.n_outputs,self.n_neurons_to_hl)


	def feed_forward(self, inputs):
		layers = [self.input_layer, self.hidden_layer, self.output_layer]
		for layer in layers:
			outputs = []
			for neuron in layer.neurons:
				outputs.append(neuron.feed_forward(inputs))
			inputs = outputs   
		return outputs
		
	def back_prop(self, expected, learning_rate = 0.01):
		# Get the output layer's error, beginning with getting the error from each neuron
		output_errors = [0.0] * self.output_layer.n_neurons
		for n_index in range(self.output_layer.n_neurons):
			error = self.output_layer.neurons[n_index].prev_out - expected[n_index]
			print("Relative error: " + str(error))
			output_errors[n_index] = error * sigmoid_prim(self.output_layer.neurons[n_index].prev_out)
			
		print("Output layer errors: " + str(output_errors))
		
		
		# Get the error from the hidden neuron(s):
		# Multiply the output error with the weight between the neurons
		# Sum the products as the total error being propagated back to the hidden layer
		# The neuron error is given by multiplying the total error with 
		# the derivative of the activation function applied to the original output value
		#hidden_error = [0.0] * len(self.hidden)
		#for neuron_index in range(len(self.hidden)):
		#	neuron_error_sum = 0.0
		#	for error_index in range(len(out_error)):
		#			neuron_error_sum = neuron_error_sum + self.outputs[error_index].weights[neuron_index] * out_error[error_index]
		#			
		#	# Phew...! Now let's get the neuron error. I think. o.O
		#	hidden_error[neuron_index] = neuron_error_sum * sigmoid_prim(self.hidden[neuron_index].prev_output)		
		
		
		# Get hidden layer's error
		hidden_errors = [0.0] * self.hidden_layer.n_neurons
		for n_index in range(self.hidden_layer.n_neurons):
			neuron_error_sum = 0.0
			for error_index in range(len(output_errors)):
				neuron_error_sum = neuron_error_sum + self.output_layer.neurons[error_index].weights[n_index] * output_errors[error_index]
				
			hidden_errors[n_index] = neuron_error_sum * sigmoid_prim(self.hidden_layer.neurons[n_index].prev_out)
			
			
		#for hidden_index in range(len(self.hidden)):	
		#	for error_index in range(len(out_error)):
		#		for output_index in range(len(self.outputs)):
		#			change = out_error[error_index] * self.hidden[hidden_index].prev_output * learning_rate					
		#			# Only take responsibility for the weight that is connected the hidden layer neuron. Hence weights[hidden_index]
		#			self.outputs[output_index].weights[hidden_index] = self.outputs[output_index].weights[hidden_index] + change
		#			self.outputs[output_index].change_out = change
		#	#print("New output weights: " + str(self.outputs[output_index].weights))			
			
		
		# Update the weights of the output layer
		for hidden_index in range(self.hidden_layer.n_neurons):		
			for error_index in range(len(output_errors)):
				for update_index in range(self.output_layer.n_neurons):
					change = output_errors[error_index] * self.hidden_layer.neurons[hidden_index].prev_out * learning_rate
					# Only take responsibility for the weight that is connected the hidden layer neuron. Hence weights[hidden_index]				
					self.output_layer.neurons[update_index].weights[hidden_index] += change
			
		# Update the weights of the hidden layer
		for input_index in range(self.input_layer.n_neurons):		
			for error_index in range(len(hidden_errors)):
				for update_index in range(self.hidden_layer.n_neurons):
					change = hidden_errors[error_index] * self.input_layer.neurons[hidden_index].prev_out * learning_rate					
					self.hidden_layer.neurons[update_index].weights[hidden_index] += change
					
		# Update the weights of the input layer
		#for input_index in range(self.input_layer.n_neurons):		
		#	for error_index in range(len(hidden_errors)):
		#		for update_index in range(self.hidden_layer.n_neurons):
		#			change = hidden_errors[error_index] * self.input_layer.neurons[hidden_index].prev_out * learning_rate					
		#			self.hidden_layer.neurons[update_index].weights[hidden_index] += change	
	
		
if __name__ == "__main__":
	net = NeuralNet(3, 3, 3)
	
	for _ in range(10):
		print("Result: " + str(net.feed_forward([1, 1, 1])))
		
		net.back_prop([1, 1, 1])
	
	

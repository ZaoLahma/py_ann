#!/usr/bin/env python3


from random import uniform
from math import exp

#------------------------------------------------
# The activation function.
def sigmoid(x):
	return (1 / (1 + exp(-x)))

# The derivate of the activation function
def sigmoid_prim(x):
	return (sigmoid(x) * (1 - x))
#------------------------------------------------

class Neuron():
	def __init__(self, num_inputs):
		self.weights = [0.0] * num_inputs
		self.prev_output = 0.0
		self.change_in = 0.0
		self.change_out = 0.0
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
		#self.inputs = [Neuron(num_inputs)] * num_inputs
		#self.hidden = [Neuron(num_inputs)] * num_hidden
		#self.outputs = [Neuron(num_hidden)] * num_outputs
		
		self.inputs = []
		self.hidden = []
		self.outputs = []
		
		for i in range(num_inputs):
			self.inputs.append(Neuron(num_inputs))
		
		for i in range(num_hidden):
			self.hidden.append(Neuron(num_inputs))	
			
		for i in range(num_outputs):
			self.outputs.append(Neuron(num_hidden))
		
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
		hidden_error = [0.0] * len(self.hidden)
		for neuron_index in range(len(self.hidden)):
			neuron_error_sum = 0.0
			for error_index in range(len(out_error)):
				for output_index in range(len(self.outputs)):
					neuron_error_sum = neuron_error_sum + self.outputs[output_index].weights[neuron_index] * out_error[error_index]
					
			# Phew...! Now let's get the neuron error. I think. o.O
			hidden_error[neuron_index] = neuron_error_sum * sigmoid_prim(self.hidden[neuron_index].prev_output)
			print("prev output: " + str(self.hidden[neuron_index].prev_output))
			print("Hidden error: " + str(hidden_error))
			
		
		for hidden_index in range(len(self.hidden)):	
			for error_index in range(len(out_error)):
				for output_index in range(len(self.outputs)):
					change = out_error[error_index] * self.hidden[hidden_index].prev_output
					print("out_error: " + str(out_error[error_index] ) + " prev output: " + str(self.hidden[hidden_index].prev_output) + " Change: " + str(change))
					self.outputs[output_index].weights[error_index] = self.outputs[output_index].weights[error_index] * change * self.outputs[output_index].change_out
					self.outputs[output_index].change_out = change
			print("New weights: " + str(self.outputs[output_index].weights))
			
		for input_index in range(len(self.inputs)):
			for error_index in range(len(hidden_error)):
				for hidden_index in range(len(self.hidden)):
					change = hidden_error[error_index] * self.inputs[input_index].prev_output
					self.hidden[hidden_index].weights[error_index] = self.hidden[hidden_index].weights[error_index] * change * self.hidden[hidden_index].change_in
					self.hidden[hidden_index].change_in = change
			print("New weights hidden layer: " + str(self.hidden[hidden_index].weights))

def run():
	print("run called")
	net = NeuralNet(4, 1, 4)
	
	for i in range(1000):
		res = net.feed_forward([1, 1, 1, 1])
	
		print("Result: " + str(res))
	
		net.back_propagate([1], res)
		
	res = net.feed_forward([1, 1, 1, 1])
	print("Result: " + str(res))

if __name__ == "__main__":
	run()

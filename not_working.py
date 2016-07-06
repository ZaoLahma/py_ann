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
		self.inputs = []
		self.hidden = []
		self.outputs = []
		
		for i in range(num_inputs + 1):
			self.inputs.append(Neuron(num_inputs + 1))
		
		for i in range(num_hidden):
			self.hidden.append(Neuron(num_inputs + 1))	
			
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
		
	def back_propagate(self, expected, actual, learning_rate, local_learn_rate):
		# Get the error from the output neuron(s)
		out_error = [0.0] * len(self.outputs)
		for neuron_index in range(len(self.outputs)):
			error = expected[neuron_index] - actual[neuron_index]
			#print("Output relative error: " + str(error))
			out_error[neuron_index] = error * sigmoid_prim(self.outputs[neuron_index].prev_output)
		
		# Get the error from the hidden neuron(s):
		# Multiply the output error with the weight between the neurons
		# Sum the products as the total error being propagated back to the hidden layer
		# The neuron error is given by multiplying the total error with 
		# the derivative of the activation function applied to the original output value
		hidden_error = [0.0] * len(self.hidden)
		for neuron_index in range(len(self.hidden)):
			neuron_error_sum = 0.0
			for error_index in range(len(out_error)):
					neuron_error_sum = neuron_error_sum + self.outputs[error_index].weights[neuron_index] * out_error[error_index]
					
			# Phew...! Now let's get the neuron error. I think. o.O
			hidden_error[neuron_index] = neuron_error_sum * sigmoid_prim(self.hidden[neuron_index].prev_output)
			
		
		for hidden_index in range(len(self.hidden)):	
			for error_index in range(len(out_error)):
				for output_index in range(len(self.outputs)):
					change = out_error[error_index] * self.hidden[hidden_index].prev_output * learning_rate
					if None != local_learn_rate:
						change = change + local_learn_rate *  self.hidden[hidden_index].change_out					
					# Only take responsibility for the weight that is connected the hidden layer neuron. Hence weights[hidden_index]
					self.outputs[output_index].weights[hidden_index] = self.outputs[output_index].weights[hidden_index] + change
					self.outputs[output_index].change_out = change
			#print("New output weights: " + str(self.outputs[output_index].weights))
			
		for input_index in range(len(self.inputs)):
			for error_index in range(len(hidden_error)):
				for hidden_index in range(len(self.hidden)):
					change = hidden_error[error_index] * self.inputs[input_index].prev_output * learning_rate
					if None != local_learn_rate:
						change = change + local_learn_rate *  self.inputs[input_index].change_in
					self.hidden[hidden_index].weights[input_index] = self.hidden[hidden_index].weights[input_index] + change
					self.hidden[hidden_index].change_in = change
			#print("New weights hidden layer: " + str(self.hidden[hidden_index].weights))

def run():
	print("run called")
	num_inputs = 3
	num_outputs = 3
	num_hidden = 8
	
	net = NeuralNet(num_inputs, num_outputs, num_hidden)
	
	pattern =         [[1, 0, 0], [0, 0, 1]]#, [1, 1, 0], [1, 0, 1], [0, 1, 0]]
	expected_output = [[0, 1, 1], [1, 1, 0]]#, [0, 0, 1], [0, 1, 0], [1, 0, 1]]
	
	first_res = net.feed_forward(pattern[0])
	
	learn_rate = 0.01
	local_learn_rate = 0.01
	
	for i in range(1000):
		for i in range(len(pattern)):
			res = net.feed_forward(pattern[i])
		
			#print("Result: " + str(res))
		
			net.back_propagate(expected_output[i], res, learn_rate, local_learn_rate)
		
	res = net.feed_forward(pattern[0])
	print("First result: " + str(first_res) + "\nResult: " + str(res))
	
	
	test_pattern = [0, 0, 1]
	res = net.feed_forward(test_pattern)
	print("Result from test: " + str(res))
	

if __name__ == "__main__":
	run()

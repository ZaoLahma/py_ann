#!/usr/bin/env python3

from random import uniform
from math import exp
from math import tanh

#------------------------------------------------
def sigmoid(x):
	return (1 / (1 + exp(-x)))
	
def sigmoid_prim(x):
	return (sigmoid(x) * (1 - x))

def sigmoid_tanh(x):
	return (tanh(x))
	
def sigmoid_tanh_prim(x):
	return (1.0 - x**2)
#------------------------------------------------

def make_matrix(X, Y, val=0.0):
	ret_val = []
	for x in range(X):
		ret_val.append([val] * Y)
	return ret_val
	
class NeuralNet:
	def __init__(self, num_in, num_hid, num_out, sigmoid_func, sigmoid_func_prim):
		self.num_in = num_in + 1
		self.num_hid = num_hid
		self.num_out = num_out
		
		self.sigmoid = sigmoid_func
		self.sigmoid_prim = sigmoid_func_prim
		
		self.activation_in = [1.0] * self.num_in
		self.activation_hid = [1.0] * self.num_hid
		self.activation_out = [1.0] * self.num_out
		
		self.weights_in = make_matrix(self.num_in, self.num_hid)
		self.weights_out = make_matrix(self.num_hid, self.num_out)

		for i in range(self.num_in):
			for j in range(self.num_hid):
				self.weights_in[i][j] = uniform(-1, 1)
				
		for j in range(self.num_hid):
			for k in range(self.num_out):
				self.weights_out[j][k] = uniform(-1, 1)		
								
		self.change_in = make_matrix(self.num_in, self.num_hid)
		self.change_out = make_matrix(self.num_hid, self.num_hid)
		
	def feed_forward(self, inputs):
		
		for i in range(self.num_in - 1):
			self.activation_in[i] = inputs[i]
			
		for j in range(self.num_hid):
			total = 0.0
			for i in range(self.num_in):
				total = total + self.activation_in[i] * self.weights_in[i][j]
			self.activation_hid[j] = self.sigmoid(total)
				
		for k in range(self.num_out):
			total = 0.0
			for j in range(self.num_hid):
				total = total + self.activation_hid[j] * self.weights_out[j][k]
			self.activation_out[k] = self.sigmoid(total)
			
		return self.activation_out

	def back_prop(self, expected, N, M):
		
		# Get ouput layer errors
		output_errors = [0.0] * self.num_out
		for k in range(self.num_out):
			error = expected[k] - self.activation_out[k]
			output_errors[k] = error * self.sigmoid_prim(self.activation_out[k])
			
		# Get hidden layer errors	
		hidden_errors = [0.0] * self.num_hid
		for j in range(self.num_hid):
			error = 0.0
			for k in range(self.num_out):
				error = error + output_errors[k] * self.weights_out[j][k]
			hidden_errors[j] = error * self.sigmoid_prim(self.activation_hid[j])
			
		# Update output layer weights	
		for j in range(self.num_hid):
			for k in range(self.num_out):
				change = output_errors[k] * self.activation_hid[j]
				self.weights_out[j][k] += N * change + M * self.change_out[j][k]
				self.change_out[j][k] = change
		
		#  Update input layer weights
		for i in range(self.num_in):
			for j in range(self.num_hid):
				change = hidden_errors[j] * self.activation_in[i]
				self.weights_in[i][j] += N * change + M * self.change_in[i][j]
				self.change_in[i][j] = change
				
		# Calculate the error
		error = 0.0
		for k in range(len(expected)):
			error += 0.5 * (expected[k] - self.activation_out[k])**2
		return error

class NeuralNetTrainer:
	def __init__(self, net, acceptable_error = 0.01):
		self.net = net
		self.acceptable_error = acceptable_error
		
	def train(self, patterns, expecteds, num_epochs = 10000, N = 0.1, M = 0.1):
		for epoch in range(num_epochs):
			training_complete = True
			
			for pattern_index in range(len(patterns)):
				self.net.feed_forward(patterns[pattern_index])
				error = self.net.back_prop(expecteds[pattern_index], N, M)
				if error > self.acceptable_error:
					training_complete = False
					
			if training_complete:
				print("Acceptably low error (<" + str(self.acceptable_error) + ") achieved after " + str(epoch) + " epochs")
				return
				
		print("Training completed without achieving acceptably low error (<" + str(self.acceptable_error) + ")")
		
if __name__ == "__main__":
	patterns = [[0, 0], [0, 1], [1, 0], [1, 1]]
	expected = [ [0],    [1],    [1],    [0]]
	
	print("Training tanh network")
	net_tanh = NeuralNet(2, 10, 1, sigmoid_tanh, sigmoid_tanh_prim)
	acceptable_error = 0.001
	trainer_tanh = NeuralNetTrainer(net_tanh, acceptable_error)
	trainer_tanh.train(patterns, expected, 10000)
	
	#Test what the net should have learnt
	print("Testing tanh network")
	for pattern in patterns:
		print(str(pattern) + " -> " + str(net_tanh.feed_forward(pattern)))
		
		
	print("Training sigmoid network")	
	net_sigmoid = NeuralNet(2, 10, 1, sigmoid, sigmoid_prim)
	acceptable_error = 0.001
	trainer_sigmoid = NeuralNetTrainer(net_sigmoid, acceptable_error)
	trainer_sigmoid.train(patterns, expected, 10000)
	
	#Test what the net should have learnt
	print("Testing sigmoid network")
	for pattern in patterns:
		print(str(pattern) + " -> " + str(net_sigmoid.feed_forward(pattern)))	
	

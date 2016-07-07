#!/usr/bin/env python3

from random import uniform
from math import exp

#------------------------------------------------
def sigmoid(x):
	return (1 / (1 + exp(-x)))
	
def sigmoid_prim(x):
	return (sigmoid(x) * (1 - x))
#------------------------------------------------

def make_matrix(X, Y, val=0.0):
	ret_val = []
	for x in range(X):
		ret_val.append([val] * Y)
		
	print("Created matrix: " + str(ret_val))
	return ret_val
	
class NeuralNet:
	def __init__(self, num_in, num_hid, num_out):
		self.num_in = num_in + 1
		self.num_hid = num_hid
		self.num_out = num_out
		
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
			self.activation_hid[j] = sigmoid(total)
				
		for k in range(self.num_out):
			total = 0.0
			for j in range(self.num_hid):
				total = total + self.activation_hid[j] * self.weights_out[j][k]
			self.activation_out[k] = sigmoid(total)
			
		return self.activation_out

	def back_prop(self, expected, N = 0.01, M  = 0.01):
		
		# Get ouput layer errors
		output_errors = [0.0] * self.num_out
		for k in range(self.num_out):
			error = expected[k] - self.activation_out[k]
			output_errors[k] = error * sigmoid_prim(self.activation_out[k])
			
		# Get hidden layer errors	
		hidden_errors = [0.0] * self.num_hid
		for j in range(self.num_hid):
			error = 0.0
			for k in range(self.num_out):
				error = error + output_errors[k] * self.weights_out[j][k]
			hidden_errors[j] = error * sigmoid_prim(self.activation_hid[j])
			
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
		
		
		
if __name__ == "__main__":
	net = NeuralNet(3, 5, 3)
	
	for _ in range(10000):
		print(net.feed_forward([1, 1, 1]))
		print(net.back_prop([1, 0, 1]))

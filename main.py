#!/usr/bin/env python3

from random import uniform
from math import exp

def sigmoid(x):
	return (1 / (1 + exp(-x)))
	
def sigmoid_prim(x):
	return (x * (1 - x))

class Perceptron:
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


if __name__ == "__main__":
	print("Main called")
	perceptron = Perceptron(4)
	
	for i in range(0, 100):
		print("Iteration: " + str(i) + ": " + str(perceptron.learn([1, 0, 0, 1], 1)))

import numpy as np

def weighted_sum(x, w, b):
    return  np.dot(x, w) + b

def sigmoid(z):
    return 1/(1+ np.exp(-z))

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate= 0.1):
        self.lr = learning_rate
        self.w1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros(hidden_size)
        self.w2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2  = np.zeros(output_size)

    def forward(self,x):
        self.z1 = weighted_sum(x, self.w1, self.b1)
        self.a1 = sigmoid(self.z1)

        self.z2 = weighted_sum(self.a1, self.w2, self.b2)
        self.a2 = sigmoid(self.z2)
        return self.a2
    
    def backward(self, x, y):
        
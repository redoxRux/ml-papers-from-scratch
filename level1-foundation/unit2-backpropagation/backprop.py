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
        error = (self.a2 - y) ** 2
        delta_output = (self.a2 - y) * (self.a2 * (1 - self.a2))
        blame_w2 = np.outer(self.a1, delta_output )
        delta_hidden = delta_output * self.w2.flatten() * (self.a1 * (1 - self.a1))
        blame_w1 =np.outer( x, delta_hidden) 


        self.w2 -= self.lr * blame_w2
        self.w1 -= self.lr * blame_w1
        self.b2 -= self.lr * delta_output
        self.b1 -= self.lr * delta_hidden


    def fit(self, X, y, epochs=100):
        for epoch in range(epochs):
            for x_i, y_i in zip(X, y):
                self.forward(x_i)
                self.backward(x_i, y_i)
    
    def predict(self,x):
        return self.forward(x)
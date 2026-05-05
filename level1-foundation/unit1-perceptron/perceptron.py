import numpy as np 

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1, threshhold=0.5):
        self.weights = np.random.randn(input_size) * 0.01
        self.lr= learning_rate
        self.threshhold = threshhold

    def predict(self, x):
        total_sum = np.dot(self.weights * x)
        return 1 if total_sum >= self.threshhold else 0

    def fit(self, X, y, epochs=100):
        for epoch in range(epochs):
            for x_i, y_i in zip(X, y):
                prediction = predict(x_i)
                error = y_i - prediction
                self.weights += self.lr * error * x_i
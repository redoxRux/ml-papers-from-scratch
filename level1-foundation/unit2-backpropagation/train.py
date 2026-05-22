import numpy as np
from backprop import NeuralNetwork

X_and = np.array([[0,0],[0,1],[1,0],[1,1]])
y_and = np.array([[0],[0],[0],[1]])  # note: shape (4,1) for output_size=1

X_xor = np.array([[0,0],[0,1],[1,0],[1,1]])
y_xor = np.array([[0],[1],[1],[0]])

def evaluate(X, y, label):
    nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1)
    nn.fit(X, y, epochs=10000)

    print(f"\n── {label} ──")
    correct = 0
    for x_i, y_i in zip(X, y):
        pred = round(float(nn.predict(x_i)))  # round 0.8 → 1, 0.2 → 0
        status = "✅" if pred == y_i else "❌"
        print(f"Input: {x_i} → Predicted: {pred} | Actual: {y_i} {status}")
        correct += (pred == y_i)

    accuracy = correct / len(y) * 100
    print(f"Accuracy: {accuracy}%")

evaluate(X_and, y_and, "AND Gate")
evaluate(X_xor, y_xor, "XOR Gate")
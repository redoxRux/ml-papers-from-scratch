import numpy as np
from perceptron import Perceptron

# ── AND gate data ──────────────────────────
X_and = np.array([[0,0],[0,1],[1,0],[1,1]])
y_and = np.array([0, 0, 0, 1])

# ── XOR gate data ──────────────────────────
X_xor = np.array([[0,0],[0,1],[1,0],[1,1]])
y_xor = np.array([0, 1, 1, 0])

def evaluate(X, y, label):
    p = Perceptron(input_size=2)
    p.fit(X, y, epochs=100)

    print(f"\n── {label} ──")
    correct = 0
    for x_i, y_i in zip(X, y):
        pred = p.predict(x_i)
        status = "✅" if pred == y_i else "❌"
        print(f"Input: {x_i} → Predicted: {pred} | Actual: {y_i} {status}")
        correct += (pred == y_i)

    accuracy = correct / len(y) * 100
    print(f"Accuracy: {accuracy}%")

evaluate(X_and, y_and, "AND Gate")
evaluate(X_xor, y_xor, "XOR Gate")
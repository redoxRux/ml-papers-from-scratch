# The Perceptron — Rosenblatt, 1958

This is my implementation of the perceptron from Frank Rosenblatt's original 1958 paper: "The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain."

## What this paper is about

Rosenblatt asked a simple but deep question — can a machine learn the way biological systems do? Not just detect input, but actually store experience and use it to make better decisions over time.

His answer was the perceptron. It takes visual input, passes signals through a network of weighted connections, and produces a classification. What makes it interesting is that the memory is not stored in one place — it is distributed across the weights of the entire network, which get stronger with each correct experience. This is directly inspired by how neurons in the brain work.

## What I built

A perceptron from scratch in NumPy, following the core ideas from the paper:

- Weighted sum of inputs
- Threshold-based firing
- Weight update rule based on error (correct answer minus predicted answer)
- Demonstration of what the perceptron can and cannot learn

## Files

- `perceptron.py` — the Perceptron class with `__init__`, `predict`, and `fit`
- `train.py` — trains on AND gate and XOR gate to show the perceptron's capability and its core limitation
- `../../../utils/plotting.py` — shared helper for visualising the decision boundary

## How to run

```bash
pip install numpy
python train.py
```

## Results

```
AND Gate — Accuracy: 100%
XOR Gate — Accuracy: 50%
```

The AND gate is linearly separable — one straight line can divide the two classes, and the perceptron learns it perfectly. The XOR gate is not linearly separable — the classes are diagonally interleaved and no straight line can separate them. The perceptron fails completely, which is exactly what Rosenblatt's paper predicts and what Minsky and Papert proved formally in 1969.

## The core limitation

The perceptron can only draw a straight line through feature space. Real world problems almost never work that way. This limitation is what eventually led to multi-layer networks and backpropagation — which is the next paper in this series.

## Key concepts from the paper

- S-points — the input layer, like pixels on a retina, fire all-or-nothing
- A-units — middle layer, sum weighted inputs and fire if they cross a threshold
- R-units — output layer, compete with each other, winner suppresses the rest
- The learning rule — only active connections update, by an amount proportional to the error and the input value
- Gamma system — active cells gain value at the expense of inactive cells, total stays constant, most stable learning system

## Paper reference

Rosenblatt, F. (1958). The perceptron: A probabilistic model for information storage and organization in the brain. Psychological Review, 65(6), 386-408.
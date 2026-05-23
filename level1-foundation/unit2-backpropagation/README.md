# Unit 2 — Backpropagation
**Paper:** Learning Representations by Back-propagating Errors — Rumelhart, Hinton, Williams (1986)

---

## Why This Paper Exists

The Perceptron (Unit 1) could only draw a straight line. It failed at any pattern where no single straight line could separate the groups — like XOR:

```
Input      Output
[0, 0]  →    0
[0, 1]  →    1
[1, 0]  →    1
[1, 1]  →    0
```

The amber dots (output=1) are always diagonal from the gray dots (output=0).
No straight line can separate them. The Perceptron is useless here.

The fix was obvious — stack multiple layers. More layers = curved boundaries = complex patterns.

But there was one massive problem nobody could solve for years:

> If you have 3 layers and the network makes a mistake at the end — which weights do you blame? The ones in layer 1? Layer 2? Layer 3?

This was called the **credit assignment problem**.

---

## Why Calculus Didn't Save Them Earlier

Calculus was already there. Newton, Leibniz — 1600s. The chain rule that backprop uses was known for centuries. So why did it take until 1986?

Three reasons:

1. **Nobody was building deep networks** — with one layer, the Perceptron learning rule worked fine. No urgency.
2. **Computers were too slow** — even if someone wrote the math on paper, running it on real data was impossible. Hinton's 1986 paper landed at exactly the right moment.
3. **The connection was the insight** — calculus gives you the chain rule. Neural networks are a thing. But somebody had to sit down and say: a neural network is just a big composition of functions, and the chain rule tells us exactly how to differentiate a composition of functions. That connection — between chain rule and layered networks — was the real invention.

Wheels existed for thousands of years. Suitcases existed for thousands of years. Wheeled suitcases were only invented in 1970.

---

## What Backpropagation Does

In one sentence — it calculates how much each weight is responsible for the error, then nudges every weight in the right direction.

The training loop:

```
forward → error → backward → update → repeat
```

Done millions of times on thousands of examples. That loop IS neural network training.

---

## How Each Node Calculates — Forward Pass

Every single node in the network does exactly two steps:

```
Step 1 — weighted sum:   z = (x₁×w₁) + (x₂×w₂) + ... + bias
Step 2 — sigmoid:        a = 1 / (1 + e^(-z))
```

`z` — raw weighted sum before sigmoid. Any number.
`a` — sigmoid output. Always between 0 and 1. This flows to the next layer as input.

The output of one layer becomes the input of the next. By the time you reach the last layer the values are no longer raw pixels — they are abstract features the network learned itself.

Our network:

```
x → [w1] → z1 → [sigmoid] → a1 → [w2] → z2 → [sigmoid] → a2 → prediction
```

In code:

```python
def forward(self, x):
    self.z1 = weighted_sum(x, self.w1, self.b1)   # z1 = x·w1 + b1
    self.a1 = sigmoid(self.z1)                     # a1 = σ(z1)
    self.z2 = weighted_sum(self.a1, self.w2, self.b2)  # z2 = a1·w2 + b2
    self.a2 = sigmoid(self.z2)                     # a2 = σ(z2) → final prediction
    return self.a2
```

---

## Why Small Random Initial Weights

**Random** — to break symmetry. If all weights start at zero every node does the exact same calculation, gets the same blame, updates the same way. They never become different from each other. Random breaks that.

**Small** — sigmoid has flat zones at both ends where the slope is nearly zero. If weights start large, z becomes huge, sigmoid saturates, blame becomes almost zero, learning dies before it starts. Small weights keep z in the healthy middle of sigmoid.

```python
self.w1 = np.random.randn(input_size, hidden_size) * 0.1   # random small
self.b1 = np.zeros(hidden_size)                             # zeros fine for bias
```

---

## Error — How Wrong Was the Prediction

```
error = (prediction - target)²  =  (a2 - y)²
```

We square it because:
- Negative and positive differences both count as wrong
- Big mistakes are punished more than small ones — being wrong by 0.4 is four times as bad as being wrong by 0.2

---

## Backward Pass — The Chain Rule

We want blame for every weight independently. The chain rule says:

```
d(error)/d(w1) = d(error)/d(a2) × d(a2)/d(z2) × d(z2)/d(a1) × d(a1)/d(z1) × d(z1)/d(w1)

d(error)/d(w2) = d(error)/d(a2) × d(a2)/d(z2) × d(z2)/d(w2)

d(error)/d(b1) = d(error)/d(a2) × d(a2)/d(z2) × d(z2)/d(a1) × d(a1)/d(z1) × d(z1)/d(b1)

d(error)/d(b2) = d(error)/d(a2) × d(a2)/d(z2) × d(z2)/d(b2)
```

Where each part comes from:

| Part | Formula | Why |
|------|---------|-----|
| `d(error)/d(a2)` | `(a2-y)` | error=(a2-y)² → power rule → (a2-y) |
| `d(a2)/d(z2)` | `a2×(1-a2)` | a2=sigmoid(z2) → derivative of sigmoid |
| `d(z2)/d(w2)` | `a1` | z2=a1×w2+b2 → differentiate w.r.t w2 → a1 |
| `d(z2)/d(b2)` | `1` | z2=a1×w2+b2 → differentiate w.r.t b2 → 1 |
| `d(z2)/d(a1)` | `w2` | z2=a1×w2+b2 → differentiate w.r.t a1 → w2 passes through as constant because we want blame on w1 not w2 |
| `d(a1)/d(z1)` | `a1×(1-a1)` | a1=sigmoid(z1) → derivative of sigmoid |
| `d(z1)/d(w1)` | `x` | z1=x×w1+b1 → differentiate w.r.t w1 → x |
| `d(z1)/d(b1)` | `1` | z1=x×w1+b1 → differentiate w.r.t b1 → 1 |

Expanding the full chain for every weight:

```
d(error)/d(w2) = (a2-y) × a2×(1-a2) × a1
d(error)/d(b2) = (a2-y) × a2×(1-a2) × 1
d(error)/d(w1) = (a2-y) × a2×(1-a2) × w2 × a1×(1-a1) × x
d(error)/d(b1) = (a2-y) × a2×(1-a2) × w2 × a1×(1-a1) × 1
```

`(a2-y) × a2×(1-a2)` appears in all four — save it as `delta_output`:

```
delta_output = (a2-y) × a2×(1-a2)

d(error)/d(w2) = delta_output × a1
d(error)/d(b2) = delta_output × 1
d(error)/d(w1) = delta_output × w2 × a1×(1-a1) × x
d(error)/d(b1) = delta_output × w2 × a1×(1-a1) × 1
```

`delta_output × w2 × a1×(1-a1)` appears in w1 and b1 — save it as `delta_hidden`:

```
delta_hidden = delta_output × w2 × a1×(1-a1)

d(error)/d(w2) = delta_output × a1
d(error)/d(b2) = delta_output × 1
d(error)/d(w1) = delta_hidden × x
d(error)/d(b1) = delta_hidden × 1
```

The pattern that never changes:

```
blame on weight = delta at that node × what that weight received as input
blame on bias   = delta at that node × 1
```

---

## Backward Pass — The Code

```python
def backward(self, x, y):

    # blame at output node
    # d(error)/d(z2) = (a2-y) × a2×(1-a2)
    delta_output = (self.a2 - y) * (self.a2 * (1 - self.a2))

    # blame at hidden node
    # d(error)/d(z1) = delta_output × w2 × a1×(1-a1)
    # w2.flatten() — squeezes w2 from (4,1) to (4,) so shapes match
    delta_hidden = delta_output * self.w2.flatten() * (self.a1 * (1 - self.a1))

    # update weights — blame on weight = delta × input it received
    # np.outer — multiplies two flat arrays into a matrix matching weight shape
    self.w2 -= self.lr * np.outer(self.a1, delta_output)  # (4,) × (1,) → (4,1)
    self.w1 -= self.lr * np.outer(x, delta_hidden)        # (2,) × (4,) → (2,4)

    # update biases — blame on bias = just the delta
    self.b2 -= self.lr * delta_output
    self.b1 -= self.lr * delta_hidden
```

---

## Why Minus in the Update Rule

```
w_new = w_old - learning_rate × blame
```

The blame tells us which direction the error INCREASES.
We go the OPPOSITE direction — downhill on the error landscape.
That minus sign is gradient descent.

Learning rate controls the step size. Too large — overshoot the bottom, never settle. Too small — takes forever.

---

## Results

```
── AND Gate ──
Input: [0 0] → Predicted: 0 | Actual: 0 ✅
Input: [0 1] → Predicted: 0 | Actual: 0 ✅
Input: [1 0] → Predicted: 0 | Actual: 0 ✅
Input: [1 1] → Predicted: 1 | Actual: 1 ✅
Accuracy: 100%

── XOR Gate ──
Input: [0 0] → Predicted: 0 | Actual: 0 ✅
Input: [0 1] → Predicted: 1 | Actual: 1 ✅
Input: [1 0] → Predicted: 1 | Actual: 1 ✅
Input: [1 1] → Predicted: 0 | Actual: 0 ✅
Accuracy: 100%
```

The Perceptron from Unit 1 could never solve XOR. This network solves it perfectly.
That is the difference one hidden layer and backpropagation makes.

---

## Files

```
unit2-backpropagation/
├── backprop.py   — NeuralNetwork class with forward and backward pass
├── train.py      — XOR and AND gate training and evaluation
└── README.md     — this file
```

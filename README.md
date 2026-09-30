# Micrograd: Scalar Autograd Engine  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Akash12373/Autograd-engine/blob/main/micrograd.ipynb)

A lightweight, purely Python implementation of a scalar-valued backpropagation engine, built from scratch. 

This project is an educational exercise inspired by Andrej Karpathy's Micrograd. It builds a dynamic computation graph on the fly and evaluates gradients using reverse-mode automatic differentiation (autodiff). 

While modern deep learning frameworks operate on multi-dimensional tensors, building this engine at the scalar level provides a transparent look at the calculus and graph theories—specifically topological sorting and the chain rule—that power complex neural network training.

## 🚀 Features

- **Custom `Value` Object:** Wraps standard floats to track gradients and mathematical operations.
- **Reverse-Mode Autodiff:** Implements the chain rule to recursively compute gradients backward through the network.
- **Topological Sorting:** Ensures gradients are calculated in the correct sequence through the directed acyclic graph (DAG).
- **Computation Graph Visualization:** Automatically generates SVG diagrams of the mathematical graph using `graphviz`.

## 🧠 What I Learned

Building this engine solidified several core mathematical and software engineering concepts:
1. **Under the Hood of PyTorch/TensorFlow:** Understanding exactly how `.backward()` functions behind the scenes without abstracting away the calculus.
2. **Graph Theory:** Implementing a topological sort to traverse a Directed Acyclic Graph (DAG) for backpropagation.
3. **Derivatives of Activation Functions:** Manually deriving and coding the local gradients for operations like exponentiation, division, and basic arithmetic.

## 💻 Usage

The engine allows you to build mathematical expressions using the `Value` class, automatically building the graph as you go.

```python
from engine import Value # Assuming your file is named engine.py

# Define inputs and weights
x1 = Value(2.00, label="x1")
w1 = Value(-3.00, label="w1")
x2 = Value(0.00, label="x2")
w2 = Value(1.00, label="w2")
b = Value(6.88137, label="b")

# Forward pass
x1w1 = x1 * w1; x1w1.label = "x1w1"
x2w2 = x2 * w2; x2w2.label = "x2w2"
n = x1w1 + x2w2 + b; n.label = "n"

# Activation function (Tanh equivalent using exp)
e = (2 * n).exp()
o = (e - 1) / (e + 1); o.label = "o"

# Backward pass
o._backward()

# Fetch the gradient of a specific node
print(f"Gradient of e: {e.grad}")

# Visualize the graph (Requires Graphviz system installation)
dot = o.visualise()
dot.render('computation_graph', format='svg', view=True)

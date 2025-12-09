# autograd_scaler — A Simple AutoGrad Engine

`autograd_scaler` is a small educational automatic differentiation engine inspired by PyTorch tensors. 
It includes:
- A `Value` class for scalar automatic differentiation
- Basic neural components (`Neuron`, `Layer`, `MLP`)
- A Graphviz-based computation graph visualizer

This project is intended to help understand how gradient tracking, backpropagation, and neural networks work internally.

---

## Features

- Reverse-mode automatic differentiation
- Operator overloading (`+`, `*`, `-`, `/`, `tanh`, etc.)
- Neuron, Layer, and MLP classes for building simple neural networks
- Graph visualization of the computation graph
- Example notebook demonstrating training on a toy dataset

---

## Project Structure

```plaintext
autograd_scaler/
│
├── value.py          # Implements the Value class & backward pass
├── models.py         # Neuron, Layer, and MLP implementations
├── vizualize.py      # Graph visualization utilities
│
└── examples/
    └── example.ipynb  # Demonstration and training example
```
---
## 🔧 Installation

Clone the repository and install it locally:

```bash
pip install -e .
```

---

## Graphviz Requirement (For Visualization)

The computation graph visualizer depends on **Graphviz**, which has two parts:

**1. The system-level Graphviz program** (required to render images)  
**2. The Python `graphviz` package** (used by the code to generate diagrams)

### Install the Graphviz System Package

#### macOS (Homebrew)

```bash
brew install graphviz
```
### Windows Installation (Graphviz)

```bash
choco install graphviz
```
### Python Graphviz Package

Install the Python wrapper for Graphviz:

```bash
pip install graphviz
```

## Example Notebook

The `examples/example.ipynb` notebook demonstrates how to:

- Build and visualize computation graphs
- Train a simple neural network using the autograd engine
- Observe gradients during backpropagation

Open the notebook to explore the system step-by-step.





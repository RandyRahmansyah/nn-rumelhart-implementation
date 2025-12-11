# Neural Network Backpropagation Implementation (Rumelhart, 1986)

A pure C++ implementation of the Backpropagation algorithm based on the seminal paper **"Learning representations by back-propagating errors"** by David E. Rumelhart, Geoffrey E. Hinton, and Ronald J. Williams (Nature, 1986).

This repository focuses on understanding the "first principles" of Deep Learning by implementing the math equations directly without using high-level frameworks like TensorFlow or PyTorch.

## Reference Paper
> **Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986).**
> Learning representations by back-propagating errors. *Nature*, 323(6088), 533-536.

## Features
* **From Scratch:** Implemented using only C++ standard library (`<vector>`, `<cmath>`). No external linear algebra libraries were used.
* **Momentum Support:** Implements Equation 9 from the paper ($\Delta w(t) = -\epsilon \partial E/\partial w(t) + \alpha \Delta w(t-1)$).
* **Symmetry Detection:** Includes the specific experiment described in the paper (detecting symmetry in 6-bit input vectors).
* **Flexible Architecture:** Supports arbitrary number of layers and neurons.

## Project Structure
```text
nn-rumelhart-implementation/
├── src/
│   └── neural.cpp       # Single-file implementation
├── LICENSE              # MIT License
└── README.md            # Documentation
````

## Getting Started

### Prerequisites

  * C++ Compiler (GCC, Clang, or MSVC)

### Build & Run Instructions

Since this is a single-file implementation, you can compile it directly using your terminal.

**Linux / macOS:**

```bash
# 1. Compile
g++ -O3 -std=c++17 src/neural.cpp -o neural_net

# 2. Run
./neural_net
```

**Windows (MinGW/CMD):**

```cmd
:: 1. Compile
g++ -O3 -std=c++17 src/neural.cpp -o neural_net.exe

:: 2. Run
neural_net.exe
```

## Experiments Included

The program automatically runs two main demonstrations:

1.  **XOR Problem:**
    A classic non-linear classification problem to test if the network can learn non-linear boundaries.

2.  **Symmetry Detection (The Paper's Experiment):**

      * **Input:** 6-bit binary vectors.
      * **Task:** Detect if the pattern is symmetric around the center.
      * **Result:** Demonstrates that hidden layers are essential for this task.

## Implemented Equations

The code strictly follows the notation from the 1986 paper:

  * **Activation:** Sigmoid function $y_j = 1 / (1 + e^{-x_j})$
  * **Error Function:** $E = \frac{1}{2} \sum (y - d)^2$
  * **Weight Update (with Momentum):**
    $$ \Delta w(t) = -\epsilon \frac{\partial E}{\partial w(t)} + \alpha \Delta w(t-1) $$
    Where:
      * $\epsilon$ (epsilon) = Learning rate
      * $\alpha$ (alpha) = Momentum coefficient

## License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

## 👤 Author

**RandyRahmansyah**

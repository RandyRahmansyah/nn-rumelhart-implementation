# Contributing to Neural Network Implementation

First off, thank you for considering contributing to this project! It's people like you that make the open-source community such an amazing place to learn, inspire, and create.

## How Can You Contribute?

### 1. Reporting Bugs
If you find a bug (e.g., the momentum calculation seems off, or the compiler throws a warning), please open an **Issue** on GitHub.
* Be descriptive.
* Include your compiler version (e.g., g++ 9.4.0).
* Describe what happened and what you expected to happen.

### 2. Suggesting Enhancements
Since this is a research implementation, we welcome:
* **Optimizations:** Making the matrix operations faster.
* **New Experiments:** Adding new datasets or scenarios (like MNIST) without breaking the original Rumelhart demo.
* **Code Refactoring:** Splitting the single `neural.cpp` into headers/sources (modularization).

### 3. Pull Requests (PR)
1.  **Fork** the repository.
2.  **Clone** your fork locally.
3.  **Create a branch** for your feature (`git checkout -b feature/amazing-feature`).
4.  **Commit** your changes. Please use clear commit messages.
5.  **Push** to the branch.
6.  **Open a Pull Request**.

## Coding Style Guidelines

Since this project uses modern C++, please adhere to the following:

* **Standard:** Use C++17 features where appropriate.
* **No External Libraries:** Try to keep dependencies minimal (Standard Library only) to maintain the "from scratch" philosophy.
* **Naming:**
    * Classes: `PascalCase` (e.g., `NeuralNetwork`)
    * Variables/Functions: `camelCase` (e.g., `calculateError`, `inputLayer`)
* **Formatting:** Please keep the indentation consistent with the existing code.

## Testing
Before submitting a PR, please compile and run the code to ensure the original experiments (XOR and Symmetry Detection) still pass with high accuracy.

```bash
g++ -O3 -std=c++17 src/neural.cpp -o neural_net
./neural_net



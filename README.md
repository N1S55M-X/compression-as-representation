# Compression-as-Representation

Exploring whether classical compression algorithms can act as effective feature representations for machine learning.

---

## Overview

Most machine learning pipelines follow a familiar pattern:

```text
Raw Data
    ↓
Neural Network
    ↓
Learned Representation
    ↓
Prediction
```

This project investigates an alternative viewpoint:

```text
Raw Data
    ↓
Compression Transform
    ↓
Compressed Representation
    ↓
Simple Learning Algorithm
    ↓
Prediction
```

The central question is:

> Can classical compression algorithms preserve enough task-relevant information that learning can be performed directly on the compressed representation?

---

## Core Hypothesis

A compression algorithm can be interpreted as a **variable transformation**.

Instead of viewing compression purely as a storage mechanism, we can view it as a method for creating a new representation of the data.

If the compressed representation preserves the important structure of the original data, then even simple learning algorithms may achieve strong performance.

Formally:

```text
X (Raw Variables)
        ↓
Compression Transform T
        ↓
Z (Compressed Variables)
        ↓
Learning Algorithm
        ↓
Prediction
```

---

## Motivation

Many successful scientific and engineering systems rely on compression:

* Physics compresses observations into equations.
* Biology compresses instructions into DNA.
* Mathematics compresses patterns into formulas.
* Machine learning compresses information into latent representations.

This project explores whether classical compression methods can serve as useful machine-learning representations.

---

## Experiments

### Experiment 1: Polar Compression

Representation:

```text
(x, y)
   ↓
(r, θ)
```

Result:

* Differentiable representation
* Trainable with gradient descent
* Significant information loss
* MNIST accuracy ≈ 50%

Observation:

Polar coordinates preserve position but lose important structural information needed for classification.

---

### Experiment 2: Quadtree Compression + Neural Network

Representation:

```text
784 Pixels
      ↓
170 Quadtree Features
      ↓
Small MLP
      ↓
Classification
```

Features:

* Mean intensity per region
* Standard deviation per region
* Hierarchical spatial decomposition

Result:

* Test Accuracy: 95.46%
* Variable Reduction: ~78%

Observation:

A simple hierarchical compression preserves most information required for digit classification.

---

### Experiment 3: Quadtree Compression + Logistic Regression

Representation:

```text
784 Pixels
      ↓
170 Quadtree Features
      ↓
Logistic Regression
      ↓
Classification
```

Result:

* Test Accuracy: 92.35%
* No hidden layers
* No deep neural network

Observation:

Most predictive power is contained within the compressed representation itself.

---

## Current Results

| Method               | Features | Model               | Test Accuracy |
| -------------------- | -------- | ------------------- | ------------- |
| Polar Compression    | ~170     | MLP                 | ~50%          |
| Quadtree Compression | 170      | MLP                 | 95.46%        |
| Quadtree Compression | 170      | Logistic Regression | 92.35%        |
---

## Polar Compression

### Training Results

<img width="627" height="481" alt="Polar Compression Results" src="https://github.com/user-attachments/assets/1508f43f-20d2-47ec-a669-29bfb919efee" />

---

## Quadtree Compression + MLP

### Training Results

<img width="473" height="240" alt="Quadtree Compression MLP Results" src="https://github.com/user-attachments/assets/0e8abec9-5b28-4b5b-9b31-945c67e34fd4" />

---

## Quadtree Compression + Logistic Regression

### Training Results

<img width="463" height="272" alt="Quadtree Compression Logistic Regression Results" src="https://github.com/user-attachments/assets/6c26631f-7241-40a9-8ab5-bdecb31febe5" />




---

## Interpretation

Current experiments suggest that:

* Some compression methods preserve task-relevant information much better than others.
* Hierarchical representations can be highly informative.
* A strong representation can reduce dependence on complex models.
* Compression can function as a form of feature engineering.

These experiments do **not** demonstrate that:

* Compression equals intelligence.
* Compression replaces neural networks.
* Any single compression method is universally superior.

The goal is to investigate compression as a representation-learning framework.

---

## Planned Benchmarks

Future representations:

* PCA
* Wavelets
* JPEG / DCT
* Quadtree Variants
* Fractal Descriptors
* Chain Codes
* Sparse Coding
* Graph-Based Compression

Evaluation Metrics:

* Classification Accuracy
* Compression Ratio
* Training Time
* Memory Usage
* Robustness to Noise

---

## Research Questions

1. Which compression algorithms preserve the most useful information?

2. How much dimensionality reduction is possible before performance degrades?

3. Can handcrafted compression representations compete with learned representations?

4. Is hierarchy more important than raw dimensionality?

5. Can compression-based representations generalize beyond image classification?

---

## Repository Structure

```text
compression-as-representation/
│
├── experiments/
│   ├── polar_compression.py
│   ├── quadtree_mlp.py
│   └── quadtree_logistic.py
│
├── results/
│   └── benchmark_results.csv
│
├── figures/
│
├── notebooks/
│
└── README.md
```

---

## Disclaimer

This repository is an exploratory research project.

The ideas presented here are hypotheses under investigation and should not be interpreted as established scientific conclusions.

Results are intended to encourage experimentation, benchmarking, and discussion around compression-based representations.

---


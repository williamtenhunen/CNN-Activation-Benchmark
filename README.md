# Comparative Analysis of Activation Functions in CNNs

This project investigates the performance and convergence dynamics of five popular activation functions (Sigmoid, Tanh, ReLU, ELU, and SELU) within custom Convolutional Neural Network (CNN) architectures. The models were evaluated on the MNIST and CIFAR-10 datasets to identify the most suitable node function for varied image complexity.

## Model Architectures

Two distinct architectures were designed to handle the specific requirements of each dataset:

### MNIST (2-Layer CNN)

A lightweight model optimized for $28\times28$ grayscale images.

- **Layer Sequence:** $Conv(32) \rightarrow MaxPool \rightarrow Conv(64) \rightarrow MaxPool \rightarrow FC(128) \rightarrow FC(10)$
- **Purpose:** Efficiently extracts digit-specific features like edges and curves

### CIFAR-10 (3-Layer CNN)

A deeper model with an additional convolutional layer for hierarchical feature learning from $32\times32$ RGB images.

- **Layer Sequence:** $Conv(32) \rightarrow MaxPool \rightarrow Conv(64) \rightarrow MaxPool \rightarrow Conv(128) \rightarrow MaxPool \rightarrow FC(512) \rightarrow FC(10)$
- **Design:** Utilizes $3\times3$ kernels with $padding = 1$ and $2\times2$ max pooling to preserve spatial dimensions and improve computational efficiency

## Training Methodology

All experiments were conducted with the following parameters:

- **Optimizer:** Adam with a learning rate of $lr = 0.001$
- **Loss Function:** Cross-Entropy
- **Training Duration:** 10 Epochs
- **Batch Size:** 64
- **Hardware:** Google Colab T4 GPU

## Performance Benchmark

ReLU emerged as the most suitable activation function for both datasets, offering the best balance of speed and generalization.

### MNIST Detailed Performance (Epoch 10)

| Activation | Val Acc | Train Acc | Val Loss | Notes |
|------------|---------|-----------|----------|-------|
| ReLU | 99.11% | 99.77% | 0.0373 | Best overall performance |
| SELU | 99.10% | 99.62% | 0.0445 | Extremely close to ReLU |
| ELU | 98.98% | 99.64% | 0.0508 | Slight overfitting observed |
| Tanh | 98.77% | 99.72% | 0.0447 | High accuracy, slower convergence |
| Sigmoid | 98.79% | 99.56% | 0.0351 | Low loss but lower accuracy |

### CIFAR-10 Detailed Performance (Epoch 10)

| Activation | Val Acc | Train Acc | Val Loss | Notes |
|------------|---------|-----------|----------|-------|
| ReLU | 75.58% | 95.64% | 1.1663 | Best generalization |
| Tanh | 74.03% | 96.98% | 1.1105 | High training accuracy, overfits |
| ELU | 73.89% | 96.40% | 1.5038 | Stable but lower accuracy |
| SELU | 72.95% | 94.50% | 1.4025 | Large train-validation gap |
| Sigmoid | 64.89% | 72.19% | 0.9908 | Suffers from vanishing gradients |

## Key Observations

- **The ReLU Advantage:** ReLU consistently achieved the fastest convergence, reaching over 99% validation accuracy on MNIST by the 6th epoch. Its primary strength is avoiding the vanishing gradient problem.
- **Vanishing Gradients:** Sigmoid suffered severely, particularly on CIFAR-10, where it plateaued at ~65% validation accuracy because its gradients approach zero for large inputs.
- **Training Dynamics:** ELU provided smooth training with less oscillation in loss curves than ReLU. However, its marginal performance gains did not justify the extra computational cost.
- **Practicality of SELU:** Despite theoretical advantages in self-normalization, SELU failed to outperform ReLU in these specific CNN architectures.
- **Overfitting:** Tanh showed significant signs of overfitting on CIFAR-10, with a notable gap between training (97%) and validation (74%) accuracy.

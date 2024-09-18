# Breast Cancer Classification with Neural Network using PyTorch

## Overview
This project demonstrates a basic neural network for binary classification using the Breast Cancer dataset from `sklearn`. The neural network is trained to predict whether a tumor is malignant or benign based on multiple features. The project utilizes `PyTorch` for building the model and running it on a GPU if available.

## Libraries Used
- **PyTorch**: For building and training the neural network.
- **Sklearn**: For loading the breast cancer dataset and data preprocessing.
- **CUDA**: For running the model on a GPU (if available).

## Steps

### 1. Data Preprocessing
- The **Breast Cancer dataset** from `sklearn` is used. It contains 569 samples, each with 30 features.
- The dataset is split into training (80%) and testing (20%) sets using `train_test_split`.
- Data is standardized using `StandardScaler` to normalize the feature values for better training results.

### 2. Neural Network Architecture
The neural network consists of:
- An **input layer** with 30 nodes (features).
- A **hidden layer** with 64 nodes and a ReLU activation function.
- An **output layer** with 1 node (binary classification) and a Sigmoid activation function.

### 3. Training
- The network is trained using **Binary Cross-Entropy Loss (BCELoss)** and the **Adam optimizer**.
- The model runs for 100 epochs, and every 10th epoch, the loss and accuracy for the training set are printed.

### 4. Evaluation
- After training, the model's accuracy is evaluated on both the training and testing datasets to assess its performance.

## Requirements
To run this project, you need the following dependencies:

- `torch`
- `torchvision`
- `scikit-learn`
- `numpy`

You can install these via pip:

```bash 
pip install torch torchvision scikit-learn numpy
```

## Notes

- You can tweak the hyperparameters (like the number of hidden nodes, learning rate, etc.) to experiment with different model configurations.
- The script includes GPU compatibility, so if a CUDA-enabled GPU is available, the model will automatically run on it.

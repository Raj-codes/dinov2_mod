from train_logistic_regression import train_logistic_regression
import numpy as np
import torch

# Convert training data to NumPy format
train_feats = train_tensor_feats.numpy()  # PyTorch tensor to NumPy
train_labels = train_tensor_labels.numpy()  # PyTorch tensor to NumPy

# Train the logistic regression model
save_path = "logistic_regression_model.joblib"  # Optional: Path to save the model
clf = train_logistic_regression(train_feats, train_labels, max_iter=200, verbose=1, save_path=save_path)

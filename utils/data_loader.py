import numpy as np
import torch
from sklearn.model_selection import train_test_split


def load_data(features_path, labels_path):
    X = np.load(features_path)
    y = np.load(labels_path)
    return X, y


def split_data(X, y, test_size=0.2):
    return train_test_split(X, y, test_size=test_size, random_state=42)


def to_tensor(X, y):
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

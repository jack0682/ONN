# data/utils.py

from .processed_dataset import ProcessedDataset

def load_dataset(config):
    train_dataset = ProcessedDataset("data/processed/train.npz")
    val_dataset = ProcessedDataset("data/processed/val.npz")
    return train_dataset, val_dataset
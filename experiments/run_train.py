# experiments/run_train.py

import torch
import yaml
import os
from torch.utils.data import DataLoader
from models.onn import ONN
from train.trainer import Trainer
from data.utils import load_dataset

# 📄 1. Config 불러오기
with open("experiments/config.yaml", "r") as f:
    config = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config["device"] = device

# 📦 2. 데이터셋 로드
train_dataset, val_dataset = load_dataset(config)
train_loader = DataLoader(train_dataset, batch_size=config["train_batch_size"], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config["val_batch_size"], shuffle=False)

# 🧠 3. ONN 모델 초기화
model = ONN(config).to(device)

# 🏋️‍♂️ 4. Trainer 초기화 및 학습 실행
trainer = Trainer(model, config, train_loader, val_loader)
trainer.train()
# experiments/run_eval.py

import torch
import yaml
from torch.utils.data import DataLoader
from models.onn import ONN
from train.evaluator import Evaluator
from data.utils import load_dataset

# 1️⃣ Config 불러오기
with open("experiments/config.yaml", "r") as f:
    config = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config["device"] = device

# 2️⃣ 평가용 데이터셋 로딩
_, val_dataset = load_dataset(config)
val_loader = DataLoader(val_dataset, batch_size=config["val_batch_size"], shuffle=False)

# 3️⃣ 저장된 모델 로드
model = ONN(config).to(device)
model.load_state_dict(torch.load(config["save_path"], map_location=device))

# 4️⃣ 평가 실행
print("\n🔍 Running Evaluation...")
evaluator = Evaluator(model, val_loader, config)
eval_results = evaluator.evaluate()

# 5️⃣ 결과 저장(optional)
import json
with open("experiments/eval_result.json", "w") as f:
    json.dump(eval_results, f, indent=4)
print("\n✅ Evaluation results saved to eval_result.json")
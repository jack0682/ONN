# train/evaluator.py

import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from utils.metrics import (
    meaning_accuracy, 
    flow_consistency,
    relation_alignment_score,
    temporal_prediction_score
)

class Evaluator:
    def __init__(self, model, test_loader, config):
        self.model = model
        self.test_loader = test_loader
        self.config = config
        self.device = config["device"]
        self.model.eval()

    def evaluate(self):
        all_intent_preds, all_intent_targets = [], []
        all_flow_preds, all_flow_targets = [], []
        all_state_preds, all_state_targets = [], []
        all_relation_preds, all_relation_targets = [], []

        with torch.no_grad():
            for state, delta, relation, targets in self.test_loader:
                state = state.to(self.device)
                delta = delta.to(self.device)
                relation = relation.to(self.device)
                targets = {k: v.to(self.device) for k, v in targets.items()}

                output = self.model(state, delta, relation)

                # 🎯 의미 목적 예측
                all_intent_preds.append(output['intent'].argmax(dim=1).cpu().numpy())
                all_intent_targets.append(targets['intent'].cpu().numpy())

                # 🔁 의미 흐름 벡터
                if 'flow' in output:
                    all_flow_preds.append(output['flow'].cpu().numpy())
                    all_flow_targets.append(targets['flow'].cpu().numpy())

                # 📈 의미 상태 예측
                if 'state_pred' in output:
                    all_state_preds.append(output['state_pred'].cpu().numpy())
                    all_state_targets.append(targets['state'].cpu().numpy())

                # 🧩 관계 구조 예측 (선택적)
                if 'relation_pred' in output:
                    all_relation_preds.append(output['relation_pred'].cpu().numpy())
                    all_relation_targets.append(targets['relation'].cpu().numpy())

        # 🔎 평가 지표 계산
        intent_acc = accuracy_score(
            np.concatenate(all_intent_targets), 
            np.concatenate(all_intent_preds)
        )
        intent_f1 = f1_score(
            np.concatenate(all_intent_targets), 
            np.concatenate(all_intent_preds), average='weighted'
        )

        flow_acc = flow_consistency(
            np.concatenate(all_flow_preds), 
            np.concatenate(all_flow_targets)
        ) if all_flow_preds else None

        state_score = temporal_prediction_score(
            np.concatenate(all_state_preds), 
            np.concatenate(all_state_targets)
        ) if all_state_preds else None

        rel_score = relation_alignment_score(
            np.concatenate(all_relation_preds), 
            np.concatenate(all_relation_targets)
        ) if all_relation_preds else None

        print("\n📊 Evaluation Summary")
        print(f"✅ Intent Accuracy       : {intent_acc:.4f}")
        print(f"✅ Intent F1 Score       : {intent_f1:.4f}")
        if flow_acc is not None:
            print(f"🔁 Flow Consistency      : {flow_acc:.4f}")
        if state_score is not None:
            print(f"📈 State Prediction Score: {state_score:.4f}")
        if rel_score is not None:
            print(f"🧩 Relation Alignment    : {rel_score:.4f}")

        return {
            "intent_acc": intent_acc,
            "intent_f1": intent_f1,
            "flow_acc": flow_acc,
            "state_pred_score": state_score,
            "relation_score": rel_score
        }
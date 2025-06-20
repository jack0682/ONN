# train/trainer.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from models.onn import ONN
from train.loss import MeaningLoss
from torch.utils.data import DataLoader

class Trainer:
    def __init__(self, model, train_loader, val_loader, config):
        """
        Trainer 초기화
        
        :param model: ONN 모델
        :param train_loader: 훈련 데이터로더
        :param val_loader: 검증 데이터로더
        :param config: 설정 (학습률, 배치 사이즈, 에폭 등)
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # Loss 함수 정의
        self.loss_fn = MeaningLoss(lambda_sem=1.0, lambda_flow=0.5, lambda_rel=1.5, lambda_pred=1.0)
        
        # 옵티마이저 정의 (Adam)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        
        # Learning Rate Scheduler (Early Stopping, ReduceLROnPlateau)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=3, factor=0.1)

        # 학습 및 평가 모드
        self.model.train()

    def train(self):
        """
        학습 루프 실행
        """
        best_val_loss = float('inf')  # 초기화
        for epoch in range(self.config['epochs']):
            print(f"Epoch {epoch+1}/{self.config['epochs']}")
            train_loss = self._train_epoch(epoch)
            val_loss = self._validate_epoch(epoch)
            
            # Learning rate 조정
            self.scheduler.step(val_loss)
            
            # 모델 저장 (검증 손실이 최소일 때)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_model(epoch, val_loss)

            print(f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
            
    def _train_epoch(self, epoch):
        """
        하나의 에폭 동안 학습
        """
        self.model.train()  # 훈련 모드
        epoch_loss = 0
        total_samples = 0
        
        # 데이터 배치 학습
        for batch_idx, (state_tensor, delta_tensor, relation_tensor, targets) in enumerate(self.train_loader):
            # 데이터를 GPU로 보내기 (가능한 경우)
            state_tensor, delta_tensor, relation_tensor, targets = state_tensor.to(self.config['device']), \
                                                                   delta_tensor.to(self.config['device']), \
                                                                   relation_tensor.to(self.config['device']), \
                                                                   targets.to(self.config['device'])

            # 모델 예측
            predictions = self.model(state_tensor, delta_tensor, relation_tensor)
            
            # Loss 계산
            loss = self.loss_fn(predictions, targets, relation_tensor, delta_tensor, targets)
            epoch_loss += loss.item() * state_tensor.size(0)  # 배치 크기만큼 누적

            # 역전파 및 파라미터 업데이트
            self.optimizer.zero_grad()  # 기울기 초기화
            loss.backward()  # 역전파
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # 기울기 클리핑
            self.optimizer.step()  # 파라미터 업데이트

            total_samples += state_tensor.size(0)

        # 에폭 전체 손실
        epoch_loss /= total_samples
        return epoch_loss

    def _validate_epoch(self, epoch):
        """
        검증 에폭
        """
        self.model.eval()  # 평가 모드
        epoch_loss = 0
        total_samples = 0
        
        with torch.no_grad():  # 평가 시에는 기울기 계산하지 않음
            for batch_idx, (state_tensor, delta_tensor, relation_tensor, targets) in enumerate(self.val_loader):
                # 데이터를 GPU로 보내기 (가능한 경우)
                state_tensor, delta_tensor, relation_tensor, targets = state_tensor.to(self.config['device']), \
                                                                       delta_tensor.to(self.config['device']), \
                                                                       relation_tensor.to(self.config['device']), \
                                                                       targets.to(self.config['device'])

                # 모델 예측
                predictions = self.model(state_tensor, delta_tensor, relation_tensor)
                
                # Loss 계산
                loss = self.loss_fn(predictions, targets, relation_tensor, delta_tensor, targets)
                epoch_loss += loss.item() * state_tensor.size(0)  # 배치 크기만큼 누적
                total_samples += state_tensor.size(0)

        # 에폭 전체 손실
        epoch_loss /= total_samples
        return epoch_loss

    def _save_model(self, epoch, val_loss):
        """
        모델 저장
        """
        print(f"Saving model... (epoch {epoch+1}, val_loss {val_loss:.4f})")
        torch.save(self.model.state_dict(), f"onn_model_epoch_{epoch+1}_val_loss_{val_loss:.4f}.pth")

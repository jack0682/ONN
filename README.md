# 🧠 Ontology Neural Network (ONN)

> 관계 기반 의미 해석을 위한 시맨틱 텐서 추론 모델

---

## 📚 소개

Ontology Neural Network(ONN)는 사물의 "정의"를 개별 객체가 아닌, 관계, 속성, 시계열 의미 변화를 기반으로 해석하는 구조적 의미 인지 신경망입니다. 단순 분류기가 아닌 **철학적 관계론에 기반한 존재론적 인지 모델**로서 다음과 같은 원리를 따릅니다:

- 객체는 **상호작용의 집합**으로 존재한다
- 의미는 **시간-관계-기하적 구조**로 표현된다
- 명시적 언어가 아닌 **텐서 연산**으로 세계를 이해한다


## 🧠 수식적 구조 (Backbone)

### 📐 기본 속성 (Basis Semantics Tensor)

| 기호           | 의미                   | 정의                               |
|----------------|------------------------|------------------------------------|
| $\mathbb{L}$   | 위치성 (Locativeness)   | 객체의 좌표/위치 정보 (spatial)       |
| $\mathbb{B}$   | 경계성 (Boundedness)    | 객체의 물리적/기능적 경계 인지        |
| $\mathbb{F}$   | 형태성 (Formness)       | 기하학적 구조 또는 시각적 패턴 인지    |
| $\mathbb{I}$   | 의도성 (Intentionality) | 목적, 사용성, 작용성 정보             |

이들은 시간에 따라 변화하며, 의미 텐서는 다음과 같이 정의됩니다:

$$
\mathcal{S}_i(t) = [\mathbb{L}_i(t), \mathbb{B}_i(t), \mathbb{F}_i(t), \mathbb{I}_i(t)] \in \mathbb{R}^d
$$

의미 변화율은:

$$
\dot{\mathcal{S}}_i(t) = \frac{\partial \mathcal{S}_i(t)}{\partial t}
$$

그리고, 상호작용 정보는 다음과 같이 구성됩니다:

$$
I_{ij}(t) = \mathcal{G}(\mathcal{S}_i(t), \mathcal{S}_j(t), R_{ij}(t))
$$

여기서 $R_{ij}(t)$는 관계 유형에 따른 시공간적 상호작용 정보를 의미합니다.


---

## 🧮 ONN 계층 구조

### 1. 의미 임베딩

$$
E_t = \text{Embed}_\theta(\mathcal{S}_i(t), \dot{\mathcal{S}}_i(t)) \in \mathbb{R}^d
$$

- 의미 텐서를 정규화된 벡터 시퀀스로 임베딩
- Conv1D, Positional Encoding 사용 가능

### 2. 의미 흐름 인코딩

$$
H_t = \text{Encoder}(E_t) = \text{GRU}(E_t) \quad \text{or Transformer}(E_t)
$$

- 의미의 시계열적 흐름을 인코딩 (변화의 패턴 학습)

### 3. 관계 상호작용 통합

$$
Z_t = \text{Fuse}(H_t, I_{ij}(t))
$$

- Graph Attention Network (GAT) 또는 Relational Transformer 기반
- 시간적 중요도(weighted relevance) 반영:

$$
\alpha_{ij}(t) = \frac{\exp(\phi(H_i(t), H_j(t), \frac{\partial \mathcal{S}_j}{\partial t}))}{\sum_k \exp(\phi(H_i(t), H_k(t), \frac{\partial \mathcal{S}_k}{\partial t}))}
$$


### 4. 의미 추론/예측 헤드

- 목적 분류 (intent):

$$
\hat{y}_t = \text{Softmax}(W Z_t + b)
$$

- 미래 상태 예측:

$$
\hat{\mathcal{S}}_i(t+1) = f(Z_t)
$$


---

## 🎯 Loss 정의

총 Loss:

$$
\mathcal{L}_{total} = \mathcal{L}_{pred} + \lambda_1 \mathcal{L}_{flow} + \lambda_2 \mathcal{L}_{relation} + \lambda_3 \mathcal{L}_{intent}
$$

- 상태 예측 오류:

$$
\mathcal{L}_{pred} = \|\hat{\mathcal{S}}_i(t+1) - \mathcal{S}_i(t+1)\|_2^2
$$

- 의미 흐름 일관성:

$$
\mathcal{L}_{flow} = 1 - \cos(\dot{\mathcal{S}}_i(t), \dot{\hat{\mathcal{S}}}_i(t))
$$

- 관계 정렬성:

$$
\mathcal{L}_{relation} = \text{MSE}(I_{ij}^{GT}, I_{ij}^{pred})
$$

- 목적 분류 손실:

$$
\mathcal{L}_{intent} = -\sum_{c} y_c \log \hat{y}_c
$$


---

## 📈 Metric 정의

| 메트릭 이름 | 설명 |
|-------------|------|
| `meaning_accuracy` | 의미 벡터의 정확도 (tol 이하 유클리드 거리) |
| `flow_consistency` | 의미 흐름의 방향 일관성 (각도) |
| `relation_alignment_score` | 관계 정렬 점수 (normalized MSE) |
| `temporal_prediction_score` | 미래 상태 예측의 시간적 정합성 |


---

## 🏗️ 디렉토리 구조

```
ONN/
├── data/                  # 의미 텐서 데이터셋
│   ├── raw/               # 원본 시뮬레이션
│   ├── processed/         # 텐서 시계열 파일
│   └── utils.py           # 전처리/시계열화 도구
│
├── models/               # ONN 모델 구조
│   ├── embedding.py
│   ├── encoder.py
│   ├── interaction.py
│   ├── predictor.py
│   └── onn.py
│
├── train/                # 학습 및 평가 루프
│   ├── loss.py
│   ├── trainer.py
│   └── evaluator.py
│
├── utils/                # 유틸리티
│   ├── logger.py
│   ├── metrics.py
│   └── graph_tools.py
│
├── experiments/          # 실행 스크립트
│   ├── run_train.py
│   ├── run_eval.py
│   └── config.yaml
│
├── README.md
└── requirements.txt
```


---

## 💡 미래 확장 방향

- ⛓️ Graph-based symbolic chaining을 통한 reasoning
- 🧠 Dual-LOGOS 구조의 상위 계층 추론 (Meta-Inference)
- 🧬 의미 구조 자동 생성 및 카테고리 합성
- 🔍 인간/로봇의 의도성 기반 상호작용 학습


---

📌 **문의 및 제안:** `jaehong_oh@csa-lab.ai`

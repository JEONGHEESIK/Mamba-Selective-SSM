# Mamba: 선형 시간 복잡도의 Selective State Space 시퀀스 모델링

<div align="center">

[English](../README.md) | 한국어

</div>

---

## 개요

Transformer의 제곱 복잡도 어텐션 메커니즘을 선형 복잡도의 Selective State Space Models (SSMs)로 대체한 최신 아키텍처인 **Mamba**를 **구현**한 것입니다. Mamba는 Transformer와 비슷하거나 더 나은 성능을 달성하면서도 긴 시퀀스에 대해 훨씬 더 효율적입니다.

## Mamba

| 특징 | Transformer | Mamba |
|------|-------------|-------|
| **복잡도** | 레이어당 O(N²) | 레이어당 O(N) |
| **긴 시퀀스** | 메모리 병목 | 효율적 확장 |
| **추론 속도** | 자기회귀 생성 시 느림 | 빠른 상수 시간 스텝 |
| **아키텍처** | Attention + FFN (2개 서브블록) | 통합된 단일 블록 |

## 프로젝트 구조

```text
mamba/
├── ssm_basic.py          # 기본 State Space Model (SSM)
├── selective_ssm.py      # Selective SSM (핵심 혁신)
├── mamba_block.py        # 완전한 Mamba 블록 및 모델
├── test_mamba.py         # 종합 테스트 및 시각화
├── example_usage.py      # 실용 예제
└── requirements.txt      # 의존성
```

## 핵심 개념

### 1. State Space Models (SSM)

SSM은 연속 시간 상태 방정식을 사용하여 시퀀스를 모델링합니다:

```
h'(t) = A·h(t) + B·x(t)    (상태 전이)
y(t) = C·h(t)               (출력)
```

디지털 계산을 위한 이산화:
```
h_t = Ā·h_{t-1} + B̄·x_t
y_t = C·h_t
```

**주요 파라미터**:
- **A**: 상태 전이 행렬 (상태가 어떻게 진화하는지)
- **B**: 입력 행렬 (입력이 상태에 미치는 영향)
- **C**: 출력 행렬 (상태가 출력을 생성하는 방법)
- **Δ (delta)**: 시간 스케일 파라미터

### 2. Selective SSM (Mamba 핵심)

고정된 파라미터를 가진 전통적인 SSM과 달리, **Selective SSM은 B, C, Δ를 입력에 의존적으로 만듭니다**:

```python
B_t = Linear_B(x_t)      # 입력 의존적
C_t = Linear_C(x_t)      # 입력 의존적
Δ_t = Softplus(Linear_Δ(x_t))  # 입력 의존적 시간 스케일
```

**"Selective"**
- 큰 Δ → 빠른 변화 (현재 입력에 집중)
- 작은 Δ → 느린 변화 (과거 정보 기억)
- 동적 B, C → 무엇을 기억/망각할지 선택

이는 **O(N) 복잡도**로 **Attention과 같은 문맥 인식**

### 3. Mamba Block (Transformer Block 대체)

```
┌─────────────────────────────────────────────────────────┐
│ TRANSFORMER BLOCK                                       │
├─────────────────────────────────────────────────────────┤
│ LayerNorm → Multi-Head Attention (O(N²)) → Add         │
│ LayerNorm → FFN (Linear→Act→Linear) → Add              │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ MAMBA BLOCK (통합, O(N))                                 │
├─────────────────────────────────────────────────────────┤
│ LayerNorm →                                             │
│   ├─ Input Projection (확장)      [FFN과 유사]          │
│   ├─ 1D Convolution (지역 정보)   [새로운 요소]          │
│   ├─ Selective SSM                [Attention 대체]      │
│   ├─ Gating Mechanism              [FFN과 유사]          │
│   └─ Output Projection (축소)     [FFN과 유사]          │
│ → Add (residual)                                        │
└─────────────────────────────────────────────────────────┘
```

**주요 차이점**:
- **Attention → Selective SSM**: O(N²) → O(N)
- **분리된 Attention + FFN → 통합 블록**: 더 효율적
- **1D Convolution 추가**: 지역 패턴 캡처

## 한계점 및 트레이드오프

Mamba 논문과 후속 연구들에서 지적된 주요 한계점:

### 1. Content-Based Reasoning의 제약
SSM의 순차적 특성상 임의의 위치 간 직접적인 비교가 어려움. Attention은 모든 토큰 쌍 간의 유사도를 직접 계산하지만, Mamba는 압축된 상태를 통해서만 정보에 접근.

### 2. In-Context Learning 성능
Few-shot learning이나 프롬프트 기반 작업에서 Transformer 대비 성능 저하. 특히 컨텍스트 내 예제를 참조해야 하는 작업.

### 3. Discrete Copying Tasks
긴 시퀀스에서 정확한 정보 복사나 인용이 필요한 작업에서 어려움. Attention의 직접적인 토큰 참조 메커니즘이 이러한 작업에 더 적합.

### 4. 학습 역학 (Training Dynamics)
- 작은 데이터셋에서 Transformer보다 수렴이 느릴 수 있음
- 특정 초기화 전략과 학습률 스케줄링이 중요
- Warmup 단계가 더 길게 필요할 수 있음

### 5. 해석 가능성
Attention weights와 달리 SSM의 숨겨진 상태는 직관적인 해석이 어려움. 모델의 의사결정 과정을 분석하기 위한 도구가 제한적.

### 하이브리드 접근법
이러한 한계를 극복하기 위해 Mamba와 Attention을 결합한 하이브리드 아키텍처들이 제안됨:
- **Jamba** (AI21 Labs): Mamba와 Attention 레이어를 교차 배치
- **Zamba** (Zyphra): 효율성과 성능의 균형을 맞춘 하이브리드 설계

## 빠른 시작

### 설치

```bash
cd /home/jeonghs/workspace/mamba
pip install -r requirements.txt
```

### 테스트 실행

```bash
# 개별 컴포넌트 테스트
python ssm_basic.py
python selective_ssm.py
python mamba_block.py

# 시각화를 포함한 종합 테스트
python test_mamba.py

# 실용 예제
python example_usage.py
```

## 성능 및 결과

코드는 GPU가 있으면 자동으로 사용합니다 (없으면 CPU 사용).

### 1. 모델 성능 (GPU 환경 테스트)
- **모델 크기**: 4M 파라미터 (6 레이어, 256 차원)
- **추론 속도**: GPU에서 약 3,300 토큰/초
- **메모리 효율성**: 4096 시퀀스 길이에서 Transformer 대비 8배 빠름

### 2. 복잡도 비교 (d_model=512)

| 시퀀스 길이 | Transformer 연산 | Mamba 연산 | 속도 향상 |
|------------|-----------------|-----------|---------|
| 128 | 8.4M | 33.6M | 0.25x |
| 512 | 134.2M | 134.2M | 1.0x |
| 1024 | 536.9M | 268.4M | **2.0x** |
| 2048 | 2.1B | 536.9M | **4.0x** |
| 4096 | 8.6B | 1.1B | **8.0x** |

*시퀀스 길이가 길어질수록 Mamba가 점점 더 효율적입니다!*

### 3. Selective 동작 시각화

`test_mamba.py`를 실행하면 Mamba가 중요한 정보에 선택적으로 적응하는 방식을 보여주는 시각화가 생성됩니다:

<div align="center">
<img src="https://github.com/user-attachments/assets/f69dc983-142b-4ef6-8b57-a1be5a733770" width="512" height="768" ></img><br/>
</div>

플롯 설명:
1. **상단**: 중요 구간(40-60 타임스텝)이 있는 입력 신호
2. **중간**: 중요 구간에서 Delta 값 증가 (빠른 적응)
3. **하단**: 선택적 처리가 반영된 출력

## 사용 예제

### 언어 모델링

```python
from mamba_block import MambaModel

model = MambaModel(
    d_model=256,
    n_layers=6,
    vocab_size=50000,
    d_state=16
)

# 텍스트 생성
tokens = torch.randint(0, 50000, (1, 10))  # 프롬프트
logits = model(tokens)
next_token = logits[:, -1, :].argmax(dim=-1)
```

### 시퀀스 분류

```python
backbone = MambaModel(d_model=128, n_layers=4, vocab_size=None)
classifier = nn.Linear(128, num_classes)

features = backbone(x)  # (batch, seq_len, d_model)
pooled = features.mean(dim=1)  # 전역 평균 풀링
logits = classifier(pooled)
```

## 코드 하이라이트

모든 코드에는 **Transformer와 비교한 상세한 주석**이 포함되어 있습니다:

```python
# ============================================================
# 4. Selective SSM - 핵심 연산
# [Transformer의 Multi-Head Self-Attention을 대체]
# Attention: Q@K^T@V로 모든 토큰 간 상호작용 (O(N²))
# Selective SSM: 입력 의존적 상태 공간으로 시퀀스 처리 (O(N))
# ============================================================
y = self.ssm(x_conv)
```


## 참고 자료

- **Mamba 논문**: [arXiv:2312.00752](https://arxiv.org/abs/2312.00752)
- **S4 (Structured State Spaces)**: [arXiv:2111.00396](https://arxiv.org/abs/2111.00396)
- **공식 구현**: [state-spaces/mamba](https://github.com/state-spaces/mamba)

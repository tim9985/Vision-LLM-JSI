# 03_DL-and-GPU — 딥러닝 기초 및 GPU 가속 정리

`03_DL-and-GPU.ipynb`를 주제별로 재구성한 개인 학습용 노트.
각 절은 **핵심 → 코드 → 헷갈리는 지점** 순서이며, 노트북의 `TODO` 셀은 마지막 부록에 모범답안을 모아 두었습니다.

> 전체 흐름
> `퍼셉트론 → 다층 퍼셉트론 → 활성화 함수 → 학습(순전파/역전파) → CNN 원리 → PyTorch CNN(MNIST) → GPU 가속(CuPy) → GPU 가속(PyTorch) → CIFAR-10`

---

## 목차

| # | 섹션 | 핵심 키워드 |
|---|---|---|
| 1 | [퍼셉트론](#1-퍼셉트론-a) | 가중합, 편향, step function, 논리회로 |
| 2 | [다층 퍼셉트론(MLP)](#2-다층-퍼셉트론-b) | 은닉층, 선형 분리 불가능, XOR |
| 3 | [활성화 함수](#3-활성화-함수-c) | Step, Sigmoid, ReLU, 벡터화 |
| 4 | [인공신경망 학습](#4-인공신경망-학습-d) | Loss, Gradient, 역전파, 경사하강법 |
| 5 | [CNN 개요와 Feature Map](#5-cnn-개요와-feature-map-e) | Kernel, Convolution, Padding/Stride |
| 6 | [PyTorch CNN 실습 (MNIST)](#6-pytorch-cnn-실습--mnist-f) | Dataset, DataLoader, nn.Module, 학습 루프 |
| 7 | [GPU 병렬 연산 (CuPy)](#7-gpu-병렬-연산--cupy-g) | 호스트/디바이스 메모리, 동기화, 전송 비용 |
| 8 | [GPU 병렬 연산 (PyTorch)](#8-gpu-병렬-연산--pytorch-h) | Tensor, device, `.to()` |
| 9 | [MNIST GPU 가속 학습](#9-mnist-gpu-가속-학습-i) | CPU vs GPU 비교, warm-up |
| 10 | [CIFAR-10 GPU 가속 학습](#10-cifar-10-gpu-가속-학습-j) | 데이터 증강, 정규화, BatchNorm, GAP, Dropout |
| — | [부록 A. TODO 모범답안](#부록-a-todo-셀-모범답안) | |
| — | [부록 B. 자주 틀리는 지점 총정리](#부록-b-자주-틀리는-지점-총정리) | |
| — | [부록 C. 용어 사전](#부록-c-용어-사전) | |

---

## 1. 퍼셉트론 (A)

### 핵심

퍼셉트론(Perceptron)은 **인공신경망의 최소 단위**다. 입력마다 가중치를 곱해 더하고(가중합), 편향을 더한 뒤, 활성화 함수를 통과시켜 출력을 만든다.

$$ z = w_1x_1 + w_2x_2 + b \qquad y = f(z) $$

| 용어 | 의미 | 직관 |
|---|---|---|
| 입력 `x` | 외부에서 들어오는 데이터 | 재료 |
| 가중치 `w` | 각 입력이 결과에 미치는 영향력 | 재료별 중요도 |
| 가중합 | `Σ wᵢxᵢ` | 재료를 중요도대로 섞은 결과 |
| 편향 `b` | 얼마나 쉽게 활성화될지 결정하는 상수 | 발동 문턱값(threshold) |
| 활성화 함수 `f` | `z`를 최종 출력으로 변환 | 켜짐/꺼짐 판정 |

**편향의 역할이 핵심이다.** `z ≥ 0`일 때 1을 출력하므로, `b = -0.7`은 "가중합이 0.7을 넘어야 켜진다"는 뜻이다. 즉 편향의 절댓값이 클수록(음수) 발동이 까다로워진다.

### 코드

```python
import numpy as np

def step_function(z_):
    if z_ >= 0:
        return 1
    return 0

def AND(x1, x2):
    x = np.array([x1, x2])    # 입력
    w = np.array([0.5, 0.5])  # 가중치
    b = -0.7                  # 편향

    z = np.sum(x * w) + b     # 가중합 + 편향
    y = step_function(z)      # 활성화 함수
    return y
```

### 논리회로 가중치 정리 (외우지 말고 편향의 위치로 이해할 것)

| 게이트 | w1 | w2 | b | 발동 조건 |
|---|---|---|---|---|
| AND | 0.5 | 0.5 | -0.7 | 둘 다 1이어야 0.7 초과 |
| OR | 0.5 | 0.5 | -0.2 | 하나만 1이어도 0.2 초과 |
| NAND | -0.5 | -0.5 | 0.7 | AND의 부호를 전부 뒤집음 |
| NOR | -0.5 | -0.5 | 0.2 | OR의 부호를 전부 뒤집음 |
| XOR | — | — | — | **단층으로는 불가능** |

> NAND = AND의 `(w, b)` 부호 반전, NOR = OR의 `(w, b)` 부호 반전. 이 규칙만 기억하면 4개를 다 만들 수 있다.

### 헷갈리는 지점

- **노트북 셀 10의 수식은 오타다.** 본문에는 `y = 0 (z ≥ 0), 1 (z < 0)`으로 적혀 있지만, 실제 코드와 뒤쪽 셀 44의 정의는 `1 (z ≥ 0), 0 (z < 0)`이 맞다. 코드 쪽이 정답.
- `np.sum(x * w)`는 **원소별 곱 후 합**이다. 행렬곱 `x @ w`와 결과는 같지만, `*`는 element-wise, `@`는 내적이라는 차이를 확실히 구분할 것.
- `x * w`가 `[0.5, 0.5]`처럼 **배열**을 만들고, `np.sum()`이 그걸 스칼라로 줄인다. 여기에 스칼라 `b`를 더하는 것이므로 shape이 맞는다.

---

## 2. 다층 퍼셉트론 (B)

### 핵심

| 구조 | 층 구성 | 해결 가능 문제 |
|---|---|---|
| 단층 퍼셉트론 (SLP) | 입력층 – 출력층 | **선형 분리 가능** 문제만 |
| 다층 퍼셉트론 (MLP) | 입력층 – **은닉층** – 출력층 | 비선형 분리 문제 |

XOR은 2차원 평면에 `(0,0)=0, (0,1)=1, (1,0)=1, (1,1)=0`을 찍었을 때 **직선 하나로 0과 1을 나눌 수 없다.** 그래서 단층 퍼셉트론이 원리적으로 못 푼다. 은닉층을 하나 추가해 좌표를 한 번 접어주면 선형 분리가 가능해진다.

### 코드

XOR을 은닉층 2개 노드로 직접 구현한 버전:

```python
def XOR_(x1, x2):
    x = np.array([x1, x2])

    # ---- 은닉층 노드 1: NAND와 같은 역할 ----
    z1 = np.sum(x * np.array([-0.5, -0.5])) + 0.7
    y1 = step_function(z1)

    # ---- 은닉층 노드 2: OR과 같은 역할 ----
    z2 = np.sum(x * np.array([0.5, 0.5])) + (-0.2)
    y2 = step_function(z2)

    # ---- 출력층: AND와 같은 역할 ----
    z3 = np.sum(np.array([y1, y2]) * np.array([0.5, 0.5])) + (-0.7)
    return step_function(z3)
```

같은 것을 게이트 조합으로 쓰면 한눈에 보인다:

```python
# XOR(x1, x2) = AND( NAND(x1, x2), OR(x1, x2) )
def XOR(x1, x2):
    s1 = NAND(x1, x2)  # 은닉층 노드 1
    s2 = OR(x1, x2)    # 은닉층 노드 2
    return AND(s1, s2) # 출력층
```

### 헷갈리는 지점

- 은닉층 노드는 "새로운 종류의 연산"이 아니다. **똑같은 퍼셉트론을 병렬로 여러 개 둔 것**이고, 그 출력이 다음 층의 입력이 될 뿐이다.
- "층을 늘린다"와 "노드를 늘린다"는 다른 얘기다. XOR은 층(깊이)이 필요한 문제다. 단층에서 노드만 늘려도 여전히 선형 결합이라 못 푼다.

---

## 3. 활성화 함수 (C)

### 핵심

활성화 함수는 **다음 뉴런으로 신호를 얼마나 전달할지 결정**한다. 더 중요한 역할은 **비선형성 주입**이다. 활성화 함수가 없으면 층을 아무리 쌓아도 결국 하나의 선형 변환으로 축약되어 버려 층을 쌓는 의미가 사라진다.

| 함수 | 수식 | 출력 범위 | 특징 |
|---|---|---|---|
| Step | `1 if x ≥ 0 else 0` | {0, 1} | 미분값이 항상 0 → **학습 불가**. 개념 설명용 |
| Sigmoid | `1 / (1 + e^{-x})` | (0, 1) | 확률로 해석 가능. 이진 분류 출력층에 사용 |
| ReLU | `max(0, x)` | [0, ∞) | 계산이 싸고 기울기 소실이 적음. 은닉층 기본값 |

### 코드 — 벡터화가 포인트

```python
# 스칼라 하나만 처리 (층에 뉴런 100개면 100번 호출해야 함 → 비효율)
def step_function(z_):
    if z_ >= 0:
        return 1
    return 0

# NumPy 배열 전체를 한 번에 처리 (층 전체를 1회 호출로 끝냄)
def step(x):
    return np.array(x > 0, dtype=int)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)
```

### 헷갈리는 지점

- **`np.maximum` vs `np.max`**
  - `np.maximum(0, x)` — 두 배열을 **원소별로 비교**해 큰 값을 남긴다. ReLU는 이것.
  - `np.max(x)` — 배열 안에서 **가장 큰 값 하나**를 반환한다. ReLU에 쓰면 안 된다.
- `np.array(x > 0, dtype=int)`에서 `x > 0`은 bool 배열이고, `dtype=int`가 `True→1, False→0`으로 바꾼다.
- 스칼라 버전 `step_function`은 `z_ >= 0`(0 포함), 배열 버전 `step`은 `x > 0`(0 제외)이라 **`x = 0`에서 결과가 다르다.** 실습에서는 문제되지 않지만 알고는 있을 것.

---

## 4. 인공신경망 학습 (D)

### 핵심 개념 지도

```
데이터 입력
   ↓  순전파 (Forward Propagation)  ── 예측값 y_pred 생성
손실 계산 (Loss Function)          ── 정답과 얼마나 다른가
   ↓  역전파 (Backpropagation)      ── 손실의 기울기를 층을 거슬러 계산
가중치 수정 (경사하강법)            ── w ← w - η · dL/dw
   ↺  Epoch 만큼 반복
```

| 용어 | 정의 | 너무 작으면 | 너무 크면 |
|---|---|---|---|
| **Epoch** | 전체 학습 데이터를 한 번 다 보는 사이클 | 과소적합(Underfitting) | 과적합(Overfitting) |
| **학습률 η** | 한 번에 가중치를 얼마나 움직일지 | 학습이 너무 느림 | 최적점을 지나쳐 발산/불안정 |
| **손실 함수** | 예측과 정답의 차이를 수치화 | — | — |
| **Gradient** | 손실을 줄이려면 가중치를 어느 방향으로 움직여야 하는지 | — | — |

**연쇄 법칙(Chain Rule)** — 역전파의 전부다.

$$ \frac{dL}{dw} = \frac{dL}{dy} \times \frac{dy}{dz} \times \frac{dz}{dw} $$

**경사하강법(Gradient Descent)**

$$ w_{new} = w_{old} - \eta \frac{dL}{dw} $$

> 기울기의 **반대 방향**으로 가야 손실이 줄어들기 때문에 부호가 `-`다.

### 4-1. 단층 신경망 학습 (논리회로)

이진 분류이므로 **활성화 = Sigmoid, 손실 = Binary Cross-Entropy** 조합을 쓴다.

```python
def binary_cross_entropy(y_true, y_pred):
    eps = 1e-8   # log(0) = -inf 방지용
    return -np.mean(
        y_true * np.log(y_pred + eps)
        + (1 - y_true) * np.log(1 - y_pred + eps)
    )

def train_single_neuron(X, y, learning_rate=0.1, epochs=10000):
    sample_count, input_count = X.shape

    w = np.zeros((input_count, 1))
    b = np.zeros((1,))
    loss_history = []

    for epoch in range(epochs):
        # 1. 순전파
        z = X @ w + b
        y_pred = sigmoid(z)

        # 2. 손실
        loss = binary_cross_entropy(y, y_pred)
        loss_history.append(loss)

        # 3. 기울기
        error = y_pred - y                      # ← Sigmoid + BCE 조합의 결과
        w_gradient = X.T @ error / sample_count
        b_gradient = np.mean(error, axis=0)

        # 4. 업데이트
        w -= learning_rate * w_gradient
        b -= learning_rate * b_gradient

    return w, b, loss_history
```

**왜 기울기가 그냥 `y_pred - y`인가?**
Sigmoid의 미분 `σ'(z) = σ(z)(1-σ(z))`와 BCE의 미분에 들어 있는 `1/(y_pred(1-y_pred))`가 **정확히 약분**되기 때문이다. 그래서 두 함수는 항상 짝으로 쓰인다. (다중 분류에서 Softmax + Cross-Entropy가 짝인 이유도 완전히 같다.)

### 학습 결과 해석

| 게이트 | 학습된 Weight | 학습된 Bias | 예측 |
|---|---|---|---|
| AND | 양수, 양수 | 절댓값 큰 음수 | [0,0,0,1] |
| OR | 양수, 양수 | AND보다 작은 음수 | [0,1,1,1] |
| NAND | 음수, 음수 | 큰 양수 | [1,1,1,0] |
| NOR | 음수, 음수 | NAND보다 작은 양수 | [1,0,0,0] |
| **XOR** | **0에 가까움** | **0에 가까움** | **확률 ≈ 0.5, 정확도 ≈ 50%** |

> **가장 중요한 교훈**: XOR의 손실은 epoch를 아무리 늘려도 내려가지 않는다.
> **모델 구조가 문제에 맞지 않으면 데이터와 학습 시간을 늘려도 해결되지 않는다.**
> 손실 곡선이 평평하게 눕는다면 학습률/epoch가 아니라 **모델 구조**를 먼저 의심할 것.

### 4-2. 다층 신경망 학습 (XOR)

구조: 입력 2 → 은닉 4 → 출력 1

| 배열 | shape | 설명 |
|---|---|---|
| `X` | (4, 2) | 샘플 4개, 특징 2개 |
| `W1` | (2, 4) | 입력 → 은닉 |
| `b1` | (1, 4) | 은닉층 편향 (브로드캐스팅) |
| `Y1` | (4, 4) | 은닉층 출력 |
| `W2` | (4, 1) | 은닉 → 출력 |
| `b2` | (1, 1) | 출력층 편향 |
| 출력 | (4, 1) | |

```python
def train_multi_neuron(X, y, hidden_size=4, learning_rate=1.0, epochs=20000, seed=42):
    sample_count, input_count = X.shape
    rng = np.random.default_rng(seed)

    # 가중치는 반드시 '무작위'로 초기화 (0으로 두면 학습 안 됨)
    W1 = rng.normal(0.0, 1.0, size=(input_count, hidden_size))
    b1 = np.zeros((1, hidden_size))
    W2 = rng.normal(0.0, 1.0, size=(hidden_size, 1))
    b2 = np.zeros((1, 1))

    for epoch in range(epochs):
        # --- 순전파 ---
        Z1 = X @ W1 + b1
        Y1 = sigmoid(Z1)
        Z2 = Y1 @ W2 + b2
        y_pred = sigmoid(Z2)

        loss = binary_cross_entropy(y, y_pred)

        # --- 역전파: 출력층 ---
        dZ2 = (y_pred - y) / sample_count
        dW2 = Y1.T @ dZ2
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        # --- 역전파: 은닉층 (연쇄 법칙) ---
        dY1 = dZ2 @ W2.T          # 출력층 오차를 은닉층으로 되돌림
        dZ1 = dY1 * Y1 * (1 - Y1) # ← sigmoid 미분 σ(1-σ)
        dW1 = X.T @ dZ1
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        # --- 업데이트 ---
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
```

### 헷갈리는 지점

- **단층에서는 `w`를 0으로 초기화해도 되는데, 다층에서는 안 된다.**
  다층에서 모든 가중치를 0(또는 같은 값)으로 두면 은닉 노드 4개가 **완전히 똑같은 기울기**를 받아 영원히 같은 값으로 움직인다(대칭성 문제). 그래서 `rng.normal()`로 무작위 초기화한다.
- `dZ1 = dY1 * Y1 * (1 - Y1)` — 여기서 `Y1`은 **활성화 함수를 통과한 뒤의 값**이다. `Z1`이 아니다. Sigmoid 미분을 출력값으로 표현할 수 있어서 이렇게 쓴다.
- `keepdims=True`를 빼면 shape이 `(4,)`로 줄어들어 `b`와 브로드캐스팅이 어긋난다.
- `dZ2`에서 이미 `/ sample_count`로 나눴기 때문에 `dW2`에서 또 나누지 않는다. 평균을 어디서 한 번 취하는지 위치를 추적할 것.

---

## 5. CNN 개요와 Feature Map (E)

### 핵심

**합성곱 신경망(CNN)** 은 이미지처럼 **공간 구조**를 가진 데이터를 위한 신경망이다.

- 이미지를 1차원으로 펴지 않고 2D/3D 형태를 **유지**한 채 처리한다 → 주변 픽셀 간의 관계를 보존
- **커널(Kernel/Filter)** 를 이미지 위에서 이동시키며 합성곱 → **Feature Map** 생성
- **커널 1개당 Feature Map 1개**
- **Pooling** 으로 크기를 줄여 연산량을 낮추고 위치 변화에 둔감하게 만듦

### Feature Map 값의 의미

| 값 | 해석 |
|---|---|
| 큰 양수 | 필터가 찾는 패턴이 **강하게** 존재 |
| 0 근처 | 해당 특징이 거의 없음 |
| 큰 음수 | 필터 패턴과 **반대 방향** 특징이 존재 |

CNN에서는 합성곱 결과에 ReLU를 자주 적용해 **음수는 0으로 버리고 양수만 남긴다.**

### 출력 크기 공식 — 반드시 암기

$$ H_{out} = \left\lfloor \frac{H + 2P - K}{S} \right\rfloor + 1 \qquad W_{out} = \left\lfloor \frac{W + 2P - K}{S} \right\rfloor + 1 $$

`H, W`: 입력 높이/너비, `P`: padding, `K`: 커널 크기, `S`: stride

**자주 쓰는 조합**

| K | P | S | 결과 |
|---|---|---|---|
| 3 | 1 | 1 | **크기 유지** (`same`) — CNN에서 가장 흔함 |
| 3 | 0 | 1 | 양옆 1픽셀씩 줄어듦 |
| 2 | 0 | 2 | **정확히 절반** — MaxPool의 표준 설정 |

### 코드 — NumPy로 직접 구현한 2D Convolution

```python
def convolution2d(image, kernel, bias=0.0, stride=1, padding=0):
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    # 가장자리 zero padding
    padded_image = np.pad(image, ((padding, padding), (padding, padding)),
                          mode="constant", constant_values=0)

    output_height = (padded_image.shape[0] - kernel_height) // stride + 1
    output_width  = (padded_image.shape[1] - kernel_width)  // stride + 1
    output = np.zeros((output_height, output_width), dtype=np.float32)

    for output_y in range(output_height):
        for output_x in range(output_width):
            start_y = output_y * stride
            start_x = output_x * stride

            image_region = padded_image[start_y:start_y + kernel_height,
                                        start_x:start_x + kernel_width]

            output[output_y, output_x] = np.sum(image_region * kernel) + bias

    return output
```

> `padded_image.shape[0]`에는 이미 `H + 2P`가 반영되어 있어서, 공식의 `+2P`를 다시 더하지 않는다.

### CNN에서 가중치와 편향의 정체

| 신경망 일반 | CNN에서는 |
|---|---|
| 가중치 `w` | **커널(필터) 각 칸의 값** |
| 편향 `b` | 약한 특징을 무시하거나 증폭시키는 값 |

Edge 검출 실습에서는 학습 대신 값을 직접 지정했다:

```python
# x축 방향 edge (= 세로 선) 검출
dx_edge_filter = np.array([[-1, 0, 1],
                           [-1, 0, 1],
                           [-1, 0, 1]], dtype=np.float32)

# y축 방향 edge (= 가로 선) 검출
dy_edge_filter = np.array([[-1, -1, -1],
                           [ 0,  0,  0],
                           [ 1,  1,  1]], dtype=np.float32)

dx_bias = -0.2   # 0.2 이하의 약한 특징은 ReLU에서 잘려나가게 만듦
dy_bias = -0.2

input_image = img_gray.astype(np.float32) / 255.   # 0~1 실수로 변환

vertical_output = convolution2d(input_image, dx_edge_filter, dx_bias, stride=1, padding=1)
vertical_feature_map = relu(vertical_output)       # 활성화 함수 적용
```

### 부호와 ReLU의 관계 (중요)

`[[-1,0,1],[-1,0,1],[-1,0,1]]` 필터에서

- **양수 결과** = 왼쪽이 어둡고 오른쪽이 밝다
- **음수 결과** = 왼쪽이 밝고 오른쪽이 어둡다

즉 **부호는 edge의 방향 정보**를 담고 있다. ReLU를 적용하면 "오른쪽이 급격히 밝아지는 곳"만 남는다. 필터의 부호를 뒤집으면(`-dx_edge_filter`) 정확히 반대 방향의 edge만 검출된다.

### 헷갈리는 지점

- **왜 `/ 255.`로 나누나?** 0~255 정수 그대로 넣으면 가중합이 수천 단위가 되어 수치적으로 불안정해진다. 딥러닝은 입력을 0~1 또는 정규화된 범위로 맞추는 것이 기본이다.
- `bias = -0.2`는 "임계값 0.2"라는 뜻이다. ReLU와 짝을 이룰 때만 필터링 효과가 생긴다. ReLU 없이 bias만 빼면 그냥 전체가 어두워질 뿐이다.
- `padding=1`을 줘야 3×3 커널에서 입력과 출력 크기가 같아진다. 안 주면 매 층마다 이미지가 조금씩 줄어든다.

---

## 6. PyTorch CNN 실습 — MNIST (F)

### 환경 설정 (Jetson 기준)

```bash
# 버전 확인
dpkg -l | grep nvidia-jetpack
cat /usr/local/cuda/version.json

# JetPack 6.2.1 + CUDA 12.6 → torch 2.8.0 / torchvision 0.23.0 / cu126
pip install torch==2.8.0 torchvision --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 --no-deps
pip uninstall torchvision
pip install torchvision==0.23.0 --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 --no-deps
pip install "sympy>=1.13.3" --no-deps

# 모델 구조 시각화용
pip install Pillow==10.4.0 aggdraw==1.3.19 visualtorch==1.4.1 --no-deps
```

> Jetson에서는 일반 PyPI가 아니라 **jetson-ai-lab 인덱스**를 써야 한다. `--no-deps`는 의존성이 임의로 다른 버전을 덮어쓰는 것을 막기 위함이다.

### 6-1. 재현성 — 난수 고정

```python
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
```

> 세 라이브러리가 **각자 다른 난수 생성기**를 쓰기 때문에 셋 다 고정해야 한다. GPU 사용 시에는 `torch.cuda.manual_seed_all(seed)`도 필요하며, 그래도 cuDNN 알고리즘 선택 때문에 완전히 동일하지는 않을 수 있다.

### 6-2. Dataset

```python
transform = transforms.ToTensor()

train_dataset = datasets.MNIST(root="src/datasets", train=True,  transform=transform, download=True)
test_dataset  = datasets.MNIST(root="src/datasets", train=False, transform=transform, download=True)
# Train: 60,000장 / Test: 10,000장
```

`transforms.ToTensor()`가 자동으로 해 주는 두 가지:

1. **형태 변환**: `(H, W, C)` → PyTorch 표준인 `(C, H, W)`
2. **스케일링**: `0~255` 정수 → `0.0~1.0` float32

```python
image, label = train_dataset[0]
# Image shape: torch.Size([1, 28, 28])   ← (채널, 높이, 너비)
# Label: 5,  Minimum: 0.0,  Maximum: 1.0
```

### 6-3. DataLoader

| Dataset | DataLoader |
|---|---|
| 이미지 **1장 + 라벨 1개** 반환 | 여러 장을 **Batch로 묶어** 반환 |
| `train_dataset[0]` | `for images, labels in train_loader:` |

DataLoader가 제공하는 기능: Batch 생성 / Batch 단위 반복 / 데이터 순서 섞기 / 로딩 과정 관리

```python
train_loader = DataLoader(train_dataset, batch_size=64,   shuffle=True,  num_workers=0)
test_loader  = DataLoader(test_dataset,  batch_size=1000, shuffle=False, num_workers=0)

images, labels = next(iter(train_loader))
# Images shape: torch.Size([64, 1, 28, 28])   ← [Batch, Channel, Height, Width]
# Labels shape: torch.Size([64])
```

> - 학습 데이터는 `shuffle=True` (매 epoch마다 순서를 섞어 편향된 학습 방지)
> - 테스트 데이터는 `shuffle=False` (평가는 순서가 결과에 영향을 주지 않으므로 섞을 필요 없음)
> - 테스트는 역전파가 없어 메모리 여유가 있으므로 batch를 크게 잡아도 된다 (1000)

### 6-4. 모델 정의

```
[입력] 1×28×28
  ↓ Conv1 (1→8, k3 p1)     8×28×28
  ↓ ReLU                    8×28×28
  ↓ MaxPool (k2 s2)         8×14×14
  ↓ Conv2 (8→16, k3 p1)    16×14×14
  ↓ ReLU                   16×14×14
  ↓ MaxPool (k2 s2)        16× 7× 7
  ↓ Flatten                784
  ↓ FC1                     64
  ↓ FC2                     10
```

```python
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8,  kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1   = nn.Linear(16 * 7 * 7, 64)
        self.fc2   = nn.Linear(64, 10)

    def forward(self, x, return_features=False):
        conv1_output = self.conv1(x)
        relu1_output = F.relu(conv1_output)
        pool1_output = self.pool(relu1_output)
        conv2_output = self.conv2(pool1_output)
        relu2_output = F.relu(conv2_output)
        pool2_output = self.pool(relu2_output)
        flattened    = torch.flatten(pool2_output, start_dim=1)
        fc1_output   = F.relu(self.fc1(flattened))
        logits       = self.fc2(fc1_output)

        if return_features:
            features = {"conv1": conv1_output, "relu1": relu1_output, "pool1": pool1_output,
                        "conv2": conv2_output, "relu2": relu2_output, "pool2": pool2_output}
            return logits, features
        return logits
```

**파라미터 수 계산 (직접 해 볼 것)**

| Layer | 계산 | 개수 |
|---|---|---|
| conv1 | `1×8×3×3 + 8` | 80 |
| conv2 | `8×16×3×3 + 16` | 1,168 |
| fc1 | `784×64 + 64` | 50,240 |
| fc2 | `64×10 + 10` | 650 |
| **합계** | | **52,138** |

> Convolution은 파라미터가 아주 적고(80개, 1168개), **Fully Connected가 전체의 97%를 차지**한다. 뒤에 나오는 CIFAR-10 모델에서 Global Average Pooling으로 FC를 줄이는 이유가 바로 이것이다.

```python
parameter_count = sum(p.numel() for p in model.parameters())
trainable_parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
```

**`16 * 7 * 7`이 어디서 나왔나?** 입력 28×28 → MaxPool 2회 → 28/2/2 = 7. 채널은 conv2의 out_channels인 16. 그래서 Flatten 후 크기가 `16×7×7 = 784`. **이 숫자는 입력 크기가 바뀌면 반드시 다시 계산해야 한다.**

### 6-5. 순전파 확인

```python
dummy_input  = torch.randn(1, 1, 28, 28)   # 학습 전 shape 검증용 더미 입력
dummy_output = model(dummy_input)
# Input shape:  torch.Size([1, 1, 28, 28])
# Output shape: torch.Size([1, 10])
```

Batch 하나로 각 단계 shape을 추적하면 구조가 명확해진다:

```
Input:  [64, 1, 28, 28]
conv1:  [64, 8, 28, 28]     relu1: [64, 8, 28, 28]     pool1: [64, 8, 14, 14]
conv2:  [64,16, 14, 14]     relu2: [64,16, 14, 14]     pool2: [64,16,  7,  7]
Logits: [64, 10]
```

### 6-6. Feature Map / 필터 시각화

```python
def show_feature_maps(feature_tensor, title, max_maps=8, cmap="gray"):
    if feature_tensor.ndim != 4:
        raise ValueError("Feature Tensor는 [Batch, Channel, Height, Width] 형태여야 합니다.")

    feature_maps = feature_tensor[0].detach().cpu()   # 배치 첫 장만, 그래프에서 분리, CPU로
    channel_count = min(feature_maps.shape[0], max_maps)
    ...

image, label = test_dataset[0]
input_batch = image.unsqueeze(0).to(device)   # [1,28,28] → [1,1,28,28] Batch 차원 추가

model.eval()
with torch.no_grad():
    logits, features = model(input_batch, return_features=True)
```

**`.detach().cpu()`의 의미**

| 메서드 | 하는 일 | 왜 필요한가 |
|---|---|---|
| `.detach()` | 자동 미분 그래프에서 분리 | 시각화용 값에 gradient를 추적할 필요가 없음 |
| `.cpu()` | GPU → CPU 메모리로 복사 | matplotlib/NumPy는 GPU 텐서를 못 읽음 |
| `.numpy()` | NumPy 배열로 변환 | (CPU 텐서에만 가능) |

**학습 전 vs 학습 후**
- 학습 **전**: 필터가 무작위라 Feature Map에 의미 있는 패턴이 없다.
- 학습 **후**: 선, 방향, 경계에 반응하는 필터가 만들어진다. 2번째 층은 1번째 층의 특징을 **조합**해 획이나 모양 일부에 반응한다.

```python
conv1_weights = model.conv1.weight.detach().cpu()
print(conv1_weights.shape)   # torch.Size([8, 1, 3, 3]) — 3×3 필터 8개
```

### 6-7. 손실 함수와 Optimizer

```python
criterion = nn.CrossEntropyLoss()                       # 다중 클래스 분류
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

> ⚠️ **`nn.CrossEntropyLoss`에는 Softmax를 거치지 않은 raw logit을 그대로 넣는다.**
> 내부에 `LogSoftmax + NLLLoss`가 이미 들어 있다. 앞에 `softmax`를 직접 붙이면 **이중 적용**이 되어 학습이 망가진다.
> 확률값이 보고 싶을 때만 **추론 단계에서** `torch.softmax(logits, dim=1)`를 쓴다.

### 6-8. 학습 루프

Batch 1개의 학습 순서:
1. 이미지와 Label을 device로 이동
2. **기존 Gradient 초기화**
3. 순전파
4. Loss 계산
5. 역전파
6. Weight 업데이트

```python
def train_one_epoch(model, data_loader, criterion, optimizer, device):
    model.train()                    # 학습 모드

    total_loss, correct_count, sample_count = 0.0, 0, 0

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()        # 이전 Batch의 Gradient 제거 (필수!)

        logits = model(images)       # 순전파
        loss = criterion(logits, labels)
        loss.backward()              # 역전파 → 각 파라미터의 .grad 채움
        optimizer.step()             # .grad를 이용해 파라미터 수정

        batch_size = images.size(0)
        total_loss    += loss.item() * batch_size          # 평균 loss × batch 크기 = 합
        predictions    = logits.argmax(dim=1)
        correct_count += (predictions == labels).sum().item()
        sample_count  += batch_size

    return total_loss / sample_count, correct_count / sample_count
```

```python
def evaluate(model, data_loader, criterion, device):
    model.eval()                     # 평가 모드

    total_loss, correct_count, sample_count = 0.0, 0, 0

    with torch.no_grad():            # Gradient 계산 비활성화
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)

            batch_size = images.size(0)
            total_loss    += loss.item() * batch_size
            correct_count += (logits.argmax(dim=1) == labels).sum().item()
            sample_count  += batch_size

    return total_loss / sample_count, correct_count / sample_count
```

### 학습 루프 4대 필수 요소 — 하나라도 빠지면 조용히 망가진다

| 코드 | 빠뜨리면 |
|---|---|
| `optimizer.zero_grad()` | Gradient가 **누적**되어 파라미터가 폭주 |
| `loss.backward()` | Gradient가 계산되지 않아 학습이 전혀 안 됨 |
| `optimizer.step()` | Gradient는 계산되지만 파라미터가 안 바뀜 |
| `model.train()` / `model.eval()` | Dropout·BatchNorm이 잘못된 모드로 동작 |

**`model.train()` vs `model.eval()`은 단순한 플래그가 아니다.**

| Layer | train 모드 | eval 모드 |
|---|---|---|
| `Dropout` | 무작위로 뉴런 비활성화 | **전부 활성화** |
| `BatchNorm` | 현재 batch의 평균/분산 사용 + 통계 갱신 | **누적된 이동 평균 사용** |

> MNIST의 `SimpleCNN`에는 둘 다 없어서 차이가 안 보이지만, CIFAR-10 모델에는 둘 다 있으므로 **모드 전환을 빼먹으면 정확도가 눈에 띄게 떨어진다.**

**`with torch.no_grad()`와 `model.eval()`은 다른 것이다.**
- `model.eval()` — Dropout/BatchNorm의 **동작 방식**을 바꾼다
- `torch.no_grad()` — **Gradient 그래프 생성을 막아** 메모리와 속도를 아낀다
- 평가할 때는 **둘 다** 써야 한다.

### 6-9. 전체 학습 실행

```python
epochs = 3
train_loss_history, train_accuracy_history = [], []
test_loss_history,  test_accuracy_history  = [], []

training_start_time = time.perf_counter()

for epoch in range(epochs):
    epoch_start_time = time.perf_counter()

    train_loss, train_accuracy = train_one_epoch(model, train_loader, criterion, optimizer, device)
    test_loss,  test_accuracy  = evaluate(model, test_loader, criterion, device)

    train_loss_history.append(train_loss);     train_accuracy_history.append(train_accuracy)
    test_loss_history.append(test_loss);       test_accuracy_history.append(test_accuracy)

    epoch_time = time.perf_counter() - epoch_start_time
    print(f"Epoch {epoch+1}/{epochs} | Train loss: {train_loss:.4f} | "
          f"Train accuracy: {train_accuracy*100:.2f}% | Test loss: {test_loss:.4f} | "
          f"Test accuracy: {test_accuracy*100:.2f}% | Time: {epoch_time:.2f}s")

print(f"Total CPU training time: {time.perf_counter() - training_start_time:.2f}s")
```

### 6-10. 추론

```python
image, true_label = test_dataset[0]
input_batch = image.unsqueeze(0).to(device)   # Batch 차원 추가

model.eval()
with torch.no_grad():
    logits        = model(input_batch)
    probabilities = torch.softmax(logits, dim=1)          # 여기서만 softmax
    predicted_label = probabilities.argmax(dim=1).item()

print("Confidence:", probabilities[0, predicted_label].item())
```

> `dim=1`인 이유: logits의 shape이 `[Batch, 10]`이므로 **클래스 축(1번 축)** 을 따라 softmax/argmax를 해야 한다. `dim=0`으로 하면 배치 방향으로 계산되어 완전히 틀린다.

### 6-11. 모델 저장/불러오기

```python
# 저장 — 가중치(state_dict)만 저장하는 것이 표준
torch.save(model.state_dict(), "src/models/MNIST/MNIST_CNN.pth")

# 불러오기 — 모델 구조는 코드로 다시 만들어야 한다
loaded_model = SimpleCNN()
state_dict = torch.load("src/models/MNIST/MNIST_CNN.pth", map_location="cpu", weights_only=True)
loaded_model.load_state_dict(state_dict)
loaded_model.to(device)
loaded_model.eval()          # 불러온 뒤 반드시 eval 모드로
```

| 인자 | 의미 |
|---|---|
| `map_location="cpu"` | GPU에서 저장한 모델을 CPU 환경에서도 열 수 있게 함 |
| `weights_only=True` | 텐서/기본 자료형만 읽어 **임의 코드 실행을 차단**. 보안상 권장 |

> **`state_dict`는 구조가 아니라 가중치만 담고 있다.** 불러오기 전에 반드시 같은 클래스를 정의해 인스턴스를 만들어야 한다.

### 헷갈리는 지점

- `unsqueeze(0)` ↔ `squeeze(0)`
  - `unsqueeze(0)`: `[1,28,28]` → `[1,1,28,28]` — 모델에 넣기 위해 **Batch 차원 추가**
  - `squeeze(0)`: `[1,28,28]` → `[28,28]` — `plt.imshow`에 넣기 위해 **채널 차원 제거**
- `loss.item() * batch_size` — `criterion`이 반환하는 loss는 **batch 평균**이다. 전체 평균을 정확히 구하려면 배치 크기를 곱해 합으로 되돌린 뒤 총 샘플 수로 나눠야 한다. 마지막 배치 크기가 다를 수 있어 그냥 평균의 평균을 내면 틀린다.
- `.item()`은 원소가 **하나**인 텐서를 Python 숫자로 꺼낸다. 텐서 그대로 리스트에 쌓으면 그래프가 메모리에 계속 남는다.

---

## 7. GPU 병렬 연산 — CuPy (G)

### 핵심

`CuPy`는 **NumPy와 호환되는 GPU 배열 라이브러리**다. 문법은 거의 같지만 **연산이 실행되는 물리적 장치**가 다르다.

| | NumPy | CuPy |
|---|---|---|
| 실행 장치 | CPU 코어 (Jetson Orin Nano: 6개) | GPU 코어 (Jetson Orin Nano: 1024개) |
| 데이터 위치 | Host memory (RAM) | Device memory (VRAM) |
| 강점 | 복잡한 제어 흐름, 순차 처리 | **대규모 단순 연산의 병렬 처리** |
| 타입 | `numpy.ndarray` | `cupy.ndarray` |

### 설치 (JetPack 6.2 / CUDA 12.6 / NumPy 1.21.5 기준)

```bash
python -m pip download --no-deps --only-binary=:all: "cupy-cuda12x==12.3.0" -d ~/Downloads
python -m pip install --no-deps "fastrlock==0.8.3"
python -m pip install --no-deps --only-binary=:all: ~/Downloads/cupy_cuda12x-12.3.0-*.whl
```

### 메모리 이동

```python
x_cpu = np.array([1, 2, 3, 4])   # RAM
x_gpu = cp.array([1, 2, 3, 4])   # VRAM

x_from_cpu_to_gpu = cp.asarray(x_cpu)    # CPU → GPU
x_from_gpu_to_cpu = cp.asnumpy(x_gpu)    # GPU → CPU

# 서로 다른 장치의 배열은 직접 연산 불가
try:
    y = x_cpu + x_gpu
except RuntimeError as e:
    print(f"잘못된 연산: {e}")
```

> **Jetson의 통합 메모리(UMA)에서도 마찬가지다.** Jetson Orin Nano는 CPU와 GPU가 물리적으로 같은 메모리 칩을 공유하지만, **논리적으로는 서로 다른 영역을 각자 관리**한다. 그래서 여전히 명시적 복사가 필요하다.

### 동기화 — 시간 측정의 함정

```python
cp.cuda.Stream.null.synchronize()      # 이전 GPU 작업이 끝날 때까지 대기
start = time.perf_counter()

y_gpu = x_gpu ** 2

cp.cuda.Stream.null.synchronize()      # 측정 종료 전에도 반드시 대기
end = time.perf_counter()
```

> **GPU 연산은 비동기(asynchronous)로 실행된다.** CPU는 GPU에 작업을 던져 놓고 곧바로 다음 줄로 넘어간다. 동기화 없이 시간을 재면 **"작업을 지시하는 데 걸린 시간"** 만 측정되어 말도 안 되게 빠른 결과가 나온다.
>
> PyTorch에서의 대응 함수는 `torch.cuda.synchronize()`다.

### 실습 1 — 이미지 잔상(Motion Trail) 효과

**원리**: 원본을 오른쪽으로 조금씩 이동한 복사본들을 만들고, 이동 거리가 멀수록 작은 가중치를 곱해 전부 더한다.

```text
입력 배열:  [10, 20, 30, 40, 50]
가중치:     [0.6, 0.3, 0.1]

이동 없음:  [10, 20, 30, 40, 50] × 0.6
오른쪽 1칸: [ 0, 10, 20, 30, 40] × 0.3
오른쪽 2칸: [ 0,  0, 10, 20, 30] × 0.1
                 ↓ 같은 위치끼리 합산
```

**CPU와 GPU의 처리 방식 차이 — 이 절의 핵심**

| | 반복 구조 | 병렬화되는 부분 |
|---|---|---|
| **CPU (NumPy)** | 출력 픽셀을 **하나씩** 순차 계산 (`for y: for x: for shift:`) | 없음 |
| **GPU (CuPy)** | **이동 횟수만** 순차 반복 (`for shift:`) | 각 단계의 **이미지 전체 연산**을 병렬 처리 |

> GPU 가속의 본질은 "반복문을 없애는 것"이 아니라, **가장 안쪽의 픽셀 단위 반복을 배열 연산으로 바꿔치기하는 것**이다. 32번의 shift 반복은 그대로 남지만, 각 반복이 1280×720 = 92만 픽셀을 한 번에 처리한다.

```python
weights_np = np.linspace(1.0, 0.05, TRAIL_LENGTH, dtype=np.float32)
weights_np /= np.sum(weights_np)     # 가중치 합을 1로 정규화 → 밝기 유지
```

### 전송 비용 — 반드시 짚고 넘어갈 것

```python
compute_speedup = cpu_time / gpu_time                          # 연산만 비교
gpu_total_time  = upload_time + gpu_time + download_time
total_speedup   = cpu_time / gpu_total_time                    # 전송 포함
```

> **`compute_speedup`과 `total_speedup`은 크게 다르다.** GPU 연산 자체는 수십 배 빨라도, CPU↔GPU 데이터 전송 시간을 포함하면 이득이 확 줄어든다.
>
> **실무 원칙**: 데이터를 GPU에 한 번 올리고 **거기서 최대한 많은 연산을 끝낸 뒤** 결과만 내려받는다. 매 연산마다 왕복시키면 GPU를 쓰는 의미가 없다.

### GPU Warm-up

```python
warmup_image   = cp.zeros((64, 64), dtype=cp.float32)
warmup_weights = cp.asarray(weights_np[:4])
_ = motion_trail_cp(image=warmup_image, weights=warmup_weights)
cp.cuda.Stream.null.synchronize()
```

> GPU는 **첫 연산에서** CUDA Context 생성, cuDNN 초기화, 메모리 할당 등 준비 작업을 한다. 이 초기화 시간이 측정에 섞이면 첫 결과가 실제보다 훨씬 느리게 나온다. 그래서 측정 전에 같은 형태의 연산을 몇 번 미리 돌린다.

### 실습 2 — Convolution의 반복 순서 뒤집기

| | 반복 대상 | 한 번의 반복에서 |
|---|---|---|
| `convolution2d` (NumPy) | **출력 픽셀** (H×W번) | 커널 크기 영역을 잘라 가중합 → 픽셀 1개 생성 |
| `convolution2d_cp` (CuPy) | **커널 원소** (3×3 = 9번) | 모든 출력 위치의 대응 픽셀에 가중치를 곱해 **출력 배열 전체에 누적** |

```text
[NumPy]  출력 위치 하나 선택 → 커널 영역 추출 → 가중합 → 출력 픽셀 하나 저장
[CuPy]   커널 원소 하나 선택 → 모든 출력 위치의 대응 입력 픽셀 선택
                            → 배열 전체에 가중치 적용 → 출력 배열 전체에 누적
```

**계산 순서는 완전히 다르지만 최종 결과는 동일하다.** 92만 번 반복이 9번 반복으로 줄어드는 것이 가속의 정체다.

### 헷갈리는 지점

- **`np.sum(cupy_array)`는 쓰지 말 것.** 노트북 셀 284에서 CuPy 배열에 NumPy 함수를 쓰는 예가 나오는데, 동작은 하지만 반환 타입이 헷갈리게 된다. **CuPy 배열에는 `cp.` 함수를 쓰고, Python 숫자가 필요하면 `.item()`을 붙인다.**
  ```python
  total = cp.sum(x_gpu).item()   # ← 이렇게
  ```
- `cv2.imshow()`와 `plt.imshow()`는 **NumPy 배열만** 받는다. CuPy 결과는 반드시 `cp.asnumpy()`로 내려받고 넘겨야 한다.
- `cp.asarray()` vs `cp.array()` — `asarray`는 이미 CuPy 배열이면 복사하지 않고, `array`는 항상 복사한다. 전송에는 `asarray`가 낫다.

---

## 8. GPU 병렬 연산 — PyTorch (H)

### 핵심

`PyTorch`는 GPU 연산 + **자동 미분** + 신경망 구성을 모두 지원하는 딥러닝 프레임워크다. CuPy가 "GPU용 NumPy"라면 PyTorch는 "GPU용 NumPy + 학습 엔진"이다.

| 모듈 | 역할 |
|---|---|
| `torch.Tensor` | 다차원 배열 (NumPy `ndarray`의 GPU 버전) |
| `torch.autograd` / `torch.no_grad` | 자동 미분과 Gradient 계산 제어 |
| `torch.nn` / `torch.nn.functional` | 신경망 Layer와 Loss Function |
| `torch.optim` | SGD, Adam 등 Optimizer |
| `torch.utils.data` | Dataset과 DataLoader |
| `torchvision` | 이미지 Dataset, Transform, 사전 학습 모델 |

### Device 지정 — 실무 표준 패턴

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("CUDA Availability:", torch.cuda.is_available())
print("Selected device:", device)
```

> 이렇게 쓰면 **GPU가 없는 환경에서도 코드가 그대로 돈다.** `torch.device("cuda")`를 하드코딩하면 CPU 머신에서 바로 터진다.

### ⚠️ 모델과 Tensor는 반드시 같은 Device에 있어야 한다

```python
model = SimpleCNN().to(device)                  # 모델을 GPU로
images, labels = next(iter(train_loader))       # 데이터는 아직 CPU에!

try:
    outputs = model(images)                     # RuntimeError 발생
except RuntimeError as e:
    print(f"에러: {e}")

# 해결
images = images.to(device)
labels = labels.to(device)
```

> 이것이 PyTorch 초보자가 **가장 자주 만나는 에러**다.
> `Expected all tensors to be on the same device...` 메시지를 보면 `.to(device)`를 빠뜨린 곳을 찾으면 된다.
>
> 학습 루프에서 `.to(device)`가 필요한 3곳: **모델 1회**, **images 매 배치**, **labels 매 배치**.

### `.to(device)`의 미묘한 차이

| 대상 | 동작 |
|---|---|
| **모델** | **제자리(in-place)** 로 이동. `model.to(device)`만 써도 되지만 관례상 `model = model.to(device)` |
| **텐서** | **새 텐서를 반환**. `images.to(device)`만 쓰면 아무 일도 안 일어난다. 반드시 `images = images.to(device)` |

---

## 9. MNIST GPU 가속 학습 (I)

### 학습 구조 복습 (Batch 1개 기준)

1. 이미지와 Label 준비 (device로 이동)
2. 순전파로 결과 예측
3. Loss 계산
4. 기존 Gradient 초기화
5. 역전파 수행
6. Optimizer로 Weight 수정

### CPU 버전과 GPU 버전의 차이 — 딱 4가지

| # | 항목 | CPU | GPU |
|---|---|---|---|
| 1 | device | `torch.device("cpu")` | `torch.device("cuda")` |
| 2 | 동기화 | 불필요 | 시간 측정 전후로 `torch.cuda.synchronize()` |
| 3 | Warm-up | 불필요 | 측정 전에 더미 순전파 몇 번 |
| 4 | DataLoader | — | `pin_memory=True` (전송 속도 향상) |

> **`train_one_epoch()`과 `evaluate()` 함수는 한 글자도 안 바꿔도 된다.** 이미 `device`를 인자로 받아 `.to(device)`를 하도록 설계했기 때문이다. 이것이 device를 인자로 빼는 이유다.

### 상수 정의 (main 함수 스타일)

```python
SEED = 42
DATA_ROOT = "src/datasets"
BATCH_SIZE = 64
TEST_BATCH_SIZE = 1000
EPOCHS = 3
LEARNING_RATE = 0.001
NUM_WORKERS = 0
```

### 결과 비교

```python
if __name__ == "__main__":
    cpu_training_time = main_cpu()
    gpu_training_time = main_gpu()

    speedup = cpu_training_time / gpu_training_time
    print(f"GPU speedup: {speedup:.2f}x")
```

> MNIST의 `SimpleCNN`은 파라미터가 5만 개 정도로 매우 작아서 **GPU 가속 효과가 생각보다 크지 않을 수 있다.** 오히려 배치마다 CPU↔GPU 전송이 일어나 병목이 된다. GPU의 진가는 다음 절의 CIFAR-10처럼 **파라미터가 100만 개 이상인 모델**에서 드러난다.

---

## 10. CIFAR-10 GPU 가속 학습 (J)

### MNIST vs CIFAR-10

| | MNIST | CIFAR-10 |
|---|---|---|
| 이미지 | 1×28×28 (흑백) | **3×32×32 (RGB)** |
| 내용 | 손글씨 숫자 | 비행기, 자동차, 새, 고양이 등 실사물 |
| 학습 데이터 | 60,000 | **50,000** |
| 테스트 데이터 | 10,000 | 10,000 |
| 배경 | 단순, 객체가 중앙 | **복잡, 위치·방향 다양** |
| 난이도 | 낮음 | 높음 |

### 10-1. 데이터 전처리 — 증강과 정규화

| 기법 | 목적 | 적용 대상 |
|---|---|---|
| **데이터 증강** (Augmentation) | 이미지를 매번 조금씩 변형해 다양성 확보 → **과적합 완화** | **학습 데이터만** |
| **정규화** (Normalization) | 채널별 픽셀 분포를 통일 → Gradient/Optimizer 안정화 | **학습 + 테스트 모두** |

```python
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)   # PyTorch 공식 제공 채널별 평균
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)   # 채널별 표준편차

train_transform = transforms.Compose([
    transforms.RandomCrop(size=32, padding=4),    # 상하좌우로 조금씩 이동한 효과
    transforms.RandomHorizontalFlip(p=0.5),       # 50% 확률로 좌우 반전
    transforms.ToTensor(),
    transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),                        # 증강 없음!
    transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
])
```

> **`Compose`의 순서가 중요하다.** `RandomCrop`/`RandomHorizontalFlip`은 PIL 이미지에 동작하므로 `ToTensor()` **앞**에, `Normalize`는 텐서에 동작하므로 `ToTensor()` **뒤**에 와야 한다.
>
> **증강을 테스트에 적용하면 안 되는 이유**: 평가는 매번 같은 조건에서 이뤄져야 재현 가능한 성능 측정이 된다. 반면 **정규화는 학습·테스트에 똑같이** 적용해야 모델이 같은 분포의 입력을 받는다.

### 10-2. 정규화 후 픽셀 값 범위

```python
image, label = train_dataset[0]
# Image shape: torch.Size([3, 32, 32])
# Minimum/Maximum: 약 -2.0 ~ 2.0   ← 0~1이 아니다!
```

`(픽셀 - mean) / std`를 적용하면 채널별 범위가 이렇게 된다:

| 채널 | 범위 |
|---|---|
| R | -1.989 ~ 2.059 |
| G | -1.980 ~ 2.126 |
| B | -1.707 ~ 2.116 |

### 10-3. 시각화를 위한 역정규화 (필수)

```python
# 그냥 그리면 경고: "Clipping input data to the valid range for imshow..."
image_rgb = image.permute(1, 2, 0)     # ❌ 범위가 [0,1]을 벗어남

# 역정규화 후 그리기
mean = torch.tensor(CIFAR10_MEAN).view(3, 1, 1)
std  = torch.tensor(CIFAR10_STD).view(3, 1, 1)

image_denormalized = image * std + mean                          # 정규화의 역연산
image_rgb = image_denormalized.permute(1, 2, 0).clamp(0, 1)      # (C,H,W) → (H,W,C)
plt.imshow(image_rgb)
```

| 코드 | 이유 |
|---|---|
| `.view(3, 1, 1)` | `(3,)` 튜플을 `(3,1,1)`로 만들어 `(3,32,32)` 텐서와 **채널별 브로드캐스팅** |
| `image * std + mean` | `(x - mean)/std`의 역연산 |
| `.permute(1, 2, 0)` | PyTorch `(C,H,W)` → matplotlib `(H,W,C)` |
| `.clamp(0, 1)` | 부동소수 오차로 살짝 벗어난 값을 잘라냄 |

### 10-4. DataLoader — Batch를 줄이는 이유

```python
train_loader = DataLoader(train_dataset, batch_size=32,  shuffle=True,
                          num_workers=0, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=128, shuffle=False,
                          num_workers=0, pin_memory=True)
```

> MNIST(64)보다 batch를 줄인 이유: 이미지가 더 크고(32×32×3 vs 28×28×1) 모델의 채널 수도 훨씬 많아 **GPU 메모리 사용량이 급증**하기 때문이다. `pin_memory=True`는 CPU 메모리를 페이지 고정해 GPU 전송을 빠르게 한다(GPU 사용 시에만 의미 있음).

### 10-5. 모델 구조

```
[입력] 3×32×32
  ↓ ConvBlock 1 (3→64)      64×16×16
  ↓ ConvBlock 2 (64→128)   128× 8× 8
  ↓ ConvBlock 3 (128→256)  256× 4× 4
  ↓ Global Average Pooling 256× 1× 1
  ↓ Flatten                256
  ↓ Dropout(0.3)           256
  ↓ Fully Connected         10
```

각 `ConvBlock`의 내부:

```
Conv → BatchNorm → ReLU → Conv → BatchNorm → ReLU → MaxPool
```

**채널 수를 늘려 가는 이유**

| 채널 | 학습하는 특징 |
|---|---|
| 64 | 비교적 단순한 색상과 경계 |
| 128 | 질감과 부분적인 형태 |
| 256 | 객체를 구분하기 위한 복잡한 특징 |

**새로 등장한 Layer 3종**

| Layer | 역할 |
|---|---|
| `BatchNorm2d` | Convolution 출력 분포를 조정해 **학습 안정화**. 각 batch의 Feature Map을 정규화 |
| `AdaptiveAvgPool2d` | Feature Map의 공간 영역을 평균 하나로 압축 (`256×4×4 → 256×1×1`). **FC 파라미터 수 대폭 감소** |
| `Dropout` | 학습 중 일부 Feature를 무작위 비활성화. **과적합 완화** |

```python
class ConvBlock(nn.Module):
    """Conv2d → BatchNorm2d → ReLU → Conv2d → BatchNorm2d → ReLU → MaxPool2d"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ]
        self.block = nn.Sequential(*layers)   # 리스트를 하나의 Sequential로

    def forward(self, x):
        return self.block(x)
```

> **`bias=False`인 이유**: 바로 뒤에 `BatchNorm2d`가 온다. BatchNorm이 평균을 빼는 순간 Conv의 편향은 **완전히 상쇄되어 사라진다.** 아무 효과 없는 파라미터이므로 아예 두지 않는다. **`Conv + BatchNorm` 조합에서는 항상 `bias=False`가 관례다.**

```python
class CIFAR10CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(in_channels=3,   out_channels=64),    # → [B,  64, 16, 16]
            ConvBlock(in_channels=64,  out_channels=128),   # → [B, 128,  8,  8]
            ConvBlock(in_channels=128, out_channels=256),   # → [B, 256,  4,  4]
        )
        self.global_average_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))  # → [B, 256, 1, 1]
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features=256, out_features=10),
        )

    def forward(self, x):                       # [B, 3, 32, 32]
        x = self.features(x)                    # [B, 256, 4, 4]
        x = self.global_average_pool(x)         # [B, 256, 1, 1]
        x = torch.flatten(x, start_dim=1)       # [B, 256]
        logits = self.classifier(x)             # [B, 10]
        return logits
```

**파라미터 수 비교 — Global Average Pooling의 위력**

| 모델 | 총 파라미터 |
|---|---|
| `SimpleCNN` (MNIST) | 약 5.2만 |
| `CIFAR10CNN` | 약 **114.9만** |

> GAP를 쓰지 않고 `256×4×4 = 4096`을 그대로 Linear에 넣었다면 FC 하나만으로 `4096×10 + 10 = 40,970`개가 필요했다. GAP를 거치면 `256×10 + 10 = 2,570`개로 **16배 줄어든다.** 게다가 GAP는 학습 파라미터가 **0개**다.

### 10-6. GPU 전용 실행

```python
if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU를 사용할 수 없습니다.")

device = torch.device("cuda")
model = CIFAR10CNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)   # Adam이 아니라 AdamW
```

> **Adam vs AdamW**: AdamW는 weight decay(가중치 감쇠)를 Adam보다 올바른 방식으로 적용한다. 파라미터가 많고 과적합이 우려되는 큰 모델에서는 AdamW가 기본 선택이다.

### 10-7. GPU Warm-up

```python
warmup_input = torch.zeros(64, 3, 32, 32, device=device)

model.eval()
with torch.no_grad():
    for _ in range(3):
        _ = model(warmup_input)

torch.cuda.synchronize()
```

> CuPy의 `cp.cuda.Stream.null.synchronize()`에 대응하는 PyTorch 함수가 `torch.cuda.synchronize()`다. 이유는 동일하다 — CUDA Context 생성, cuDNN 초기화, 메모리 할당 시간을 측정에서 제외하기 위함.

### 10-8. 학습 지표 시각화

```python
epoch_axis = range(1, EPOCHS + 1)

plt.plot(epoch_axis, train_loss_history, marker="o", label="Train loss")
plt.plot(epoch_axis, test_loss_history,  marker="o", label="Test loss")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("CNN Loss")
plt.grid(); plt.legend(); plt.show()

plt.plot(epoch_axis, np.array(train_accuracy_history) * 100, marker="o", label="Train accuracy")
plt.plot(epoch_axis, np.array(test_accuracy_history)  * 100, marker="o", label="Test accuracy")
plt.xlabel("Epoch"); plt.ylabel("Accuracy (%)"); plt.title("CNN Accuracy")
plt.grid(); plt.legend(); plt.show()
```

**그래프 읽는 법**

| 패턴 | 진단 |
|---|---|
| Train/Test loss 모두 감소 | 정상 학습 중 |
| Train loss ↓, **Test loss ↑** | **과적합(Overfitting)** — 증강·Dropout·조기 종료 필요 |
| 둘 다 높은 채로 정체 | **과소적합(Underfitting)** — epoch·모델 용량 부족 |
| Loss가 튀거나 발산 | 학습률이 너무 큼 |

> 노트북에서는 `EPOCHS = 5`로 돌리기 때문에 CIFAR-10이 **충분히 학습되지 않아 분류 실패 사례가 눈에 띈다.** 이건 코드 오류가 아니라 의도된 결과다.

### 10-9. 체크포인트 저장 (MNIST보다 발전된 방식)

```python
save_data = {
    "model_name": "CIFAR10CNN",
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),   # 학습 재개용
    "test_accuracy": test_accuracy_history[-1],
    "epochs": EPOCHS,
    "class_names": CIFAR10_CLASSES,
    "mean": CIFAR10_MEAN,     # 추론 시 같은 전처리를 쓰기 위해
    "std": CIFAR10_STD,
}
torch.save(save_data, "src/models/CIFAR10/CIFAR10_CNN.pth")

# 불러오기
loaded_model = CIFAR10CNN()
state_dict = torch.load("src/models/CIFAR10/CIFAR10_CNN.pth", map_location="cpu", weights_only=True)
loaded_model.load_state_dict(state_dict["model_state_dict"])   # ← 딕셔너리에서 꺼내야 함
loaded_model.to(device)
loaded_model.eval()

print("모델 이름:", state_dict["model_name"])
print(f"마지막 테스트 정확도: {state_dict['test_accuracy'] * 100:.2f}%")
```

> **MNIST 방식과의 차이**: MNIST는 `state_dict`만 저장했지만, CIFAR-10은 **딕셔너리에 메타데이터까지 담았다.** 그래서 불러올 때 `state_dict["model_state_dict"]`로 한 번 더 꺼내야 한다.
>
> `optimizer_state_dict`까지 저장하면 Adam의 모멘텀 상태가 보존되어 **학습을 중단한 지점부터 이어서** 학습할 수 있다. 전처리 통계(`mean`/`std`)를 같이 저장하는 것도 중요하다 — **추론할 때 학습과 다른 전처리를 쓰면 성능이 무너진다.**

---

## 부록 A. TODO 셀 모범답안

> 직접 풀어 본 뒤에 확인할 것.

### A-1. 논리 게이트 (셀 22~25)

```python
def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2                      # AND(-0.7)보다 문턱이 낮음
    return step_function(np.sum(x * w) + b)

def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])    # AND의 부호를 전부 반전
    b = 0.7
    return step_function(np.sum(x * w) + b)

def NOR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])    # OR의 부호를 전부 반전
    b = 0.2
    return step_function(np.sum(x * w) + b)

def XOR(x1, x2):
    return AND(NAND(x1, x2), OR(x1, x2))   # 단층으로는 불가능
```

### A-2. Smoothing / Sharpening 필터 (셀 133)

```python
# Smoothing (평균 필터) — 합이 1이 되도록 정규화해야 밝기가 유지된다
smoothing_filter = np.ones((3, 3), dtype=np.float32)
smoothing_filter = smoothing_filter / smoothing_filter.sum()   # 각 칸 1/9

# Sharpening — 합이 이미 1이라 정규화 불필요
sharpening_filter = np.array([[ 0, -1,  0],
                              [-1,  5, -1],
                              [ 0, -1,  0]], dtype=np.float32)

smoothing_bias  = 0.0    # 밝기를 그대로 유지하고 싶으므로 0
sharpening_bias = 0.0

smoothing_output  = convolution2d(input_image, smoothing_filter,  smoothing_bias,  stride=1, padding=1)
sharpening_output = convolution2d(input_image, sharpening_filter, sharpening_bias, stride=1, padding=1)

# Smoothing 결과는 이미 0 이상이라 ReLU가 불필요
smoothing_feature_map  = smoothing_output
# Sharpening은 음수가 생길 수 있음
sharpening_feature_map = relu(sharpening_output)
```

> **필터 합의 의미**: 합 = 1이면 전체 밝기 유지, 합 = 0이면(edge 필터) 평탄한 영역이 0이 되어 경계만 남는다.

### A-3. 이미지 잔상 효과 (셀 303, 305)

```python
def motion_trail_np(image, weights):
    """NumPy — 출력 픽셀을 하나씩 순차 계산"""
    height, width = image.shape
    trail_length = len(weights)
    output = np.zeros((height, width), dtype=np.float32)

    for output_y in range(height):
        for output_x in range(width):
            value = 0.0
            for shift in range(trail_length):
                input_x = output_x - shift        # 오른쪽으로 shift만큼 이동한 효과
                if input_x < 0:
                    break                         # 왼쪽 경계를 벗어나면 중단
                value += image[output_y, input_x] * weights[shift]
            output[output_y, output_x] = value

    return output


def motion_trail_cp(image, weights):
    """CuPy — 이동 횟수만 순차 반복, 각 단계의 이미지 전체는 GPU 병렬 처리"""
    height, width = image.shape
    trail_length = len(weights)
    output = cp.zeros((height, width), dtype=cp.float32)

    for shift in range(trail_length):
        input_region  = image[:, :width - shift]   # 원본의 왼쪽 부분
        output_region = output[:, shift:]          # 출력의 오른쪽 부분에 대응
        output_region += input_region * weights[shift]   # 배열 전체 연산 = 병렬

    return output
```

> ⚠️ **`range(TRAIL_LENGTH)`가 아니라 `range(len(weights))`를 써야 한다.** GPU warm-up 셀이 `weights_np[:4]`(길이 4)를 넘기는데 `TRAIL_LENGTH`는 32이므로, 상수를 그대로 쓰면 **IndexError**가 난다.

**CPU 버전 `if input_x < 0: break`의 의미**: 이미지 왼쪽 경계를 벗어난 픽셀은 잔상에 기여하지 않는다(zero padding과 같은 효과). GPU 버전에서는 `image[:, :width-shift]`로 슬라이싱해 자연스럽게 같은 효과를 낸다.

### A-4. CuPy 배열 준비 / 결과 회수 (셀 311, 317)

```python
# CPU → GPU
upload_start = time.perf_counter()
input_cp   = cp.asarray(input_np)
weights_cp = cp.asarray(weights_np)
cp.cuda.Stream.null.synchronize()
upload_time = time.perf_counter() - upload_start

# GPU → CPU
download_start = time.perf_counter()
output_cp_np = cp.asnumpy(output_cp)
download_time = time.perf_counter() - download_start
```

### A-5. CuPy Convolution (셀 330)

```python
def convolution2d_cp(image, kernel, bias=0.0, stride=1, padding=0):
    """커널 원소를 순차 반복하고, 각 원소의 출력 배열 전체 연산은 GPU에서 병렬 처리"""
    kernel_height, kernel_width = kernel.shape

    padded_image = cp.pad(image, ((padding, padding), (padding, padding)),
                          mode="constant", constant_values=0)

    output_height = (padded_image.shape[0] - kernel_height) // stride + 1
    output_width  = (padded_image.shape[1] - kernel_width)  // stride + 1
    output = cp.zeros((output_height, output_width), dtype=cp.float32)

    # 커널의 각 원소에 대해 (3x3이면 9번만 반복)
    for kernel_y in range(kernel_height):
        for kernel_x in range(kernel_width):
            # 모든 출력 위치에 대응하는 입력 픽셀을 한 번에 선택
            input_region = padded_image[
                kernel_y : kernel_y + output_height * stride : stride,
                kernel_x : kernel_x + output_width  * stride : stride,
            ]
            # 배열 전체에 커널 가중치를 곱해 누적 (GPU 병렬)
            output += input_region * kernel[kernel_y, kernel_x]

    return output + bias
```

**슬라이싱 이해**: 출력 위치 `(oy, ox)`에 필요한 입력 픽셀은 `padded[oy*stride + ky, ox*stride + kx]`다. `oy`가 `0 ~ output_height-1`을 훑으므로, 시작 `ky`에서 `stride` 간격으로 `output_height`개를 뽑으면 된다 → `ky : ky + output_height*stride : stride`.

```python
# --- CPU 처리 ---
cpu_start = time.perf_counter()
output_dx_np = convolution2d(input_np, dx_edge_filter, dx_bias, stride=1, padding=1)
output_dy_np = convolution2d(input_np, dy_edge_filter, dy_bias, stride=1, padding=1)
cpu_time = time.perf_counter() - cpu_start
relu_dx_np = relu(output_dx_np)
relu_dy_np = relu(output_dy_np)

# --- GPU 처리 (변환 → 워밍업 → 처리) ---
input_gpu = cp.asarray(input_np)
dx_kernel_gpu = cp.asarray(dx_edge_filter)
dy_kernel_gpu = cp.asarray(dy_edge_filter)

# 워밍업
_ = convolution2d_cp(cp.zeros((64, 64), dtype=cp.float32), dx_kernel_gpu, 0.0, 1, 1)
cp.cuda.Stream.null.synchronize()

# 측정
cp.cuda.Stream.null.synchronize()
gpu_start = time.perf_counter()
output_dx_cp = convolution2d_cp(input_gpu, dx_kernel_gpu, dx_bias, stride=1, padding=1)
output_dy_cp = convolution2d_cp(input_gpu, dy_kernel_gpu, dy_bias, stride=1, padding=1)
cp.cuda.Stream.null.synchronize()
gpu_time = time.perf_counter() - gpu_start

relu_dx_cp_np = cp.asnumpy(cp.maximum(0, output_dx_cp))
relu_dy_cp_np = cp.asnumpy(cp.maximum(0, output_dy_cp))

print(f"CPU: {cpu_time:.4f}s / GPU: {gpu_time:.4f}s / Speedup: {cpu_time/gpu_time:.2f}x")
print("최대 오차:", np.abs(relu_dx_np - relu_dx_cp_np).max())   # 결과 일치 확인
```

> `relu()`는 `np.maximum`을 쓰므로 CuPy 배열에는 `cp.maximum(0, x)`를 써야 한다.
> 두 결과의 최대 오차를 찍어 보면 부동소수 오차 수준(`1e-6` 이하)으로 **같은 결과**임을 확인할 수 있다.

### A-6. Loss / Accuracy 그래프 (셀 243, 245)

```python
epoch_axis = range(1, epochs + 1)

plt.figure(figsize=(8, 5))
plt.plot(epoch_axis, train_loss_history, marker="o", label="Train loss")
plt.plot(epoch_axis, test_loss_history,  marker="o", label="Test loss")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("CNN Loss")
plt.grid(); plt.legend(); plt.show()

plt.figure(figsize=(8, 5))
plt.plot(epoch_axis, np.array(train_accuracy_history) * 100, marker="o", label="Train accuracy")
plt.plot(epoch_axis, np.array(test_accuracy_history)  * 100, marker="o", label="Test accuracy")
plt.xlabel("Epoch"); plt.ylabel("Accuracy (%)"); plt.title("CNN Accuracy")
plt.grid(); plt.legend(); plt.show()
```

> 정확도는 `0.0~1.0`으로 저장되어 있으므로 `* 100`을 해야 백분율 축과 맞는다. 리스트에는 `*`가 안 되므로 `np.array()`로 감싼다.

### A-7. 학습 후 Feature Map 시각화 (셀 248)

```python
image, true_label = test_dataset[0]
input_batch = image.unsqueeze(0).to(device)

model.eval()
with torch.no_grad():
    logits, features = model(input_batch, return_features=True)

show_feature_maps(features["conv1"], title="Trained Conv1 Feature Maps", max_maps=8,  cmap="viridis")
show_feature_maps(features["conv2"], title="Trained Conv2 Feature Maps", max_maps=16, cmap="viridis")

# 6개 Layer별 대표 Feature Map 4개씩 비교
layer_names = ["conv1", "relu1", "pool1", "conv2", "relu2", "pool2"]

plt.figure(figsize=(14, 3 * len(layer_names)))
for row, name in enumerate(layer_names):
    feature_maps = features[name][0].detach().cpu()
    for column in range(4):
        plt.subplot(len(layer_names), 4, row * 4 + column + 1)
        plt.imshow(feature_maps[column], cmap="viridis")
        plt.title(f"{name} / ch{column}", fontsize=9)
        plt.axis("off")
plt.tight_layout()
plt.show()
```

### A-8. GPU 학습 함수 `main_gpu()` (셀 389)

```python
def main_gpu():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU를 사용할 수 없습니다.")
    torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda")               # ① CPU 버전과 다른 부분

    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root=DATA_ROOT, train=True,  transform=transform, download=True)
    test_dataset  = datasets.MNIST(root=DATA_ROOT, train=False, transform=transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,      shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)   # ② pin_memory
    test_loader  = DataLoader(test_dataset,  batch_size=TEST_BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # ③ GPU Warm-up — 초기화 시간을 측정에서 제외
    warmup_input = torch.zeros(BATCH_SIZE, 1, 28, 28, device=device)
    model.eval()
    with torch.no_grad():
        for _ in range(3):
            _ = model(warmup_input)
    torch.cuda.synchronize()

    torch.cuda.synchronize()                    # ④ 측정 시작 전 동기화
    training_start_time = time.perf_counter()

    for epoch in range(EPOCHS):
        epoch_start_time = time.perf_counter()

        # train_one_epoch / evaluate는 CPU 버전과 완전히 동일한 함수를 재사용
        train_loss, train_accuracy = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss,  test_accuracy  = evaluate(model, test_loader, criterion, device)

        torch.cuda.synchronize()
        epoch_time = time.perf_counter() - epoch_start_time

        print(f"Epoch {epoch+1}/{EPOCHS} | Train loss: {train_loss:.4f} | "
              f"Train accuracy: {train_accuracy*100:.2f}% | Test loss: {test_loss:.4f} | "
              f"Test accuracy: {test_accuracy*100:.2f}% | Time: {epoch_time:.2f}s")

    torch.cuda.synchronize()                    # ④ 측정 종료 전 동기화
    total_training_time = time.perf_counter() - training_start_time
    print(f"Total GPU training time: {total_training_time:.2f}s")

    return total_training_time
```

### A-9. CIFAR-10 학습 루프 (셀 456)

```python
EPOCHS = 5

train_loss_history, train_accuracy_history = [], []
test_loss_history,  test_accuracy_history  = [], []

torch.cuda.synchronize()
training_start_time = time.perf_counter()

for epoch in range(EPOCHS):
    epoch_start_time = time.perf_counter()

    train_loss, train_accuracy = train_one_epoch(model, train_loader, criterion, optimizer, device)
    test_loss,  test_accuracy  = evaluate(model, test_loader, criterion, device)

    train_loss_history.append(train_loss);     train_accuracy_history.append(train_accuracy)
    test_loss_history.append(test_loss);       test_accuracy_history.append(test_accuracy)

    torch.cuda.synchronize()
    epoch_time = time.perf_counter() - epoch_start_time

    print(f"Epoch {epoch+1}/{EPOCHS} | Train loss: {train_loss:.4f} | "
          f"Train accuracy: {train_accuracy*100:.2f}% | Test loss: {test_loss:.4f} | "
          f"Test accuracy: {test_accuracy*100:.2f}% | Time: {epoch_time:.2f}s")

torch.cuda.synchronize()
total_training_time = time.perf_counter() - training_start_time
print(f"Total GPU training time: {total_training_time:.2f}s")
```

> `train_one_epoch()` 안에서 `model.train()`이, `evaluate()` 안에서 `model.eval()`이 호출되므로 루프에서 따로 모드를 바꿀 필요가 없다. 단, **Warm-up 때 `model.eval()`로 바꿔 놓았기 때문에** 학습 루프에 진입하면서 `train_one_epoch()`이 다시 `model.train()`을 불러 주는 구조에 의존한다는 점은 알고 있을 것.

---

## 부록 B. 자주 틀리는 지점 총정리

### 형태(shape) 관련

| 상황 | 올바른 처리 |
|---|---|
| 이미지 1장을 모델에 입력 | `image.unsqueeze(0)` — Batch 차원 추가 |
| 텐서를 `plt.imshow`에 전달 | `image.squeeze(0)` (흑백) / `image.permute(1,2,0)` (컬러) |
| Flatten 후 Linear 크기 | 입력 크기와 Pooling 횟수로 **직접 계산** (`16*7*7`) |
| Softmax/argmax 축 | `dim=1` (클래스 축). `dim=0`은 배치 축 |
| 브로드캐스팅용 통계값 | `torch.tensor(MEAN).view(3,1,1)` |

### Device 관련

| 증상 | 원인 |
|---|---|
| `Expected all tensors to be on the same device` | `images.to(device)` 또는 `labels.to(device)` 누락 |
| `.to(device)`를 했는데 안 됨 | 텐서는 **반환값을 다시 대입**해야 함: `images = images.to(device)` |
| `TypeError: can't convert cuda:0 tensor to numpy` | `.cpu()`를 먼저 붙일 것 |
| CuPy + NumPy 배열 연산 실패 | `cp.asarray()` / `cp.asnumpy()`로 한쪽으로 맞출 것 |

### 학습 루프 관련

| 실수 | 결과 |
|---|---|
| `optimizer.zero_grad()` 누락 | Gradient 누적 → 발산 |
| `model.eval()` 누락 | Dropout/BatchNorm이 학습 모드로 동작 → 평가 정확도 하락 |
| `torch.no_grad()` 누락 | 메모리 낭비, 속도 저하 |
| CrossEntropyLoss 앞에 softmax 추가 | 이중 적용 → 학습 저하 |
| 다층 신경망 가중치를 0으로 초기화 | 대칭성 문제로 학습 불가 |
| 테스트 데이터에 증강 적용 | 평가 결과가 매번 달라짐 |
| 학습/테스트에 다른 정규화 적용 | 성능 붕괴 |

### GPU 측정 관련

| 실수 | 결과 |
|---|---|
| 동기화 없이 시간 측정 | 비현실적으로 빠른 결과 (지시 시간만 측정됨) |
| Warm-up 없이 측정 | 첫 측정에 초기화 시간이 포함되어 느리게 나옴 |
| 전송 시간을 빼고 speedup 계산 | 실제 이득을 과대평가 |
| 매 연산마다 CPU↔GPU 왕복 | 전송 오버헤드가 연산 이득을 잡아먹음 |

---

## 부록 C. 용어 사전

| 용어 | 정의 |
|---|---|
| **Perceptron** | 인공신경망의 최소 단위. 가중합 + 편향 → 활성화 함수 |
| **Weight / Bias** | 각 입력의 영향력 / 활성화 문턱값. CNN에서는 커널 값 / 특징 임계값 |
| **Activation Function** | 신호를 출력으로 변환. 비선형성을 주입해 층을 쌓는 의미를 만듦 |
| **Epoch** | 전체 학습 데이터를 한 번 다 보는 사이클 |
| **Batch** | 한 번에 모델에 넣는 데이터 묶음 |
| **Loss Function** | 예측과 정답의 차이. BCE(이진), CrossEntropy(다중) |
| **Gradient** | 손실을 줄이는 방향과 크기. 연쇄 법칙으로 계산 |
| **Forward / Backpropagation** | 예측 생성 / 손실 기울기를 역방향으로 전달 |
| **Gradient Descent** | `w ← w - η·dL/dw` |
| **Learning Rate (η)** | 한 번에 가중치를 얼마나 움직일지 |
| **Underfitting / Overfitting** | 학습 부족 / 학습 데이터에만 과도하게 맞춰짐 |
| **Kernel / Filter** | Convolution에 사용하는 작은 격자. CNN의 가중치 |
| **Feature Map** | Convolution 결과. 커널 1개당 1개 생성 |
| **Padding / Stride** | 가장자리 추가 픽셀 / 커널 이동 간격 |
| **Pooling** | 크기 축소. MaxPool은 영역 최댓값을 취함 |
| **Logit** | Softmax를 거치지 않은 모델의 raw 출력 |
| **Softmax** | Logit을 클래스별 확률로 변환 |
| **BatchNorm** | 배치별 정규화로 학습 안정화 |
| **Global Average Pooling** | Feature Map 전체를 평균 1개로 압축. FC 파라미터 대폭 감소 |
| **Dropout** | 학습 중 일부 뉴런을 무작위 비활성화해 과적합 완화 |
| **Data Augmentation** | 학습 이미지를 무작위 변형해 다양성 확보 |
| **state_dict** | 모델의 가중치만 담은 딕셔너리. 구조는 코드로 따로 정의 |
| **Host / Device Memory** | CPU 메모리(RAM) / GPU 메모리(VRAM) |
| **Synchronize** | GPU 작업 완료까지 CPU가 대기. 시간 측정에 필수 |
| **Warm-up** | 측정 전 더미 연산으로 GPU 초기화를 끝내 두는 것 |
| **UMA** | 통합 메모리 아키텍처. Jetson처럼 CPU/GPU가 메모리 칩을 공유하는 구조 |

---

*원본: `03_DL-and-GPU.ipynb` — Physical AI의 Vision-LLM 융합 시청각 멀티모달 시스템 (김규래)*

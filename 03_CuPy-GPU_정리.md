# 03_DL-and-GPU — CuPy GPU 가속 정리 (실습 정답 포함)

`03_DL-and-GPU.ipynb`의 **§G. GPU 병렬 연산 (CuPy)** 파트만 따로 떼어 정리한 노트.
노트북의 `TODO` 셀은 **전부 정답을 채워** 셀 번호 순서대로 실었으므로, 노트북과 나란히 놓고 대조하며 볼 수 있습니다.

> **이 문서의 핵심 한 줄**
> GPU 가속은 "반복문을 없애는 것"이 아니라, **가장 안쪽의 픽셀 단위 반복을 배열 전체 연산으로 바꿔치기하는 것**이다.
> 그래서 `convolution2d`의 `921,600번 반복`이 `convolution2d_cp`의 `9번 반복`으로 바뀐다.

---

## 목차

| # | 내용 | 노트북 셀 |
|---|---|---|
| 1 | [CuPy란 무엇인가](#1-cupy란-무엇인가) | 256–264 |
| 2 | [호스트 메모리 vs 디바이스 메모리](#2-호스트-메모리-vs-디바이스-메모리) | 265–279 |
| 3 | [NumPy와 CuPy 문법 비교](#3-numpy와-cupy-문법-비교) | 280–295 |
| 4 | [동기화와 시간 측정](#4-동기화와-시간-측정--가장-많이-틀리는-부분) | 290–292 |
| 5 | [실습 1 — 이미지 잔상 효과 (정답)](#5-실습-1--이미지-잔상-효과-정답-포함) | 296–324 |
| 6 | [실습 2 — Convolution을 CuPy로 재작성 (정답)](#6-실습-2--convolution을-cupy로-재작성-정답-포함) | 325–338 |
| 7 | [두 실습에서 얻는 일반 원칙](#7-두-실습에서-얻는-일반-원칙) | — |
| — | [부록 A. 전체 실행 코드](#부록-a-복사해서-바로-실행-가능한-전체-코드) | |
| — | [부록 B. NumPy ↔ CuPy 치환표](#부록-b-numpy--cupy-치환표) | |
| — | [부록 C. CuPy 관련 에러 대처](#부록-c-cupy-관련-에러-대처) | |

---

## 1. CuPy란 무엇인가

### 핵심

`CuPy`는 **NumPy와 호환되는 GPU 배열 라이브러리**다. 문법은 거의 그대로 두고, **연산이 실행되는 물리적 장치**만 CPU에서 GPU로 옮긴다.

- NumPy와 유사한 문법으로 GPU 배열 연산
- NVIDIA CUDA를 활용해 대규모 연산을 GPU에서 가속

| | NumPy | CuPy |
|---|---|---|
| 실행 장치 | CPU 코어 | GPU 코어 |
| Jetson Orin Nano 기준 | **6개** 고성능 코어 | **1024개** CUDA 코어 |
| 데이터 위치 | Host memory (RAM) | Device memory (VRAM) |
| 타입 | `numpy.ndarray` | `cupy.ndarray` |
| 잘하는 일 | 복잡하고 정교한 제어 흐름, 순차 처리 | **단순한 연산의 대규모 병렬 처리** |

> 코어 수가 170배 차이 나지만, **CPU 코어 하나가 GPU 코어 하나보다 훨씬 똑똑하다.**
> GPU는 "1024명이 동시에 같은 단순 작업을 하는" 구조라서, 조건 분기가 복잡한 작업에는 오히려 불리하다.
> 그래서 **"모든 픽셀에 같은 연산"** 같은 작업에서만 가속 효과가 난다.

### 설치 (JetPack 6.2 / CUDA 12.6 / NumPy 1.21.5 기준)

```bash
python -m pip download --no-deps --only-binary=:all: "cupy-cuda12x==12.3.0" -d ~/Downloads
python -m pip install --no-deps "fastrlock==0.8.3"
python -m pip install --no-deps --only-binary=:all: ~/Downloads/cupy_cuda12x-12.3.0-*.whl
```

> - 패키지 이름의 `cuda12x`가 **CUDA 12.x용 빌드**라는 뜻이다. CUDA 버전이 다르면 다른 패키지를 받아야 한다.
> - `fastrlock`은 CuPy의 필수 의존성인데 `--no-deps` 때문에 자동 설치되지 않으므로 따로 깔아 준다.
> - `--no-deps`를 쓰는 이유: 의존성 해결 과정에서 pip가 NumPy를 멋대로 다른 버전으로 덮어쓰면 JetPack 환경이 깨지기 때문이다.

### 설치 확인 (셀 261–264)

```python
import numpy as np
import cupy as cp

print("NumPy :", np.__version__)
print("CuPy  :", cp.__version__)

print("GPU count:", cp.cuda.runtime.getDeviceCount())
print("CUDA runtime:", cp.cuda.runtime.runtimeGetVersion())

x_cpu = np.arange(10)  # NumPy (host memory)
x_gpu = cp.arange(10)  # CuPy  (CUDA device memory)
y_cpu = x_cpu ** 2     # CPU에서 NumPy로 계산
y_gpu = x_gpu ** 2     # GPU에서 CuPy로 계산

cp.cuda.Stream.null.synchronize()

print("NumPy result:", y_cpu)   # [ 0  1  4  9 16 25 36 49 64 81]
print("CuPy result:",  y_gpu)   # [ 0  1  4  9 16 25 36 49 64 81]
print("NumPy type:", type(y_cpu))   # <class 'numpy.ndarray'>
print("CuPy type:",  type(y_gpu))   # <class 'cupy.ndarray'>
```

**정상 출력 예시**

```text
NumPy : 1.21.5
CuPy  : 12.3.0

GPU count: 1
CUDA runtime: 12060
```

> `runtimeGetVersion()`이 반환하는 `12060`은 **12.6.0**을 뜻한다 (`major*1000 + minor*10`).

---

## 2. 호스트 메모리 vs 디바이스 메모리

### 핵심

| 배열 | 저장 위치 | 부르는 이름 |
|---|---|---|
| `np.array([1,2,3,4])` | 시스템 메모리 (RAM) | **Host memory** |
| `cp.array([1,2,3,4])` | 그래픽 메모리 (VRAM) | **Device memory** |

### ⚠️ Jetson의 통합 메모리(UMA)에서도 복사는 필요하다

일반 PC는 RAM과 VRAM이 물리적으로 분리되어 있다. 반면 **Jetson Orin Nano는 CPU와 GPU가 같은 메모리 칩을 공유**하는 **통합 메모리 아키텍처(Unified Memory Architecture, UMA)** 를 쓴다.

그럼에도:

> 물리적으로 나뉘어 있지 않더라도, **논리적으로는 서로 다른 영역에 분리되어** CPU와 GPU가 각자 할당된 메모리를 관리한다.
> 따라서 NumPy 배열과 CuPy 배열을 함께 연산하려면 **어느 한쪽 메모리 공간으로 먼저 복사해야 한다.**

"같은 칩이니까 그냥 되겠지"라고 생각하면 아래 에러를 만난다.

### 복사 (셀 274–279)

```python
# CPU → GPU  (NumPy → CuPy)
x_from_cpu_to_gpu = cp.asarray(x_cpu)
print(type(x_from_cpu_to_gpu))   # <class 'cupy.ndarray'>

# GPU → CPU  (CuPy → NumPy)
x_from_gpu_to_cpu = cp.asnumpy(x_gpu)
print(type(x_from_gpu_to_cpu))   # <class 'numpy.ndarray'>

# 서로 다른 장치의 배열은 직접 연산 불가
try:
    y = x_cpu + x_gpu
except RuntimeError as e:
    print(f"잘못된 연산: {e}")
```

| 방향 | 함수 | 기억법 |
|---|---|---|
| CPU → GPU | `cp.asarray(x)` | **cp**로 만드니까 GPU로 **올린다** |
| GPU → CPU | `cp.asnumpy(x)` | **numpy**로 만드니까 CPU로 **내린다** |

> **`cp.asarray()` vs `cp.array()`**
> `asarray`는 이미 CuPy 배열이면 **복사하지 않고 그대로 반환**하고, `array`는 **항상 새로 복사**한다.
> 전송 함수로는 `asarray`가 낫다 — 이미 GPU에 있는 데이터를 불필요하게 다시 복사하지 않는다.

### OpenCV / Matplotlib와의 연결 (셀 293–294)

```python
ret, frame = cap.read()              # NumPy, CPU

frame_gpu  = cp.asarray(frame)       # CPU → GPU
result_gpu = frame_gpu ** 2          # GPU 연산
result_cpu = cp.asnumpy(result_gpu)  # GPU → CPU

cv2.imshow("Result", result_cpu)     # OpenCV는 NumPy만 받는다
plt.imshow(result_cpu)               # Matplotlib도 NumPy만 받는다
```

> **`cv2`와 `plt`는 CuPy 배열을 읽지 못한다.** 화면에 뿌리기 직전에 반드시 `cp.asnumpy()`로 내려야 한다.
> 반대로 말하면, **GPU에 올린 뒤 화면 출력 직전까지는 계속 GPU에 두는 것**이 이상적이다.

---

## 3. NumPy와 CuPy 문법 비교

### 축약 함수 (셀 282–284)

```python
# --- NumPy ---
x_cpu = np.array([1, 2, 3, 4])

total   = np.sum(x_cpu)
maximum = np.max(x_cpu)
minimum = np.min(x_cpu)
mean    = np.mean(x_cpu)
index   = np.argmax(x_cpu)
# → 전부 NumPy 스칼라 타입

# --- CuPy (권장) ---
x_gpu = cp.array([1, 2, 3, 4])

total   = cp.sum(x_gpu).item()
maximum = cp.max(x_gpu).item()
minimum = cp.min(x_gpu).item()
mean    = cp.mean(x_gpu).item()
index   = cp.argmax(x_gpu).item()
# → 전부 Python 기본 타입 (int, float)
```

> **주의 (노트북 셀 284가 보여주는 함정)**
> CuPy 배열에 `np.sum()`을 써도 **에러 없이 동작한다.** CuPy가 NumPy의 디스패치 규약을 구현해 두어, 내부적으로 CuPy 구현으로 넘어가기 때문이다.
> 문제는 **반환 타입이 `cupy.ndarray`(0차원)** 라는 점이다. 그걸 그대로 Python 숫자처럼 쓰거나 f-string에 넣으면 나중에 혼란스러워진다.
>
> → **CuPy 배열에는 `cp.` 함수를 쓰고, Python 숫자가 필요하면 `.item()`을 붙인다.**

### 배열 생성과 행렬곱 (셀 286–288)

```python
# --- NumPy ---
a_cpu = np.random.rand(1000, 1000).astype(np.float32)
b_cpu = np.random.rand(1000, 1000).astype(np.float32)
c_cpu = a_cpu @ b_cpu

# --- CuPy ---
a_gpu = cp.random.rand(1000, 1000, dtype=cp.float32)   # dtype을 인자로 직접 지정 가능
b_gpu = cp.random.rand(1000, 1000, dtype=cp.float32)
c_gpu = a_gpu @ b_gpu
```

> `cp.random.rand()`는 `dtype` 인자를 직접 받지만 `np.random.rand()`는 받지 않아 `.astype()`이 필요하다. **이런 미세한 차이가 곳곳에 있으므로 "완전 호환"이라고 믿으면 안 된다.**

### 문법은 같지만 실행되는 곳이 다르다

> Jetson Orin Nano에서
> - `NumPy` 배열 → **6개의 고성능 CPU 코어**가 복잡하고 정교한 제어 흐름과 함께 **순차적으로** 처리
> - `CuPy` 함수 → **1024개의 GPU 코어**가 일제히 **대규모 병렬 연산** 수행

---

## 4. 동기화와 시간 측정 — 가장 많이 틀리는 부분

### 핵심

**GPU 연산은 비동기(asynchronous)로 실행된다.** CPU는 GPU에게 "이거 해라"라고 **지시만 던져 놓고 곧바로 다음 줄로 넘어간다.**

```python
y_gpu = x_gpu ** 2     # ← 이 줄이 끝났다고 계산이 끝난 게 아니다!
end = time.perf_counter()   # ← 여기서 재면 "지시하는 데 걸린 시간"만 측정됨
```

그래서 동기화 없이 시간을 재면 **비현실적으로 빠른(사실상 0에 가까운) 결과**가 나온다.

### 올바른 측정 (셀 290–292)

```python
# --- NumPy: 동기화 불필요 (CPU는 원래 동기적) ---
start = time.perf_counter()
y_cpu = x_cpu ** 2
end = time.perf_counter()
print(end - start)

# --- CuPy: 앞뒤로 동기화 필수 ---
cp.cuda.Stream.null.synchronize()   # ① 이전 GPU 작업이 남아 있을 수 있으니 먼저 비움
start = time.perf_counter()

y_gpu = x_gpu ** 2

cp.cuda.Stream.null.synchronize()   # ② 계산이 실제로 끝날 때까지 대기
end = time.perf_counter()
print(end - start)
```

| 라이브러리 | 동기화 함수 |
|---|---|
| CuPy | `cp.cuda.Stream.null.synchronize()` |
| PyTorch | `torch.cuda.synchronize()` |

> **두 번의 동기화가 각각 하는 일**
> ① **시작 전** — 앞선 작업이 아직 GPU 큐에 남아 있으면 그 시간까지 측정에 섞인다. 큐를 비우고 출발한다.
> ② **종료 전** — 지시만 내려진 상태에서 시계를 멈추면 안 되므로, 실제 완료를 기다린다.

### GPU Warm-up (셀 312–313)

```python
warmup_image   = cp.zeros((64, 64), dtype=cp.float32)
warmup_weights = cp.asarray(weights_np[:4])

_ = motion_trail_cp(image=warmup_image, weights=warmup_weights)

cp.cuda.Stream.null.synchronize()
```

> GPU는 **첫 연산에서** CUDA Context 생성, 커널 컴파일/로딩, 메모리 할당 같은 준비 작업을 한다.
> 이 초기화 시간이 측정에 섞이면 **첫 결과만 유독 느리게** 나온다. 그래서 측정 전에 같은 형태의 연산을 미리 한 번 돌려 둔다.
>
> ⚠️ **warm-up이 `weights_np[:4]`(길이 4)를 넘긴다는 점을 반드시 볼 것.** `TRAIL_LENGTH`(=32)를 함수 안에 상수로 박아 두면 여기서 **IndexError**가 난다. → 함수는 `len(weights)`를 써야 한다.

---

## 5. 실습 1 — 이미지 잔상 효과 (정답 포함)

### 원리

원본을 오른쪽으로 조금씩 이동한 복사본을 만들고, 이동 거리가 멀수록 작은 가중치를 곱해 전부 더한다.

```text
원본 이미지                × weight[0]
오른쪽으로 1픽셀 이동한 것 × weight[1]
오른쪽으로 2픽셀 이동한 것 × weight[2]
...
```

작은 예시:

```text
입력 배열:  [10, 20, 30, 40, 50]
가중치:     [0.6, 0.3, 0.1]

이동 없음:  [10, 20, 30, 40, 50] × 0.6
오른쪽 1칸: [ 0, 10, 20, 30, 40] × 0.3
오른쪽 2칸: [ 0,  0, 10, 20, 30] × 0.1
                  ↓ 같은 위치끼리 합산
```

가중치가 거리에 따라 작아지므로 **가까운 잔상은 선명하고 먼 잔상은 흐리게** 나타난다.

### CPU와 GPU의 처리 방식 차이 — 이 실습의 전부

| | 반복 구조 | 반복 횟수 (1280×720, TRAIL=32) | 병렬화되는 부분 |
|---|---|---|---|
| **CPU (NumPy)** | 출력 픽셀을 **하나씩** 순차 계산 | `720 × 1280 × 32 ≈ 2,949만` | 없음 |
| **GPU (CuPy)** | **이동 횟수만** 순차 반복 | **`32`** | 각 단계의 **92만 픽셀 전체**를 병렬 처리 |

---

### 📓 셀 299 — 글로벌 변수 (원본 그대로)

```python
IMAGE_PATH = "src/images/seagull.jpg"

TARGET_WIDTH = 1280
TARGET_HEIGHT = 720

TRAIL_LENGTH = 32
```

### 📓 셀 301 — 이미지 불러오기 (원본 그대로)

```python
img_gray = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)

if img_gray is None:
    raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {IMAGE_PATH}")

img_gray = cv2.resize(
    img_gray,
    (TARGET_WIDTH, TARGET_HEIGHT),
    interpolation=cv2.INTER_AREA
)

input_np = img_gray.astype(np.float32) / 255.0

plt.imshow(input_np, cmap="gray")
plt.axis("off");
```

> - `cv2.IMREAD_GRAYSCALE`로 읽으면 처음부터 2차원 배열이라 `cvtColor`가 필요 없다.
> - `cv2.resize`의 크기 인자는 **`(width, height)` 순서**다. NumPy shape의 `(height, width)`와 **반대**이므로 늘 헷갈리는 지점.
> - `INTER_AREA`는 **축소**에 가장 적합한 보간법이다.
> - `/ 255.0`으로 0~1 float32로 바꿔야 가중치 합산 시 값이 폭주하지 않는다.

### ✅ 셀 303 — 잔상 효과 함수 (CPU) · **정답**

```python
def motion_trail_np(image, weights):
    """
    NumPy를 이용해 이미지 잔상 효과를 생성한다.

    출력 픽셀을 Python 반복문으로 하나씩 계산한다.
    """

    # 필요한 변수 정의 및 출력 배열 생성
    height, width = image.shape
    trail_length = len(weights)
    output = np.zeros((height, width), dtype=np.float32)

    for output_y in range(height):
        for output_x in range(width):

            accumulated = 0.0

            for shift in range(trail_length):
                # x값 shift (오른쪽으로 shift만큼 밀린 잔상 = 왼쪽 픽셀을 가져옴)
                input_x = output_x - shift

                # 왼쪽 경계를 벗어나면 더 이상 기여할 픽셀이 없다
                if input_x < 0:
                    break

                # shift한 픽셀을 누적
                accumulated += image[output_y, input_x] * weights[shift]

            # 출력 배열에 누적 픽셀 저장
            output[output_y, output_x] = accumulated

    return output
```

**핵심 포인트**

| 코드 | 이유 |
|---|---|
| `input_x = output_x - shift` | 출력의 `x`에 잔상을 만들려면 **왼쪽(`-shift`)의 원본 픽셀**을 끌어온다. `+shift`가 아니다 |
| `if input_x < 0: break` | 왼쪽 경계 밖은 0으로 취급 (zero padding과 같은 효과). `shift`가 커질수록 더 벗어나므로 `continue`가 아니라 **`break`** 로 끊어도 된다 |
| `trail_length = len(weights)` | `TRAIL_LENGTH` 상수를 박지 않는다 (warm-up 호환) |
| `dtype=np.float32` | 입력이 float32이므로 맞춰 준다 |

### ✅ 셀 305 — 잔상 효과 함수 (GPU) · **정답**

```python
def motion_trail_cp(image, weights):
    """
    CuPy를 이용해 이미지 잔상 효과를 생성한다.

    이동 거리는 Python에서 순차적으로 반복하지만,
    각 이동 거리의 이미지 전체 계산은 GPU에서 병렬 처리한다.
    """

    # 필요한 변수 정의 및 출력 배열 생성
    height, width = image.shape
    trail_length = len(weights)
    output = cp.zeros((height, width), dtype=cp.float32)

    for shift in range(trail_length):
        # 배열 slicing
        input_region  = image[:, :width - shift]   # 원본의 왼쪽 (width - shift)열
        output_region = output[:, shift:]          # 출력의 오른쪽 (width - shift)열

        # GPU가 병렬 연산할 output_region 계산 코드
        output_region += input_region * weights[shift]

    return output
```

**슬라이싱이 왜 이렇게 되는가**

CPU 버전의 `output[y, x] += image[y, x - shift]`를 **x 전체에 대해 한 번에** 쓰면:

```text
output[:, shift:]  ←  image[:, :width - shift]
   출력 x = shift .. width-1        입력 x = 0 .. width-1-shift
   (양쪽 모두 길이 width - shift로 정확히 대응)
```

`shift = 0`일 때는 `image[:, :width]`(전체) → `output[:, 0:]`(전체)가 되어 원본 그대로 더해진다.

| 항목 | 설명 |
|---|---|
| `output_region`은 **뷰(view)** 다 | `output[:, shift:]`는 복사본이 아니라 원본을 가리키는 창이다. 그래서 `+=`가 `output`을 직접 수정한다 |
| `+=`를 `=`로 바꾸면 안 된다 | 누적이 아니라 덮어쓰기가 되어 마지막 shift만 남는다 |
| 반복은 32번뿐 | 각 반복이 92만 픽셀을 통째로 처리한다 |

> **CPU 버전의 `if input_x < 0: break`에 해당하는 처리가 GPU 버전에는 안 보이는 이유**
> 슬라이싱 `image[:, :width - shift]` 자체가 "경계를 벗어나는 부분은 아예 대상에서 제외"하는 역할을 한다.
> 출력의 왼쪽 `shift`개 열은 `output[:, shift:]`에 포함되지 않으므로 그 shift 단계에서는 건드려지지 않는다 — 결과적으로 CPU 버전과 **완전히 동일**하다.

### 📓 셀 307 — 가중치 설정 (원본 그대로)

```python
# 이동 거리가 멀어질수록 가중치를 작게 설정
weights_np = np.linspace(1.0, 0.05, TRAIL_LENGTH, dtype=np.float32)

# 전체 가중치의 합을 1로 정규화
weights_np /= np.sum(weights_np)
```

> **합을 1로 정규화하는 이유**: 32장을 그냥 더하면 픽셀값이 32배가 되어 이미지가 새하얗게 날아간다. 합이 1이면 **전체 밝기가 원본과 같은 수준으로 유지**된다. (§5의 Smoothing 필터를 정규화하는 이유와 완전히 같다.)

### 📓 셀 309 — NumPy CPU 처리 (원본 그대로)

```python
cpu_start = time.perf_counter()

output_np = motion_trail_np(
    image=input_np,
    weights=weights_np
)

cpu_end = time.perf_counter()

cpu_time = cpu_end - cpu_start
```

> ⚠️ 이 셀은 **오래 걸린다.** 2,949만 번의 Python 반복문이 돌기 때문이다. 실행 후 기다릴 것.

### ✅ 셀 311 — CuPy 배열 준비 · **정답**

```python
# NumPy 배열을 CuPy 배열로 변환

upload_start = time.perf_counter()

input_cp   = cp.asarray(input_np)
weights_cp = cp.asarray(weights_np)

cp.cuda.Stream.null.synchronize()

upload_end = time.perf_counter()

upload_time = upload_end - upload_start
```

> 전송도 비동기일 수 있으므로 **`synchronize()`가 이미 스켈레톤에 들어 있다.** 전송 시간을 정확히 재기 위한 장치다.

### 📓 셀 313 — GPU 워밍업 (원본 그대로)

```python
warmup_image   = cp.zeros((64, 64), dtype=cp.float32)
warmup_weights = cp.asarray(weights_np[:4])

_ = motion_trail_cp(
    image=warmup_image,
    weights=warmup_weights
)

cp.cuda.Stream.null.synchronize()
```

> 다시 강조: **`weights_np[:4]`** 다. `motion_trail_cp`가 `TRAIL_LENGTH`를 하드코딩했다면 여기서 터진다.

### 📓 셀 315 — CuPy GPU 처리 (원본 그대로)

```python
cp.cuda.Stream.null.synchronize()

gpu_start = time.perf_counter()

output_cp = motion_trail_cp(
    image=input_cp,
    weights=weights_cp
)

cp.cuda.Stream.null.synchronize()

gpu_end = time.perf_counter()

gpu_time = gpu_end - gpu_start
```

### ✅ 셀 317 — GPU 결과를 CPU로 가져오기 · **정답**

```python
# output_cp를 CPU 메모리로 복사

download_start = time.perf_counter()

output_cp_np = cp.asnumpy(output_cp)

download_end = time.perf_counter()

download_time = download_end - download_start
```

> `cp.asnumpy()`는 **내부적으로 동기화를 포함**하므로 별도의 `synchronize()` 호출이 스켈레톤에 없다. 결과를 실제로 읽어야 하니 기다릴 수밖에 없기 때문이다.

### 📓 셀 319–321 — 결과 비교 (원본 그대로)

```python
compute_speedup = cpu_time / gpu_time

gpu_total_time = upload_time + gpu_time + download_time
total_speedup  = cpu_time / gpu_total_time
```

```python
print("========== NumPy CPU ==========")
print(f"Processing time: {cpu_time:.4f} seconds")
print()

print("========== CuPy GPU ==========")
print(f"CPU to GPU: {upload_time:.4f} seconds")
print(f"GPU processing: {gpu_time:.4f} seconds")
print(f"GPU to CPU: {download_time:.4f} seconds")
print(f"Total GPU time: {gpu_total_time:.4f} seconds")
print()

print("========== Comparison ==========")
print(f"Compute-only speedup: {compute_speedup:.2f}x")
print(f"Including transfers: {total_speedup:.2f}x")
```

### 결과 해석 가이드 — 이 실습이 진짜 말하려는 것

| 지표 | 의미 | 볼 것 |
|---|---|---|
| `compute_speedup` | **연산만** 비교 | 크게 나온다. GPU의 순수 연산력 |
| `total_speedup` | **전송 포함** 비교 | 확연히 줄어든다 |
| `upload_time` + `download_time` | 전송 오버헤드 | `gpu_time`과 비교해 볼 것 |

> **핵심 교훈**: 데이터가 작거나 연산이 가벼우면 **전송 시간이 연산 시간보다 커서 GPU가 오히려 손해**일 수 있다.
>
> **실무 원칙** — 데이터를 GPU에 **한 번 올리고, 거기서 최대한 많은 연산을 끝낸 뒤, 결과만 한 번 내려받는다.**
> 매 연산마다 `asarray` ↔ `asnumpy`로 왕복시키면 GPU를 쓰는 의미가 사라진다.

### 📓 셀 323 — 이미지 출력 (원본 그대로)

```python
plt.figure(figsize=(18,12))

plt.subplot(1,3,1), plt.imshow(input_np,     cmap="gray"), plt.title("Original Image")
plt.subplot(1,3,2), plt.imshow(output_np,    cmap="gray"), plt.title("Motion Trail (CPU)")
plt.subplot(1,3,3), plt.imshow(output_cp_np, cmap="gray"), plt.title("Motion Trail (GPU)")

for ax in plt.gcf().axes:
    ax.axis("off")
```

> CPU 결과와 GPU 결과가 **눈으로 구분되지 않아야 정상**이다. 수치로 확인하려면:
> ```python
> print("최대 오차:", np.abs(output_np - output_cp_np).max())   # 1e-6 이하면 동일
> ```

---

## 6. 실습 2 — Convolution을 CuPy로 재작성 (정답 포함)

> **이 절이 이 문서의 핵심이다.**

### 6-1. 기존 NumPy 버전 (§E에서 정의, 셀 117)

```python
def convolution2d(image, kernel, bias=0.0, stride=1, padding=0):
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    padded_image = np.pad(image, ((padding, padding), (padding, padding)),
                          mode="constant", constant_values=0)

    output_height = (padded_image.shape[0] - kernel_height) // stride + 1
    output_width  = (padded_image.shape[1] - kernel_width)  // stride + 1
    output = np.zeros((output_height, output_width), dtype=np.float32)

    # 출력 픽셀을 하나씩 순회 (H × W 번)
    for output_y in range(output_height):
        for output_x in range(output_width):
            start_y = output_y * stride
            start_x = output_x * stride

            image_region = padded_image[start_y:start_y + kernel_height,
                                        start_x:start_x + kernel_width]

            output[output_y, output_x] = np.sum(image_region * kernel) + bias

    return output
```

```text
출력 위치 하나 선택
→ 커널 크기의 이미지 영역 추출
→ 이미지 영역과 커널의 가중합 계산
→ 출력 픽셀 하나 저장
```

> convolution의 원리를 직접 확인하기에는 좋지만, **이미지의 높이 × 너비만큼 Python 반복문**을 돌아야 해서 큰 이미지에서는 매우 느리다. 1280×720이면 **921,600번**이다.

### 6-2. 발상의 전환 — 반복 순서를 뒤집는다

| | 무엇을 순회하나 | 반복 횟수 (1280×720, 3×3 커널) | 한 번의 반복에서 하는 일 |
|---|---|---|---|
| `convolution2d` | **출력 픽셀** | **921,600** | 커널 영역을 잘라 가중합 → **픽셀 1개** 생성 |
| `convolution2d_cp` | **커널 원소** | **9** | 모든 출력 위치의 대응 입력 픽셀에 가중치를 곱해 **출력 배열 전체**에 누적 |

```text
[NumPy]  출력 위치 하나 선택 → 커널 크기 영역 추출 → 가중합 → 출력 픽셀 하나 저장
[CuPy]   커널 원소 하나 선택 → 모든 출력 위치의 대응 입력 픽셀 선택
                             → 배열 전체에 현재 커널 가중치 적용
                             → 출력 배열 전체에 누적
```

> **계산 순서는 완전히 다르지만, 최종적으로 각 출력 픽셀에는 이미지 영역과 커널의 동일한 가중합이 저장된다.**
> 덧셈의 순서를 바꾼 것뿐이기 때문이다 — `(a+b)+c = a+(b+c)`.

**그림으로 보면**

```text
[NumPy] 픽셀 (0,0)에 대해 커널 9칸을 다 더함 → 픽셀 (0,1)에 대해 9칸 → ... (92만 번)

           k00 k01 k02
           k10 k11 k12   ← 한 픽셀마다 이 9칸을 전부 훑는다
           k20 k21 k22

[CuPy]  k00에 대해 92만 픽셀 전부 누적 → k01에 대해 92만 픽셀 전부 누적 → ... (9번)

           k00 → output += (대응 입력 전체) * k00     ← 이 한 줄이 GPU에서 병렬 실행
           k01 → output += (대응 입력 전체) * k01
            ...
```

### ✅ 셀 330 — Convolution 함수 정의 (CuPy) · **정답**

```python
def convolution2d_cp(image, kernel, bias=0.0, stride=1, padding=0):
    """
    CuPy를 이용한 2차원 convolution.

    커널의 위치는 Python 반복문으로 순차 처리하고,
    각 위치에 해당하는 출력 배열 전체 연산은 GPU에서 처리한다.

    Parameters
    ----------
    image : cp.ndarray
        입력 이미지, shape = (height, width)
    kernel : cp.ndarray
        CNN 필터, shape = (kernel_height, kernel_width)
    bias : float
        필터 출력에 더할 편향
    stride : int
        필터가 이동하는 간격
    padding : int
        입력 이미지 가장자리에 추가할 픽셀의 개수

    Returns
    -------
    output : cp.ndarray
        편향까지 적용된 convolution 결과
    """

    kernel_height, kernel_width = kernel.shape

    # 이미지 가장자리에 zero padding 적용 (cp.pad — np.pad와 인자가 동일)
    padded_image = cp.pad(
        image,
        ((padding, padding), (padding, padding)),
        mode="constant",
        constant_values=0
    )

    # 출력 크기는 NumPy 버전과 동일한 공식
    output_height = (padded_image.shape[0] - kernel_height) // stride + 1
    output_width  = (padded_image.shape[1] - kernel_width)  // stride + 1

    # 계산한 크기로 출력 배열을 GPU에 준비
    output = cp.zeros((output_height, output_width), dtype=cp.float32)

    # 커널 원소를 하나씩 순회 (3x3이면 단 9번)
    for kernel_y in range(kernel_height):
        for kernel_x in range(kernel_width):

            # 모든 출력 위치에서 현재 커널 원소와 대응되는 입력 픽셀을 한 번에 선택
            input_region = padded_image[
                kernel_y : kernel_y + output_height * stride : stride,
                kernel_x : kernel_x + output_width  * stride : stride,
            ]

            # 배열 전체에 현재 커널 가중치를 적용하여 출력 배열 전체에 누적 (GPU 병렬)
            output += input_region * kernel[kernel_y, kernel_x]

    # 편향은 마지막에 한 번만 더한다
    return output + bias
```

### 6-3. 슬라이싱 인덱스 유도 — 이 부분만 이해하면 끝

NumPy 버전에서 출력 픽셀 `(oy, ox)`가 참조하는 입력 픽셀은:

```python
padded_image[oy * stride + ky, ox * stride + kx]
```

여기서 **`(ky, kx)`를 고정**하고 `oy`를 `0 ~ output_height-1`로 훑으면, 행 인덱스는

```text
ky + 0*stride,  ky + 1*stride,  ky + 2*stride,  ...,  ky + (output_height-1)*stride
```

즉 **시작 `ky`, 간격 `stride`, 개수 `output_height`** 인 등차수열이다. 파이썬 슬라이스로 쓰면:

```python
ky : ky + output_height * stride : stride
```

열도 완전히 같은 논리로 `kx : kx + output_width * stride : stride`.

| 슬라이스 요소 | 값 | 의미 |
|---|---|---|
| start | `kernel_y` | 커널 원소의 상대 위치만큼 밀어서 시작 |
| stop | `kernel_y + output_height * stride` | 정확히 `output_height`개만 뽑기 위한 끝 |
| step | `stride` | 커널이 이동하는 간격 |

> **왜 `stop`을 이렇게 잡는가?** `padded_image.shape[0]`을 그대로 쓰면 `stride > 1`일 때 개수가 하나 더 나올 수 있어 `output`과 shape이 어긋난다. **개수를 명시적으로 계산해 넣는 것이 안전하다.**

### 6-4. `bias`를 마지막에 한 번만 더하는 이유

NumPy 버전은 픽셀마다 `np.sum(...) + bias`로 **한 번** 더한다. CuPy 버전에서 루프 안에 `+ bias`를 넣으면 커널 원소 수만큼, 즉 3×3 커널이면 **9번 더해진다.** 결과적으로 출력 전체가 `bias × 8`만큼 어긋난다 (`dx_bias = -0.2`이면 `-1.6`).

```python
# ❌ 틀림 — bias가 kernel_height × kernel_width 번 더해짐
output += input_region * kernel[kernel_y, kernel_x] + bias

# ✅ 맞음 — 루프가 끝난 뒤 한 번만
return output + bias
```

> 이것이 이 함수에서 **가장 흔한 실수**다. 결과 이미지가 전체적으로 이상하게 밝거나 어두우면 여기를 의심할 것.

### ✅ 셀 332 — NumPy CPU 처리 · **정답**

```python
# 기존에 정의한 convolution2d 함수를 사용하여 convolution을 CPU로 처리

cpu_start = time.perf_counter()

output_dx_np = convolution2d(
    image=input_np,
    kernel=dx_edge_filter,
    bias=dx_bias,
    stride=1,
    padding=1
)

output_dy_np = convolution2d(
    image=input_np,
    kernel=dy_edge_filter,
    bias=dy_bias,
    stride=1,
    padding=1
)

cpu_time = time.perf_counter() - cpu_start

# 활성화 함수 (ReLU) 적용
relu_dx_np = relu(output_dx_np)
relu_dy_np = relu(output_dy_np)
```

> - `dx_edge_filter`, `dy_edge_filter`, `dx_bias`, `dy_bias`는 **§E(셀 120–122)에서 이미 정의**한 것을 재사용한다.
> - `input_np`는 §5 실습에서 만든 **1280×720 이미지**다. 원본 `seagull.jpg`보다 크므로 CPU 처리가 상당히 오래 걸린다.
> - ReLU는 시간 측정 **바깥**에 둔다. 비교 대상은 convolution 연산이지 ReLU가 아니기 때문이다.

### ✅ 셀 334 — CuPy GPU 처리 (변환–워밍업–처리 파이프라인) · **정답**

```python
# convolution을 GPU로 처리 (변환-워밍업-처리 파이프라인 구현)

# ---------- ① 변환: CPU → GPU ----------
input_cp     = cp.asarray(input_np)
dx_filter_cp = cp.asarray(dx_edge_filter)
dy_filter_cp = cp.asarray(dy_edge_filter)
cp.cuda.Stream.null.synchronize()

# ---------- ② 워밍업 ----------
warmup_image = cp.zeros((64, 64), dtype=cp.float32)
_ = convolution2d_cp(
    image=warmup_image,
    kernel=dx_filter_cp,
    bias=dx_bias,
    stride=1,
    padding=1
)
cp.cuda.Stream.null.synchronize()

# ---------- ③ 처리 (시간 측정) ----------
cp.cuda.Stream.null.synchronize()
gpu_start = time.perf_counter()

output_dx_cp = convolution2d_cp(
    image=input_cp,
    kernel=dx_filter_cp,
    bias=dx_bias,
    stride=1,
    padding=1
)

output_dy_cp = convolution2d_cp(
    image=input_cp,
    kernel=dy_filter_cp,
    bias=dy_bias,
    stride=1,
    padding=1
)

cp.cuda.Stream.null.synchronize()
gpu_time = time.perf_counter() - gpu_start

# ---------- ④ ReLU 적용 후 CPU로 회수 ----------
relu_dx_cp_np = cp.asnumpy(cp.maximum(0, output_dx_cp))
relu_dy_cp_np = cp.asnumpy(cp.maximum(0, output_dy_cp))
```

> ⚠️ **`relu()` 함수를 그대로 쓰지 말 것.** §C에서 정의한 `relu(x)`는 `np.maximum(0, x)`이다. CuPy 배열에 넘겨도 디스패치 덕분에 동작은 하지만, **GPU 코드에서는 `cp.maximum(0, x)`를 명시적으로 쓰는 편이 의도가 분명하다.**
>
> 순서도 중요하다 — **GPU에서 ReLU까지 끝낸 뒤 내려받는다.** 먼저 `asnumpy`로 내리고 CPU에서 ReLU를 돌리면 92만 픽셀 연산을 굳이 CPU로 되돌리는 셈이다.

### ✅ 셀 336 — 결과 출력 · **정답**

```python
# CPU/GPU 결과 비교 및 출력

speedup = cpu_time / gpu_time

print("========== NumPy CPU ==========")
print(f"Processing time: {cpu_time:.4f} seconds")
print()

print("========== CuPy GPU ==========")
print(f"Processing time: {gpu_time:.4f} seconds")
print()

print("========== Comparison ==========")
print(f"Speedup: {speedup:.2f}x")
print()

print("========== Accuracy ==========")
print(f"Vertical  max difference: {np.abs(relu_dx_np - relu_dx_cp_np).max():.8f}")
print(f"Horizontal max difference: {np.abs(relu_dy_np - relu_dy_cp_np).max():.8f}")
print("Shape (CPU / GPU):", relu_dx_np.shape, "/", relu_dx_cp_np.shape)
```

> **정확도 확인을 반드시 넣을 것.** 두 함수는 계산 순서가 다르므로 "정말 같은 결과인가"를 검증해야 의미가 있다.
> 최대 오차가 `1e-6` 수준이면 **부동소수 누적 순서 차이일 뿐 동일한 결과**다.

### 📓 셀 338 — 이미지 출력 (원본 그대로)

```python
plt.figure(figsize=(18,8))

plt.subplot(2,3,1), plt.imshow(input_np,      cmap="gray"), plt.title("Input Image")
plt.subplot(2,3,2), plt.imshow(relu_dx_np,    cmap="gray"), plt.title("Vertical Convolution (CPU)")
plt.subplot(2,3,3), plt.imshow(relu_dx_cp_np, cmap="gray"), plt.title("Vertical Convolution (GPU)")
plt.subplot(2,3,4), plt.imshow(input_np,      cmap="gray"), plt.title("Input Image")
plt.subplot(2,3,5), plt.imshow(relu_dy_np,    cmap="gray"), plt.title("Horizontal Convolution (CPU)")
plt.subplot(2,3,6), plt.imshow(relu_dy_cp_np, cmap="gray"), plt.title("Horizontal Convolution (GPU)")

for ax in plt.gcf().axes:
    ax.axis("off")
```

> 2열(CPU)과 3열(GPU)이 **눈으로 구분되지 않아야 정상**이다.

---

## 7. 두 실습에서 얻는 일반 원칙

### 원칙 1 — 안쪽 반복문을 배열 연산으로 바꾼다

| 실습 | CPU 반복 | GPU 반복 | 줄어든 배수 |
|---|---|---|---|
| 잔상 효과 | `H × W × TRAIL` = 2,949만 | `TRAIL` = 32 | **약 92만 배** |
| Convolution | `H × W` = 92만 | `K × K` = 9 | **약 10만 배** |

**두 경우 모두 "픽셀을 도는 반복문"이 사라지고 "커널/이동 거리를 도는 반복문"만 남았다.** 이것이 GPU 가속 코드를 짜는 표준 패턴이다.

### 원칙 2 — 남은 반복문이 "무엇을 도는지" 보라

GPU 코드에도 Python `for`문이 남아 있다. 하지만 **횟수가 데이터 크기와 무관하게 작다(9, 32)**. 이미지가 4K로 커져도 반복 횟수는 그대로다. 반면 CPU 버전은 픽셀 수에 정비례해 늘어난다.

> **좋은 GPU 코드의 판별법**: 이미지 크기를 2배로 키웠을 때 **Python 반복 횟수가 그대로**인가?

### 원칙 3 — 측정에는 항상 동기화 + 워밍업

```python
# 표준 측정 템플릿
_ = my_gpu_function(작은_더미_입력)          # 워밍업
cp.cuda.Stream.null.synchronize()

cp.cuda.Stream.null.synchronize()           # 측정 시작 전
start = time.perf_counter()
result = my_gpu_function(진짜_입력)
cp.cuda.Stream.null.synchronize()           # 측정 종료 전
elapsed = time.perf_counter() - start
```

### 원칙 4 — 전송은 비용이다

```text
[나쁨]  CPU → GPU → CPU → GPU → CPU → GPU → CPU   (연산마다 왕복)
[좋음]  CPU → GPU → (연산1 → 연산2 → 연산3) → CPU  (한 번 올리고 한 번 내림)
```

### 원칙 5 — 정확도를 반드시 검증한다

계산 **순서**를 바꿨으므로 결과가 같은지 확인해야 한다. `np.abs(a - b).max()`가 `1e-6` 이하면 부동소수 오차 수준으로 동일하다.

---

## 부록 A. 복사해서 바로 실행 가능한 전체 코드

> 노트북 §G 전체를 하나로 이어 붙인 버전. `src/images/seagull.jpg`가 있는 디렉터리에서 실행.

```python
import time

import cv2
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt


# ==================== 공통 설정 ====================
IMAGE_PATH = "src/images/seagull.jpg"
TARGET_WIDTH = 1280
TARGET_HEIGHT = 720
TRAIL_LENGTH = 32


def relu(x):
    return np.maximum(0, x)


# ==================== 이미지 준비 ====================
img_gray = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
if img_gray is None:
    raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {IMAGE_PATH}")

img_gray = cv2.resize(img_gray, (TARGET_WIDTH, TARGET_HEIGHT),
                      interpolation=cv2.INTER_AREA)
input_np = img_gray.astype(np.float32) / 255.0


# ==================== 실습 1: 잔상 효과 ====================
def motion_trail_np(image, weights):
    """NumPy — 출력 픽셀을 하나씩 순차 계산"""
    height, width = image.shape
    trail_length = len(weights)
    output = np.zeros((height, width), dtype=np.float32)

    for output_y in range(height):
        for output_x in range(width):
            accumulated = 0.0
            for shift in range(trail_length):
                input_x = output_x - shift
                if input_x < 0:
                    break
                accumulated += image[output_y, input_x] * weights[shift]
            output[output_y, output_x] = accumulated

    return output


def motion_trail_cp(image, weights):
    """CuPy — 이동 거리만 순차 반복, 각 단계의 이미지 전체는 GPU 병렬"""
    height, width = image.shape
    trail_length = len(weights)
    output = cp.zeros((height, width), dtype=cp.float32)

    for shift in range(trail_length):
        input_region  = image[:, :width - shift]
        output_region = output[:, shift:]
        output_region += input_region * weights[shift]

    return output


weights_np = np.linspace(1.0, 0.05, TRAIL_LENGTH, dtype=np.float32)
weights_np /= np.sum(weights_np)

# CPU
cpu_start = time.perf_counter()
output_np = motion_trail_np(image=input_np, weights=weights_np)
cpu_time = time.perf_counter() - cpu_start

# 업로드
upload_start = time.perf_counter()
input_cp   = cp.asarray(input_np)
weights_cp = cp.asarray(weights_np)
cp.cuda.Stream.null.synchronize()
upload_time = time.perf_counter() - upload_start

# 워밍업
warmup_image   = cp.zeros((64, 64), dtype=cp.float32)
warmup_weights = cp.asarray(weights_np[:4])
_ = motion_trail_cp(image=warmup_image, weights=warmup_weights)
cp.cuda.Stream.null.synchronize()

# GPU
cp.cuda.Stream.null.synchronize()
gpu_start = time.perf_counter()
output_cp = motion_trail_cp(image=input_cp, weights=weights_cp)
cp.cuda.Stream.null.synchronize()
gpu_time = time.perf_counter() - gpu_start

# 다운로드
download_start = time.perf_counter()
output_cp_np = cp.asnumpy(output_cp)
download_time = time.perf_counter() - download_start

# 결과
compute_speedup = cpu_time / gpu_time
gpu_total_time  = upload_time + gpu_time + download_time
total_speedup   = cpu_time / gpu_total_time

print("========== Motion Trail ==========")
print(f"CPU:            {cpu_time:.4f}s")
print(f"CPU to GPU:     {upload_time:.4f}s")
print(f"GPU processing: {gpu_time:.4f}s")
print(f"GPU to CPU:     {download_time:.4f}s")
print(f"Compute-only speedup: {compute_speedup:.2f}x")
print(f"Including transfers:  {total_speedup:.2f}x")
print(f"Max difference: {np.abs(output_np - output_cp_np).max():.8f}")


# ==================== 실습 2: Convolution ====================
def convolution2d(image, kernel, bias=0.0, stride=1, padding=0):
    """NumPy — 출력 픽셀을 하나씩 순차 계산"""
    kernel_height, kernel_width = kernel.shape
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


def convolution2d_cp(image, kernel, bias=0.0, stride=1, padding=0):
    """CuPy — 커널 원소만 순차 반복, 출력 배열 전체 연산은 GPU 병렬"""
    kernel_height, kernel_width = kernel.shape
    padded_image = cp.pad(image, ((padding, padding), (padding, padding)),
                          mode="constant", constant_values=0)

    output_height = (padded_image.shape[0] - kernel_height) // stride + 1
    output_width  = (padded_image.shape[1] - kernel_width)  // stride + 1
    output = cp.zeros((output_height, output_width), dtype=cp.float32)

    for kernel_y in range(kernel_height):
        for kernel_x in range(kernel_width):
            input_region = padded_image[
                kernel_y : kernel_y + output_height * stride : stride,
                kernel_x : kernel_x + output_width  * stride : stride,
            ]
            output += input_region * kernel[kernel_y, kernel_x]

    return output + bias   # bias는 마지막에 한 번만!


dx_edge_filter = np.array([[-1, 0, 1],
                           [-1, 0, 1],
                           [-1, 0, 1]], dtype=np.float32)
dy_edge_filter = np.array([[-1, -1, -1],
                           [ 0,  0,  0],
                           [ 1,  1,  1]], dtype=np.float32)
dx_bias = -0.2
dy_bias = -0.2

# CPU
cpu_start = time.perf_counter()
output_dx_np = convolution2d(input_np, dx_edge_filter, dx_bias, stride=1, padding=1)
output_dy_np = convolution2d(input_np, dy_edge_filter, dy_bias, stride=1, padding=1)
cpu_time = time.perf_counter() - cpu_start
relu_dx_np = relu(output_dx_np)
relu_dy_np = relu(output_dy_np)

# 변환
input_cp     = cp.asarray(input_np)
dx_filter_cp = cp.asarray(dx_edge_filter)
dy_filter_cp = cp.asarray(dy_edge_filter)
cp.cuda.Stream.null.synchronize()

# 워밍업
_ = convolution2d_cp(cp.zeros((64, 64), dtype=cp.float32),
                     dx_filter_cp, dx_bias, stride=1, padding=1)
cp.cuda.Stream.null.synchronize()

# GPU
cp.cuda.Stream.null.synchronize()
gpu_start = time.perf_counter()
output_dx_cp = convolution2d_cp(input_cp, dx_filter_cp, dx_bias, stride=1, padding=1)
output_dy_cp = convolution2d_cp(input_cp, dy_filter_cp, dy_bias, stride=1, padding=1)
cp.cuda.Stream.null.synchronize()
gpu_time = time.perf_counter() - gpu_start

relu_dx_cp_np = cp.asnumpy(cp.maximum(0, output_dx_cp))
relu_dy_cp_np = cp.asnumpy(cp.maximum(0, output_dy_cp))

print("\n========== Convolution ==========")
print(f"CPU: {cpu_time:.4f}s / GPU: {gpu_time:.4f}s / Speedup: {cpu_time/gpu_time:.2f}x")
print(f"Vertical  max difference: {np.abs(relu_dx_np - relu_dx_cp_np).max():.8f}")
print(f"Horizontal max difference: {np.abs(relu_dy_np - relu_dy_cp_np).max():.8f}")


# ==================== 시각화 ====================
plt.figure(figsize=(18, 12))
plt.subplot(1,3,1), plt.imshow(input_np,     cmap="gray"), plt.title("Original Image")
plt.subplot(1,3,2), plt.imshow(output_np,    cmap="gray"), plt.title("Motion Trail (CPU)")
plt.subplot(1,3,3), plt.imshow(output_cp_np, cmap="gray"), plt.title("Motion Trail (GPU)")
for ax in plt.gcf().axes:
    ax.axis("off")

plt.figure(figsize=(18, 8))
plt.subplot(2,3,1), plt.imshow(input_np,      cmap="gray"), plt.title("Input Image")
plt.subplot(2,3,2), plt.imshow(relu_dx_np,    cmap="gray"), plt.title("Vertical Convolution (CPU)")
plt.subplot(2,3,3), plt.imshow(relu_dx_cp_np, cmap="gray"), plt.title("Vertical Convolution (GPU)")
plt.subplot(2,3,4), plt.imshow(input_np,      cmap="gray"), plt.title("Input Image")
plt.subplot(2,3,5), plt.imshow(relu_dy_np,    cmap="gray"), plt.title("Horizontal Convolution (CPU)")
plt.subplot(2,3,6), plt.imshow(relu_dy_cp_np, cmap="gray"), plt.title("Horizontal Convolution (GPU)")
for ax in plt.gcf().axes:
    ax.axis("off")
plt.show()
```

---

## 부록 B. NumPy ↔ CuPy 치환표

| 하는 일 | NumPy | CuPy |
|---|---|---|
| import | `import numpy as np` | `import cupy as cp` |
| 배열 생성 | `np.array([...])` | `cp.array([...])` |
| 0으로 채운 배열 | `np.zeros((h,w), dtype=np.float32)` | `cp.zeros((h,w), dtype=cp.float32)` |
| 연속 정수 | `np.arange(10)` | `cp.arange(10)` |
| 난수 | `np.random.rand(n,n).astype(np.float32)` | `cp.random.rand(n,n, dtype=cp.float32)` |
| 패딩 | `np.pad(...)` | `cp.pad(...)` — 인자 동일 |
| ReLU | `np.maximum(0, x)` | `cp.maximum(0, x)` |
| 합/최대/평균 | `np.sum(x)` | `cp.sum(x).item()` |
| 행렬곱 | `a @ b` | `a @ b` — 동일 |
| **CPU → GPU** | — | `cp.asarray(x)` |
| **GPU → CPU** | — | `cp.asnumpy(x)` |
| **동기화** | 불필요 | `cp.cuda.Stream.null.synchronize()` |
| 화면 출력 | `plt.imshow(x)` | `plt.imshow(cp.asnumpy(x))` |

---

## 부록 C. CuPy 관련 에러 대처

| 증상 | 원인 | 해결 |
|---|---|---|
| `RuntimeError` — NumPy와 CuPy 배열 혼합 연산 | 서로 다른 메모리 공간 | `cp.asarray()` 또는 `cp.asnumpy()`로 한쪽에 맞춘다 |
| `TypeError: Implicit conversion to a NumPy array is not allowed` | `cv2`/`plt`에 CuPy 배열을 넘김 | `cp.asnumpy()`로 내려서 전달 |
| `IndexError` — warm-up에서만 발생 | 함수에 `TRAIL_LENGTH`를 하드코딩 | `len(weights)`를 사용 |
| 결과가 전체적으로 너무 밝거나 어두움 | `bias`를 루프 안에서 반복 가산 | 루프 밖에서 **한 번만** 더한다 |
| GPU 시간이 0에 가깝게 측정됨 | 동기화 누락 | 측정 앞뒤로 `synchronize()` |
| 첫 측정만 유독 느림 | 워밍업 누락 | 측정 전 더미 연산 실행 |
| speedup이 1보다 작음 (GPU가 더 느림) | 데이터가 작거나 전송 비용이 지배적 | 정상적인 현상. 전송을 줄이거나 더 큰 데이터로 비교 |
| `output` shape이 CPU 버전과 다름 | 슬라이스 `stop`을 `padded.shape`로 잡음 | `k + output_size * stride`로 개수를 명시 |
| 마지막 shift 결과만 남음 | `output_region = ...` (덮어쓰기) | `output_region += ...` (누적) |
| `cupy.cuda.memory.OutOfMemoryError` | VRAM 부족 | 이미지 크기/배치를 줄이거나 `cp.get_default_memory_pool().free_all_blocks()` |

---

*원본: `03_DL-and-GPU.ipynb` §G — Physical AI의 Vision-LLM 융합 시청각 멀티모달 시스템 (김규래)*
*딥러닝 전반 정리는 `03_DL-and-GPU_정리.md` 참고*

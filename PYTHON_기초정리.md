# 01_Linux-and-Python — 파이썬 기초 정리

`01_Linux-and-Python.ipynb`의 Python 파트를 주제별로 재구성한 개인 학습용 노트.
각 절은 **핵심 → 코드 → 헷갈리는 지점** 순서입니다.

---

## 1. 데이터 타입과 출력

### 핵심
Python은 변수에 타입을 명시하지 않고 대입한 값으로 타입이 정해진다 (동적 타이핑).
`type()`으로 확인하고, 타입이 다르면 **문자열 연결(`+`)이 실패**하므로 변환이 필요하다.

```python
name = "철수"          # str
age = 20               # int
height = 178.31754081  # float

print(f"이름: {type(name)}, 나이: {type(age)}, 키: {type(height)}")
print("성인 여부:", age > 19)   # bool
```

### 출력하는 3가지 방법

| 방법 | 예시 | 특징 |
|---|---|---|
| 콤마 구분 | `print("나이:", age)` | 자동으로 공백 삽입, 형변환 불필요 |
| 문자열 연결 | `print("나이: " + str(age))` | **반드시 `str()` 변환 필요** |
| f-string | `print(f"나이: {age}세")` | 가장 권장. 중괄호 안에 식을 그대로 쓸 수 있음 |

```python
print("이름:", name, end=", ")
print("나이: " + str(age) + "세", end=", ")
print(f"키(m): {height/100:.2f}m", end=", ")
print(f"키(cm):", str(height)[0:5] + "cm", end="\n\n")
```

- `end=`는 출력 끝에 붙일 문자. 기본값은 `"\n"`(줄바꿈)이라, `end=", "`를 주면 **줄바꿈 없이 이어서** 출력된다.
- `{height/100:.2f}` — 콜론 뒤는 **포맷 스펙**. `.2f`는 소수점 2자리 고정.
- `str(height)[0:5]` — 숫자를 문자열로 바꾼 뒤 앞 5글자만 슬라이싱 (`178.3`).

### 헷갈리는 지점
`"20" + 5` → `TypeError`. `"20" * 3` → `"202020"` (에러가 아니라 반복!).
문자열 숫자를 계산에 쓰려면 반드시 `int()` / `float()`로 변환.

### 연습문제 (cell 32)
```python
score_str = "0.8825"

score_int = int(float(score_str) * 100)
print("최종 점수: " + str(score_int) + "점")   # 최종 점수: 88점
```
> `int("0.8825")`는 **에러**다. 소수점 문자열은 `float()`를 먼저 거쳐야 한다.
> `int()`는 반올림이 아니라 **버림**: `int(88.25)` → `88`.

---

## 2. List

### 생성
```python
lst1 = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]  # 길이 10, 인덱스 0~9
lst2 = list(map(int, input("숫자 입력 (띄어쓰기로 구분): ").split()))
```

`lst2` 한 줄을 안쪽부터 풀어보면:

| 단계 | 결과 |
|---|---|
| `input()` | `"1 2 3"` (문자열 하나) |
| `.split()` | `["1", "2", "3"]` (공백 기준 분리, 여전히 문자열) |
| `map(int, ...)` | 각 원소에 `int()` 적용 (지연 평가 객체) |
| `list(...)` | `[1, 2, 3]` (실제 리스트로 확정) |

> `map`은 그 자체로 리스트가 아니라 **이터레이터**다. `print(map(...))`하면 값이 아니라
> `<map object at 0x...>`가 나오므로 `list()`로 감싸야 한다.

### 인덱싱
```python
lst1[0]    # "0"  첫 번째
lst1[9]    # "9"  마지막
lst1[10]   # IndexError!  (인덱스는 0부터 시작하므로 마지막은 길이-1)

lst1[-1]   # "9"  뒤에서 첫 번째
lst1[-10]  # "0"  뒤에서 열 번째 = 첫 번째
```
- 양수 인덱스: `0 ~ len-1`
- 음수 인덱스: `-1 ~ -len`
- 이 범위를 벗어나면 `IndexError`

### 원소 수정 (리스트는 mutable)
```python
lst1[0]  = "X"
lst1[-1] = "Y"
# ["X", "1", ..., "8", "Y"]
```

> **노트북 실행 순서 주의**: 이 셀을 실행하면 `lst1`이 실제로 바뀐다.
> 아래 슬라이싱 셀을 다시 돌리면 `"0"`, `"9"` 대신 `"X"`, `"Y"`가 나온다.
> 결과가 이상하면 리스트 생성 셀부터 다시 실행할 것.

### 슬라이싱 `lst[start:stop:step]`

**핵심 규칙: `start`는 포함, `stop`은 제외.**

```python
lst1[0:1]         # ["0"]              1개만. stop=1은 제외되므로
lst1[0:9]         # 인덱스 0~8 (9개)   마지막 원소가 빠진다!
lst1[0:10]        # 전체 10개
lst1[0:len(lst1)] # 전체 (위와 동일한 관용 표현)
lst1[0:-1]        # 마지막 하나 빼고 전부
lst1[:]           # 전체 (복사본 생성)
lst1[:3]          # 앞 3개    start 생략 = 처음부터
lst1[3:]          # 인덱스 3부터 끝까지   stop 생략 = 끝까지
lst1[0:5:2]       # 0,2,4번   step=2 → 2칸씩 건너뜀
lst1[::2]         # 짝수 인덱스 전체
lst1[::-1]        # 뒤집기 (step=-1)

new_lst = lst1[:3] + lst2   # 리스트끼리 + 는 이어붙이기(concatenation)
```

### 헷갈리는 지점
- **인덱싱은 범위를 넘으면 에러, 슬라이싱은 에러 없이 잘라준다.**
  `lst1[100]` → `IndexError` / `lst1[0:100]` → 그냥 전체 반환
- `lst1[:]`는 **새 리스트 객체**를 만든다. `lst2 = lst1`은 같은 객체를 가리키므로
  한쪽을 바꾸면 다른 쪽도 바뀐다. 복사하려면 `lst2 = lst1[:]`.

---

## 3. 반복문과 List Comprehension

### 중첩 반복문
```python
for i in range(10):        # 바깥 루프: i가 한 번 바뀌는 동안
    for j in range(10):    # 안쪽 루프는 10번 전부 돈다
        my_tuple = (i, j)  # (0,0) (0,1) (0,2) ... (0,9) → 그 다음 i=1
        print(my_tuple)
    print("\n------\n")
```
총 `10 × 10 = 100`번 실행. **안쪽이 빨리, 바깥이 느리게** 변한다.

### List Comprehension — 같은 일을 한 줄로

```python
# 일반 for문
my_list1 = []
for i in range(10):
    my_list1.append(i)

# 컴프리헨션 (동일 결과)
my_list1 = [i for i in range(10)]
```

문법 구조:
```
[ 담을_값   for 변수 in 반복대상   if 조건 ]
    (1)           (2)               (3)
```
읽는 순서는 **(2) → (3) → (1)**: 반복하면서 → 조건을 통과한 것만 → 이 값으로 담는다.

```python
test_lst1 = [0, 10, 100]
test_lst2 = [x * 100 for x in test_lst1]   # [0, 1000, 10000]
```

### 중첩 컴프리헨션
```python
# for문 버전
my_arr1 = []
for i in range(10):
    for j in range(10):
        my_arr1.append([i, j])

# 컴프리헨션 버전 — for를 쓴 순서가 그대로 중첩 순서
my_arr2 = [[i, j] for i in range(10) for j in range(10)]

# 조건 추가
my_arr3 = [[i, j] for i in range(10) for j in range(10) if i <= 1 and j <= 1]
# [[0,0], [0,1], [1,0], [1,1]]
```
> `for`를 나열하는 순서는 일반 for문에서 **위에서 아래로 쓴 순서와 같다**. 헷갈리면
> 일반 for문으로 먼저 쓴 뒤 그대로 옮겨 적으면 된다.

### 연습문제 (cell 49) — 2차원 리스트 평탄화 + 필터
```python
scores = [[45, 88, 92],
          [30, 75, 60],
          [95, 40, 50]]

passed_scores = [score for row in scores for score in row if score >= 50]
print("합격 점수들:", passed_scores)   # [88, 92, 75, 60, 95, 50]
```
바깥 `for row in scores`로 행을 꺼내고, 안쪽 `for score in row`로 그 행의 원소를 꺼낸 뒤
`if score >= 50`을 통과한 값만 담는다. 결과는 **1차원 리스트**가 된다.

---

## 4. enumerate & zip

```python
students = ["짱구", "철수", "맹구"]
scores   = [85, 92, 78]
```

### enumerate — 인덱스를 같이 준다
```python
print(tuple(enumerate(students)))
# ((0, '짱구'), (1, '철수'), (2, '맹구'))

for idx, name in enumerate(students):
    print(f"{idx}번 학생: {name}")        # 0번부터

for idx, name in enumerate(students, start=1):
    print(f"{idx}번 학생: {name}")        # 1번부터 (사람이 읽기 좋은 번호)
```
> `for i in range(len(students))` 후 `students[i]`로 접근하는 것보다 훨씬 안전하고 읽기 쉽다.

### zip — 여러 리스트를 짝지어 준다
```python
print(tuple(zip(students, scores)))
# (('짱구', 85), ('철수', 92), ('맹구', 78))

for name, score in zip(students, scores):
    print(f"{name}의 점수: {score}점")
```
> 길이가 다르면 **짧은 쪽에 맞춰 잘린다** (에러가 아님). 의도치 않은 데이터 누락에 주의.

### 연습문제 (cell 57) — 둘을 조합
```python
for idx, (name, score) in enumerate(zip(students, scores), start=1):
    if score >= 80:
        print(f"{idx}번 {name} 학생 (점수: {score}점)")
# 1번 짱구 학생 (점수: 85점)
# 2번 철수 학생 (점수: 92점)
```
**핵심 문법**: `zip`이 만든 튜플 `('짱구', 85)`를 `enumerate`가 다시 감싸
`(0, ('짱구', 85))` 형태가 된다. 그래서 받을 때도 `idx, (name, score)`처럼
**괄호로 묶어서 풀어야(unpacking)** 한다.

---

## 5. Dictionary, JSON, YAML

### Dictionary — key로 값을 찾는 자료구조
```python
student_info = {
    "name": "맹구",
    "age": 21,
    "major": "컴퓨터공학"
}

for key, value in student_info.items():   # .items() → (key, value) 쌍
    print(f"{key}: {value}")

print("이름:", student_info["name"])

# 없는 key에 대입하면 새로 추가된다
student_info["gpa"] = 3.8
student_info["languages"] = ["Python", "C++"]   # 값으로 리스트도 가능
```
- `.keys()` key만 / `.values()` 값만 / `.items()` 쌍으로
- 없는 key를 **읽으면** `KeyError`, **쓰면** 새 항목 추가. (읽기는 `.get("key")`가 안전 — 없으면 `None`)

### JSON 저장
```python
import json

with open("맹구_개인정보.json", "w", encoding="utf-8") as f:
    json.dump(student_info, f, ensure_ascii=False, indent=4)
```
| 인자 | 의미 |
|---|---|
| `"w"` | 쓰기 모드 (기존 내용 덮어씀). 읽기는 `"r"` |
| `encoding="utf-8"` | 한글 깨짐 방지 |
| `ensure_ascii=False` | **없으면 한글이 `"\ub9f9\uad6c"`처럼 이스케이프되어 저장된다.** 반드시 지정 |
| `indent=4` | 들여쓰기로 보기 좋게 (없으면 한 줄로 압축) |

> `with open(...) as f:` 구문은 블록이 끝나면 **파일을 자동으로 닫아준다.** `f.close()` 불필요.

### YAML 저장
```python
import yaml

with open("맹구_개인정보.yaml", "w", encoding="utf-8") as f:
    yaml.safe_dump(student_info, f, allow_unicode=True, default_flow_style=False)
```
| 인자 | 의미 |
|---|---|
| `allow_unicode=True` | JSON의 `ensure_ascii=False`에 해당. 한글 그대로 저장 |
| `default_flow_style=False` | 블록 스타일(줄바꿈 형식). `True`면 `{a: 1, b: 2}` 한 줄 |

> `safe_dump` / `safe_load`를 쓸 것. 그냥 `load`는 임의 코드 실행 위험이 있다.

**JSON vs YAML**: 둘 다 딕셔너리를 파일로 저장하는 형식. JSON은 데이터 교환용으로 널리 쓰이고,
YAML은 주석이 가능하고 사람이 읽기 편해 **설정 파일**로 많이 쓴다 (로봇/ROS 설정이 대표적).

---

## 6. NumPy

```python
import numpy as np
```
리스트와 달리 **같은 타입만** 담고, 전체에 연산을 한 번에 적용(벡터화)할 수 있어 훨씬 빠르다.
영상 처리에서 이미지는 결국 NumPy 배열이므로 이 절이 가장 중요하다.

### 배열 생성과 dtype 변환
```python
str_arr = np.array(["1.5", "2.8", "3.1"])     # dtype: 문자열
float_arr = str_arr.astype(np.float64)         # 숫자로 변환
print(float_arr.dtype)   # float64

# 한 줄로
float_arr = np.array(["1.5", "2.8", "3.1"]).astype(np.float64)
```
- `.dtype` — 배열 원소의 자료형
- `.astype()` — **새 배열을 반환**한다. 원본은 그대로이므로 반드시 결과를 변수에 받아야 한다.
- 이미지 처리에서 `uint8`(0~255) ↔ `float32`(0.0~1.0) 변환에 계속 쓰인다.

### 2차원 인덱싱 / 슬라이싱
```python
grid = np.array([
    [10, 20, 30],
    [40, 50, 60],
    [70, 80, 90]
])

grid[:2, 1:]   # 우측 상단 2x2
# [[20, 30],
#  [50, 60]]
```
**`[행, 열]` 순서**이고 콤마로 구분한다. 각 자리에 리스트와 똑같은 슬라이싱 규칙이 적용된다.
- `:2` → 행 0~1
- `1:` → 열 1~2

> 리스트였다면 `lst[0][1]`처럼 대괄호를 두 번 썼겠지만, NumPy는 `grid[0, 1]` 한 번에 쓴다.
> 이미지 자르기(crop)가 정확히 이 문법이다: `img[y1:y2, x1:x2]`

### 집계 함수
```python
np_arr = np.array([1, 2, 3, 4, 5])

np_arr.sum()      # 15
np_arr.min()      # 1
np_arr.max()      # 5
np_arr.mean()     # 3.0
np_arr.argmax()   # 4  ← 최댓값이 아니라 최댓값의 "위치(인덱스)"
```
> `max`와 `argmax`의 차이가 자주 나온다. `max` = 값, `arg`max = 그 값이 있는 **인덱스**.

### axis — 어느 방향으로 계산할 것인가
```python
# [행: 학생, 열: 과목]
scores = np.array([
    [83, 92, 100],  # 0번 학생
    [76, 63,  58],  # 1번 학생
    [91, 99,  84]   # 2번 학생
])

student_means = np.mean(scores, axis=1)   # 학생별 평균 → [91.67, 65.67, 91.33]
subject_means = np.mean(scores, axis=0)   # 과목별 평균 → [83.33, 84.67, 80.67]
```

**axis 외우는 법**: `axis=n`은 **n번 축을 없애는(합치는) 방향**이다.

| | 의미 | 결과 개수 |
|---|---|---|
| `axis=0` | 행 방향으로 훑음 → **행이 사라짐** → 열별(과목별) 결과 | 열 개수만큼 (3개) |
| `axis=1` | 열 방향으로 훑음 → **열이 사라짐** → 행별(학생별) 결과 | 행 개수만큼 (3개) |
| 생략 | 전체를 하나로 | 1개 (스칼라) |

### Boolean 마스킹과 np.where
```python
scores = np.array([85, 90, 100, 60, 90, 75, 90])

is_ninety = (scores == 90)
print(is_ninety)   # [False, True, False, False, True, False, True]

idx_ninety = np.where(scores == 90)
print(list(idx_ninety[0]))   # [1, 4, 6]  ← 조건을 만족하는 인덱스

print((scores == 90).sum())   # 3  ← True를 1로 세어 개수를 구함
print((scores >= 80).sum())   # 5
```
**핵심 관용구**: `(조건).sum()` = 조건을 만족하는 **개수**.
`True == 1`, `False == 0`이라서 합이 곧 개수가 된다. 아주 자주 쓰이니 익혀둘 것.

> `np.where`는 튜플을 반환한다. 1차원 배열이면 `[0]`으로 첫 번째 원소를 꺼내야 인덱스 배열이 나온다.

### Flatten — 다차원을 1차원으로
```python
grid = np.array([[10,20,30], [40,50,60], [70,80,90]])
flat_grid = grid.flatten()

grid.shape        # (3, 3)
flat_grid.shape   # (9,)
flat_grid         # [10 20 30 40 50 60 70 80 90]
```
`.shape` = 배열의 모양(행, 열). 행 우선(row-major) 순서로 이어붙인다.

### 연습문제 (cell 79) — 정규화
```python
np_arr_normalized = np_arr / np_arr.sum()
# 합이 1이 되도록. 확률 분포로 만들 때 쓴다.
```
> `np_arr / np_arr.sum()` — 배열 ÷ 숫자 하나면 **모든 원소에 각각 적용**된다 (브로드캐스팅).
> 리스트였다면 for문을 돌려야 했을 일이다.

### 연습문제 (cell 80) — 0.0~1.0 범위로 스케일링 (Min-Max)
```python
np_arr_float32   = np_arr.astype(np.float32)
np_arr_shifted   = np_arr_float32 - np_arr_float32.min()   # 최솟값을 0으로
np_arr_final     = np_arr_shifted / np_arr_shifted.max()   # 최댓값을 1로
print(np_arr_final.dtype)   # float32
```
**순서가 중요하다**: (1) 먼저 `float32`로 바꾼다 → (2) 최솟값을 빼서 0부터 시작하게 만든다
→ (3) 최댓값으로 나눠 1까지 펴준다.

> (1)을 건너뛰면 정수 나눗셈 때문에 값이 뭉개진다. 이미지 전처리의 표준 절차이므로 통째로 외워둘 것.
> 공식: `(x - min) / (max - min)` — 위 코드는 이걸 두 단계로 나눠 쓴 것이다.

---

## 7. Section Project 전체 풀이

```python
raw_students = ["zg0505_짱구", "cs0822_철수", "mg0910_맹구", "yr0605_유리", "hi0205_훈이"]
raw_scores = np.array([
    [95,  82, 88],   # 짱구
    [70,  65, 90],   # 철수
    [98, 100, 92],   # 맹구
    [60,  75, 80],   # 유리
    [81,  88, 84],   # 훈이
])
subjects = ["국어", "수학", "과학"]
```

**(1) 이름만 추출** — `"zg0505_짱구"`에서 앞 7글자(`zg0505_`)를 잘라낸다
```python
student_names = [name[7:] for name in raw_students]
```

**(2) 전체 성적 1차원화**
```python
all_scores = raw_scores.flatten()   # 5×3 = 15개
```

**(3) 85점 이상 개수** — Boolean 마스킹 관용구
```python
over_85 = (all_scores >= 85).sum()
```

**(4) 학생별 평균** — 행별이므로 `axis=1`
```python
student_means = np.mean(raw_scores, axis=1)

print("4. 학생별 평균 점수: ", end="")
for idx, (name, mean) in enumerate(zip(student_names, student_means)):
    sep = ", " if idx < len(student_names) - 1 else "\n\n"
    print(f"{name} {mean:.1f}점", end=sep)
```
> `sep = ", " if 조건 else "\n\n"` — **삼항 연산자**. 마지막 항목에만 다른 구분자를 붙여
> `A, B, C` 형태로 깔끔하게 출력하는 패턴.

**(5) 최고 평균 학생** — `argmax`로 인덱스를 얻어 이름 리스트에 그대로 대입
```python
top_student_idx  = np.argmax(student_means)
top_student_name = student_names[top_student_idx]
```

**(6) 단일 최고점의 과목과 학생** — 이 문제의 핵심
```python
overall_max_score   = np.max(all_scores)
overall_max_idx     = np.argmax(all_scores)                           # 1차원에서의 위치
overall_max_subject = subjects[overall_max_idx % len(subjects)]       # 나머지 → 열(과목)
overall_max_student = student_names[overall_max_idx // len(subjects)] # 몫   → 행(학생)
```

**왜 `%`와 `//`인가?**
`flatten()`이 2차원을 1차원으로 폈으므로, 1차원 인덱스에서 원래 (행, 열)을 되돌려야 한다.
열이 3개이므로:

```
1차원 인덱스:  0   1   2 | 3   4   5 | 6   7   8 | ...
              국  수  과 | 국  수  과 | 국  수  과
학생(행):     └─ 짱구 ─┘ └─ 철수 ─┘ └─ 맹구 ─┘
```
- `idx // 3` (몫)    = 몇 번째 묶음인가 → **행 = 학생**
- `idx % 3` (나머지) = 묶음 안 몇 번째인가 → **열 = 과목**

예: 최고점 100은 `idx=7` → `7 // 3 = 2` (맹구), `7 % 3 = 1` (수학).

> `np.unravel_index(idx, raw_scores.shape)`를 쓰면 `(2, 1)`을 한 번에 얻을 수도 있다.

---

## 8. 최종 치트시트

### 자주 틀리는 것
| 상황 | 틀린 코드 | 맞는 코드 |
|---|---|---|
| 소수점 문자열 → 정수 | `int("0.88")` | `int(float("0.88"))` |
| 문자열 + 숫자 | `"나이" + 20` | `"나이" + str(20)` 또는 f-string |
| 리스트 복사 | `b = a` (같은 객체) | `b = a[:]` |
| 마지막 원소 포함 | `lst[0:9]` (9번 제외) | `lst[0:10]` 또는 `lst[:]` |
| 한글 JSON 저장 | `json.dump(d, f)` | `ensure_ascii=False` 추가 |
| 한글 YAML 저장 | `yaml.safe_dump(d, f)` | `allow_unicode=True` 추가 |
| astype 결과 | `arr.astype(np.float32)` 만 호출 | `arr = arr.astype(np.float32)` |

### 꼭 외울 관용구
```python
list(map(int, input().split()))          # 공백 구분 정수 입력
lst[::-1]                                # 리스트 뒤집기
[x for row in mat for x in row]          # 2차원 → 1차원
enumerate(zip(a, b), start=1)            # 번호 + 두 리스트 동시 순회
(arr >= 80).sum()                        # 조건 만족 개수
np.mean(arr, axis=1)                     # 행별(학생별) 평균
np.argmax(arr)                           # 최댓값의 위치
idx // n_cols,  idx % n_cols             # 1차원 인덱스 → (행, 열)
(x - x.min()) / (x.max() - x.min())      # 0~1 정규화
```

### 자료구조 한눈에
| | 리스트 `[]` | 튜플 `()` | 딕셔너리 `{}` | NumPy 배열 |
|---|---|---|---|---|
| 수정 | 가능 | **불가** | 가능 | 가능 |
| 접근 | 인덱스 | 인덱스 | key | 인덱스 `[행, 열]` |
| 타입 혼용 | 가능 | 가능 | 가능 | **불가** (단일 dtype) |
| 전체 연산 | 불가 (for 필요) | 불가 | 불가 | **가능** (벡터화) |

---

*원본: `01_Linux-and-Python.ipynb` / 풀이 대조: `[ANSWER]_01_Linux-and-Python.ipynb`*

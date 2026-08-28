# 정리노트

`linux-work` 브랜치의 주피터 노트북 01–04를 세션별로 재구성한 학습용 정리노트입니다.
각 파일은 **독립 실행 가능한 HTML**이라 브라우저로 바로 열 수 있고, 인쇄(Ctrl+P)하면 PDF로 저장됩니다.

## 파일

| 파일 | 원본 노트북 | 내용 |
|---|---|---|
| [`01_Linux-and-Python_정리.html`](01_Linux-and-Python_정리.html) | `01_Linux-and-Python.ipynb` | 터미널 명령어 · List/Dict · JSON/YAML · NumPy axis |
| [`02_Computer-Vision_정리.html`](02_Computer-Vision_정리.html) | `[ANSWER]_02_Computer-Vision.ipynb` | 색공간 · 필터링 · Edge · Contour · Kalman · MediaPipe |
| [`03_DL-and-GPU_정리.html`](03_DL-and-GPU_정리.html) | `03_DL-and-GPU.ipynb` | 퍼셉트론 · 역전파 · CNN · CuPy/PyTorch GPU 가속 |
| [`04_DL-Object-Detection_정리.html`](04_DL-Object-Detection_정리.html) | `04_DL-Object-Detection.ipynb` | BBox · IoU · NMS · YOLO · TensorRT · INT8 양자화 |

## 구성 방식

네 문서 모두 같은 형식을 따릅니다.

- 좌측 **사이드바 목차** — 스크롤에 따라 현재 섹션이 강조됩니다
- 섹션마다 **개념 → 코드 → 함정** 순서
- `POINT` / `함정` / `실습 팁` 블록에 헷갈리기 쉬운 지점을 모았습니다
- 다크 모드와 인쇄 레이아웃을 함께 지원합니다

## 같은 내용의 마크다운본

저장소 루트에 03 세션을 더 자세히 풀어 쓴 마크다운 문서가 있습니다.

- `03_DL-and-GPU_정리.md` — 개념 전체 + 노트북 TODO 셀 모범답안
- `03_CuPy-GPU_정리.md` — CuPy 파트 심화 + 실습 정답
- `03_함수-명령어_레퍼런스.md` — 라이브러리별 함수·명령어 빠른 조회

## 알려진 문제

`linux-work` 브랜치 `03_DL-and-GPU.ipynb`의 `convolution2d_cp`(셀 331)는 현재 동작하지 않습니다.
`start_y`가 `kernel_y`를 참조하지 않아 커널 원소가 바뀌어도 같은 영역을 보게 되고, 슬라이스가 범위를 벗어납니다.
올바른 코드는 03 정리노트의 G장에 있습니다.

---

원본: Physical AI의 Vision-LLM 융합 시청각 멀티모달 시스템 (김규래)

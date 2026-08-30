# 정리노트

`linux-work` 브랜치의 주피터 노트북 01–06을 세션별로 재구성한 학습용 정리노트입니다.
각 파일은 **독립 실행 가능한 HTML**이라 브라우저로 바로 열 수 있고, 인쇄(Ctrl+P)하면 PDF로 저장됩니다.

## 파일

| 파일 | 원본 노트북 | 내용 |
|---|---|---|
| [`01_Linux-and-Python_정리.html`](01_Linux-and-Python_정리.html) | `01` + `[ANSWER]_01` | 터미널 명령어 · List/Dict · JSON/YAML · NumPy axis |
| [`02_Computer-Vision_정리.html`](02_Computer-Vision_정리.html) | `[ANSWER]_02` | 색공간 · 필터링 · Edge · Contour · Kalman · MediaPipe |
| [`03_DL-and-GPU_정리.html`](03_DL-and-GPU_정리.html) | `03` + `[ANSWER]_03` | 퍼셉트론 · 역전파 · CNN · CuPy/PyTorch GPU 가속 |
| [`04_DL-Object-Detection_정리.html`](04_DL-Object-Detection_정리.html) | `04` + `[ANSWER]_04` | BBox · IoU · NMS · YOLO · TensorRT · INT8 양자화 |
| [`05_LLM-and-Gemma_정리.html`](05_LLM-and-Gemma_정리.html) | `05-LLM_and_Gemma` | Tokenizer · Context Window · Prompt · Memory · 로컬 Gemma |
| [`06_Vision-LLM-Multimodal_정리.html`](06_Vision-LLM-Multimodal_정리.html) | `06-Vision_LLM_Multimodal_Systems` | Vision-to-Text · 멀티모달 · ROI · STT/TTS |

## 구성 방식

여섯 문서 모두 같은 형식을 따릅니다.

- 좌측 **사이드바 목차** — 스크롤에 따라 현재 섹션이 강조됩니다
- 섹션마다 **개념 → 코드 → 함정** 순서
- `POINT` / `함정` / `실습 팁` 블록에 헷갈리기 쉬운 지점을 모았습니다
- 01–04에는 **실습 정답 코드** 챕터가 따로 있습니다 (`[ANSWER]` 노트북 기준)
- 다크 모드와 인쇄 레이아웃을 함께 지원합니다

## 같은 내용의 마크다운본

저장소 루트에 03 세션을 더 자세히 풀어 쓴 마크다운 문서가 있습니다.

- `03_DL-and-GPU_정리.md` — 개념 전체 + 노트북 TODO 셀 모범답안
- `03_CuPy-GPU_정리.md` — CuPy 파트 심화 + 실습 정답
- `03_함수-명령어_레퍼런스.md` — 라이브러리별 함수·명령어 빠른 조회

## 원본 노트북에서 발견한 문제

정리 과정에서 확인한 것들입니다. 해당 정리노트에 올바른 코드와 함께 적어 두었습니다.

1. **`[ANSWER]_03` 셀 133 — Sharpening 필터 정규화**
   이 커널의 합은 `-3.252`(음수)라서 합으로 나누면 **모든 부호가 뒤집힙니다**.
   중심값이 `+3.0 → -0.923`이 되어 선명해지는 대신 반전된 흐릿한 이미지가 나옵니다.
   계단 엣지로 확인하면 대비가 `+0.92`에서 `-0.28`로 바뀝니다.
   Sharpening 커널은 정규화 단계를 건너뛰어야 합니다.

2. **`03` (비-ANSWER) 셀 331 — `convolution2d_cp`**
   `start_y`가 `kernel_y`를 참조하지 않아 커널 원소가 바뀌어도 같은 영역을 보게 되고,
   슬라이스가 범위를 벗어납니다. `[ANSWER]_03`의 구현이 올바릅니다.

3. **`[ANSWER]_01` 셀 89 — Section Project 6번**
   `overall_max_student` 계산이 중간에 잘려 있습니다. 평탄화된 인덱스를 과목 수로 나눈
   **몫이 학생, 나머지가 과목**입니다.

---

원본: Physical AI의 Vision-LLM 융합 시청각 멀티모달 시스템 (김규래)

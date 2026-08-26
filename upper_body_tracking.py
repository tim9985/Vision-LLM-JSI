"""상반신 인식 + 손바닥 경계 판별 + 팔 관절 추적 + 경계 왕복 횟수 카운트

MediaPipe Pose Landmarker로 상반신(머리/귀/어깨/팔/손)을 추적하면서
 1) 손바닥이 [머리 위 / 머리~귀 / 귀~어깨 / 어깨 아래] 중 어느 구간에 있는지 출력
 2) 어깨-팔꿈치-손목 등 팔 관절을 하나하나 조인트 단위로 추적 (좌표, 각도, 이동 궤적)
 3) 손바닥이 경계선을 넘나든 횟수(구간 전환 빈도수)를 좌/우 손 각각 카운트

hand_dectection.py와 동일한 Jetson CSI 카메라(nvarguscamerasrc) 구성으로 바로 실행됩니다.
    python upper_body_tracking.py   (q 키로 종료)
"""

from collections import deque  # 최근 N개만 남기는 고정 길이 큐 (이동 궤적 저장용)

import cv2                                   # 영상 입출력 및 화면 그리기
import numpy as np                           # 벡터 / 각도 계산
import mediapipe as mp                       # MediaPipe 본체 (mp.Image 생성용)
from mediapipe.tasks import python           # BaseOptions (모델 경로 지정)
from mediapipe.tasks.python import vision    # PoseLandmarker (자세 추정 모델)


# ---------------------------------------------------------------- 설정 값

MODEL_PATH = "src/models/MediaPipe/pose_landmarker_full.task"  # Pose Landmarker 모델 경로

VISIBILITY_THRESHOLD = 0.5    # landmark를 신뢰할 최소 visibility (이 값 미만이면 가려진 것으로 간주)
HEAD_TOP_RATIO = 0.35         # 머리 꼭대기 추정 계수 (귀 높이에서 어깨너비 * 계수 만큼 위)
BOUNDARY_MARGIN = 0.02        # 경계 떨림 방지용 여유 폭 (정규화 좌표, hysteresis)
STABLE_FRAMES = 3             # 새 구간으로 인정하기 위해 연속으로 유지되어야 하는 프레임 수
SMOOTHING_ALPHA = 0.4         # 경계선 좌표 EMA 스무딩 계수 (클수록 반응 빠름 / 작을수록 부드러움)
TRAIL_LENGTH = 32             # 조인트 이동 궤적으로 남길 프레임 수

# Pose landmark index (MediaPipe Pose는 사람 한 명당 33개의 landmark를 반환)
NOSE = 0                                    # 코
LEFT_EAR, RIGHT_EAR = 7, 8                  # 왼쪽 / 오른쪽 귀
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12      # 왼쪽 / 오른쪽 어깨
LEFT_ELBOW, RIGHT_ELBOW = 13, 14            # 왼쪽 / 오른쪽 팔꿈치
LEFT_WRIST, RIGHT_WRIST = 15, 16            # 왼쪽 / 오른쪽 손목
LEFT_PINKY, RIGHT_PINKY = 17, 18            # 왼쪽 / 오른쪽 새끼손가락
LEFT_INDEX, RIGHT_INDEX = 19, 20            # 왼쪽 / 오른쪽 검지
LEFT_THUMB, RIGHT_THUMB = 21, 22            # 왼쪽 / 오른쪽 엄지
LEFT_HIP, RIGHT_HIP = 23, 24                # 왼쪽 / 오른쪽 골반 (어깨 각도 계산 기준점)

# 상반신만 그리기 위한 연결선 (33개 landmark 중 상반신에 해당하는 것만 이음)
UPPER_BODY_CONNECTIONS = (
    (NOSE, LEFT_EAR), (NOSE, RIGHT_EAR),                                            # 얼굴
    (LEFT_SHOULDER, RIGHT_SHOULDER),                                                # 어깨선
    (LEFT_SHOULDER, LEFT_ELBOW), (LEFT_ELBOW, LEFT_WRIST),                          # 왼팔
    (RIGHT_SHOULDER, RIGHT_ELBOW), (RIGHT_ELBOW, RIGHT_WRIST),                      # 오른팔
    (LEFT_WRIST, LEFT_PINKY), (LEFT_WRIST, LEFT_INDEX), (LEFT_WRIST, LEFT_THUMB),   # 왼손
    (LEFT_PINKY, LEFT_INDEX),                                                       # 왼손바닥
    (RIGHT_WRIST, RIGHT_PINKY), (RIGHT_WRIST, RIGHT_INDEX), (RIGHT_WRIST, RIGHT_THUMB),  # 오른손
    (RIGHT_PINKY, RIGHT_INDEX),                                                     # 오른손바닥
    (LEFT_SHOULDER, LEFT_HIP), (RIGHT_SHOULDER, RIGHT_HIP), (LEFT_HIP, RIGHT_HIP),  # 몸통
)

# 하나하나 추적할 팔 관절 (조인트 이름, landmark index) - 좌/우 팔 각각 3개 조인트
ARM_JOINTS = {
    "Left": (("SHOULDER", LEFT_SHOULDER), ("ELBOW", LEFT_ELBOW), ("WRIST", LEFT_WRIST)),
    "Right": (("SHOULDER", RIGHT_SHOULDER), ("ELBOW", RIGHT_ELBOW), ("WRIST", RIGHT_WRIST)),
}

# 손바닥 중심을 계산할 landmark (손목 / 새끼손가락 / 검지의 평균 = 손바닥 한가운데)
PALM_POINTS = {
    "Left": (LEFT_WRIST, LEFT_PINKY, LEFT_INDEX),
    "Right": (RIGHT_WRIST, RIGHT_PINKY, RIGHT_INDEX),
}

# 위에서 아래 순서의 구간 이름 (영상 좌표는 y가 커질수록 화면 아래쪽)
ZONE_NAMES = ("ABOVE HEAD", "HEAD~EAR", "EAR~SHOULDER", "BELOW SHOULDER")
ZONE_COLORS = ((0, 0, 255), (0, 128, 255), (0, 255, 255), (0, 255, 0))  # 구간별 표시 색 (BGR)

# 구간을 나누는 경계선 이름 (경계 i는 ZONE i 와 ZONE i+1 사이 = 통과 횟수를 세는 대상)
BOUNDARY_NAMES = ("HEAD-TOP", "EAR", "SHOULDER")


# ---------------------------------------------------------------- 유틸 함수

def calculate_angle(p1, p2, p3):
    """p2를 중심으로 p1-p2-p3가 이루는 각도(도) 계산"""
    # 중심점 p2에서 양쪽 점으로 향하는 두 벡터를 만든다
    vector1 = np.array([p1.x - p2.x, p1.y - p2.y, p1.z - p2.z])
    vector2 = np.array([p3.x - p2.x, p3.y - p2.y, p3.z - p2.z])

    # 각 벡터의 길이 (크기)
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)

    if magnitude1 == 0 or magnitude2 == 0:  # 두 점이 겹치면 각도를 정의할 수 없음
        return 0.0

    # 코사인 법칙: cos = (v1 · v2) / (|v1| * |v2|)
    cosine = np.dot(vector1, vector2) / (magnitude1 * magnitude2)
    cosine = np.clip(cosine, -1.0, 1.0)  # 부동소수 오차로 범위를 벗어나면 arccos에서 에러 발생

    return float(np.degrees(np.arccos(cosine)))  # 라디안 -> 도(degree)로 변환


def is_visible(landmark):
    """해당 landmark가 충분히 잘 보이는지 여부 (가려짐 / 화면 밖 판단)"""
    return getattr(landmark, "visibility", 1.0) >= VISIBILITY_THRESHOLD


def mean_y(pose, indices):
    """보이는 landmark들의 평균 y (정규화 좌표). 하나도 안 보이면 None"""
    values = [pose[i].y for i in indices if is_visible(pose[i])]  # 보이는 landmark만 추림
    return sum(values) / len(values) if values else None          # 좌우 평균 = 몸이 기울어도 안정적


def palm_center(pose, side):
    """손목 / 새끼 / 검지 landmark의 평균으로 손바닥 중심 좌표(정규화) 계산"""
    indices = [i for i in PALM_POINTS[side] if is_visible(pose[i])]  # 보이는 landmark만 사용

    if not indices:  # 손이 전혀 안 보이면 구간 판별 불가
        return None

    x = sum(pose[i].x for i in indices) / len(indices)  # 손바닥 중심 x
    y = sum(pose[i].y for i in indices) / len(indices)  # 손바닥 중심 y
    return x, y


def zone_index(y, boundaries):
    """손바닥 y좌표가 어느 구간에 속하는지 index로 반환 (0: 머리 위 ~ 3: 어깨 아래)"""
    for i, boundary in enumerate(boundaries):  # boundaries는 위에서 아래 순서
        if y < boundary:                       # 경계보다 위에 있으면 그 구간에 속함
            return i
    return len(boundaries)                     # 모든 경계보다 아래 = 마지막 구간 (어깨 아래)


def smooth(previous, current):
    """경계선 좌표 EMA(지수이동평균) 스무딩 - 경계선이 프레임마다 튀는 것을 완화"""
    if previous is None:  # 첫 프레임은 섞을 이전 값이 없음
        return current
    return previous * (1 - SMOOTHING_ALPHA) + current * SMOOTHING_ALPHA


# ---------------------------------------------------------------- 상태 클래스

class PalmZoneTracker:
    """한쪽 손바닥의 구간(zone) 상태와 경계 통과 횟수를 관리하는 클래스"""

    def __init__(self, side):
        self.side = side            # "Left" 또는 "Right"
        self.zone = None            # 현재 확정된 구간 index
        self.candidate = None       # 새로 진입 중인 후보 구간
        self.candidate_count = 0    # 후보 구간이 연속으로 유지된 프레임 수
        self.transitions = 0        # 총 구간 전환 횟수 (빈도수)
        self.boundary_counts = [0] * len(BOUNDARY_NAMES)  # 경계별 통과 횟수
        self.trail = deque(maxlen=TRAIL_LENGTH)           # 손바닥 이동 궤적 (최근 N프레임)

    def update(self, new_zone):
        """구간 후보를 갱신하고, 전환이 확정되면 (이전, 현재, 통과한 경계들) 반환"""
        if new_zone == self.zone:  # 구간이 그대로면 후보를 비우고 종료
            self.candidate, self.candidate_count = None, 0
            return None

        if new_zone == self.candidate:  # 직전과 같은 후보면 유지 프레임 수 증가
            self.candidate_count += 1
        else:                           # 다른 구간이면 후보를 새로 설정
            self.candidate, self.candidate_count = new_zone, 1

        # 같은 구간이 STABLE_FRAMES 이상 유지되어야 실제 전환으로 인정 (노이즈로 인한 오카운트 방지)
        if self.candidate_count < STABLE_FRAMES:
            return None

        previous, self.zone = self.zone, new_zone       # 이전 구간을 기억하고 현재 구간 갱신
        self.candidate, self.candidate_count = None, 0  # 후보 초기화

        if previous is None:  # 맨 처음 인식된 구간은 "전환"으로 세지 않음
            return None

        self.transitions += 1  # 전환 횟수 1 증가

        # 두 구간 사이에 놓인 모든 경계를 통과한 것으로 카운트
        # (예: 어깨 아래 -> 머리 위로 한 번에 이동하면 경계 3개를 모두 통과한 것)
        crossed = []
        for b in range(min(previous, new_zone), max(previous, new_zone)):
            self.boundary_counts[b] += 1
            crossed.append(BOUNDARY_NAMES[b])

        return previous, new_zone, crossed

    def round_trips(self):
        """경계별 왕복 횟수 (올라갔다 내려오면 2번 통과 = 1왕복)"""
        return [count // 2 for count in self.boundary_counts]


# ---------------------------------------------------------------- 화면 그리기

def draw_skeleton(frame, points, visible):
    """상반신 skeleton(선)과 관절(점) 그리기"""
    # landmark를 연결하는 선 (양 끝점이 모두 잘 보일 때만 그림)
    for start, end in UPPER_BODY_CONNECTIONS:
        if visible[start] and visible[end]:
            cv2.line(frame, points[start], points[end], (0, 255, 0), 2)

    arm_indices = {idx for joints in ARM_JOINTS.values() for _, idx in joints}  # 팔 관절 index 집합

    for i, point in enumerate(points):
        if not visible[i] or i > RIGHT_HIP:  # 안 보이거나 하반신 landmark면 건너뜀
            continue
        # 팔 관절은 빨간 큰 점(+흰 테두리), 나머지 상반신 landmark는 파란 작은 점
        if i in arm_indices:
            cv2.circle(frame, point, 7, (0, 0, 255), -1)
            cv2.circle(frame, point, 9, (255, 255, 255), 1)
        else:
            cv2.circle(frame, point, 4, (255, 0, 0), -1)


def draw_boundaries(frame, boundaries_px, w):
    """머리 / 귀 / 어깨 경계선을 화면에 가로선으로 표시"""
    for name, y in zip(BOUNDARY_NAMES, boundaries_px):
        cv2.line(frame, (0, y), (w, y), (200, 200, 200), 1, cv2.LINE_AA)  # 화면을 가로지르는 경계선
        cv2.putText(frame, name, (w - 170, y - 8),                        # 경계선 이름 (우측에 표기)
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)


def draw_trail(frame, trail, color):
    """이동 궤적 그리기 (최근 위치일수록 굵게 = 움직인 방향이 보이도록)"""
    for i in range(1, len(trail)):
        thickness = max(1, int(i / len(trail) * 5))  # 오래된 점일수록 얇게
        cv2.line(frame, trail[i - 1], trail[i], color, thickness)


def draw_panel(frame, info_lines):
    """좌측 상단 반투명 정보 패널 (구간 / 카운트 / 각도 텍스트)"""
    if not info_lines:  # 표시할 내용이 없으면 원본 그대로 반환
        return frame

    h = frame.shape[0]                                     # 프레임 높이
    overlay = frame.copy()                                 # 반투명 합성을 위한 복사본
    panel_bottom = min(24 + 22 * len(info_lines), h - 10)  # 줄 수에 맞춰 패널 높이 결정
    cv2.rectangle(overlay, (10, 10), (490, panel_bottom), (0, 0, 0), -1)  # 검은 배경 박스
    frame = cv2.addWeighted(overlay, 0.45, frame, 0.55, 0)  # 원본과 섞어 반투명 처리

    y = 34
    for text, color in info_lines:  # 한 줄씩 아래로 내려가며 출력
        cv2.putText(frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
        y += 22

    return frame


# ---------------------------------------------------------------- 초기화

base_option = python.BaseOptions(model_asset_path=MODEL_PATH)                  # 모델 경로 지정하는 옵션
options = vision.PoseLandmarkerOptions(base_options=base_option, num_poses=1)  # 모델 경로와 최대 인원 수 지정
pose_detector = vision.PoseLandmarker.create_from_options(options)             # 해당 옵션으로 자세 검출 객체 생성

# Jetson CSI 카메라 입력 파이프라인 (hand_dectection.py와 동일)
pipeline = (
    "nvarguscamerasrc sensor-id=0 ! "                                      # CSI 카메라 소스
    "video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1 ! "  # GPU 메모리상의 영상 포맷
    "nvvidconv ! "                                                         # 하드웨어 가속 변환
    "video/x-raw, format=BGRx ! "                                          # CPU 메모리로 내리며 BGRx로 변환
    "videoconvert ! "                                                      # 색공간 변환
    "video/x-raw, format=BGR ! "                                           # OpenCV가 사용하는 BGR 포맷
    "queue leaky=downstream max-size-buffers=1 ! "                         # 밀리면 오래된 프레임 버림 (지연 방지)
    "appsink drop=true max-buffers=1 sync=false"                           # 항상 최신 프레임만 가져옴
)

cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)  # GStreamer 백엔드로 카메라 열기

trackers = {side: PalmZoneTracker(side) for side in ("Left", "Right")}  # 좌/우 손바닥 추적기
joint_trails = {side: {name: deque(maxlen=TRAIL_LENGTH) for name, _ in joints}
                for side, joints in ARM_JOINTS.items()}                 # 팔 조인트별 이동 궤적
boundaries = None  # 스무딩된 경계선 y (정규화 좌표, 위에서 아래 순서로 3개)


# ---------------------------------------------------------------- 메인 루프

while True:
    ret, frame = cap.read()  # 카메라에서 프레임 한 장 읽기

    if not ret:                            # 프레임을 못 읽으면 종료
        break
    if cv2.waitKey(1) & 0xFF == ord("q"):  # q 키를 누르면 종료
        break

    # 이미지 좌우 반전 및 RGB로 색공간 변환 (전처리)
    frame = cv2.flip(frame, 0)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    h, w = frame.shape[:2]  # 프레임 높이와 너비

    # 프레임 내 사람(상반신) 탐지
    result = pose_detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

    info_lines = []  # 이번 프레임에 패널로 출력할 (텍스트, 색) 목록
    pose = result.pose_landmarks[0] if result.pose_landmarks else None  # 첫 번째 사람의 landmark

    if pose is not None:
        points = [(int(p.x * w), int(p.y * h)) for p in pose]  # 프레임 크기 기준 각 landmark 좌표
        visible = [is_visible(p) for p in pose]                # landmark별 신뢰 여부

        draw_skeleton(frame, points, visible)  # 상반신 뼈대 그리기

        # ---- 머리 / 귀 / 어깨 기준 경계선 계산
        ear_y = mean_y(pose, (LEFT_EAR, RIGHT_EAR))                 # 귀 높이 (좌우 평균)
        shoulder_y = mean_y(pose, (LEFT_SHOULDER, RIGHT_SHOULDER))  # 어깨 높이 (좌우 평균)

        if ear_y is None and visible[NOSE]:
            ear_y = pose[NOSE].y  # 귀가 안 보이면 코 높이로 대체

        if ear_y is not None and shoulder_y is not None:
            # 머리 꼭대기는 landmark가 없으므로 어깨 너비를 기준 길이로 삼아 추정
            if visible[LEFT_SHOULDER] and visible[RIGHT_SHOULDER]:
                shoulder_width = abs(pose[LEFT_SHOULDER].x - pose[RIGHT_SHOULDER].x)
            else:
                shoulder_width = abs(shoulder_y - ear_y) * 2  # 한쪽 어깨만 보이면 귀~어깨 거리로 대체
            head_top_y = ear_y - HEAD_TOP_RATIO * shoulder_width  # 귀보다 위쪽이므로 y가 더 작아짐

            current = np.sort(np.array([head_top_y, ear_y, shoulder_y]))  # 위에서 아래 순서 보장
            boundaries = smooth(boundaries, current)                      # 이전 경계와 섞어 부드럽게
    else:
        info_lines.append(("No person detected", (0, 0, 255)))  # 사람이 안 잡힌 경우 안내

    if boundaries is not None:
        draw_boundaries(frame, [int(y * h) for y in boundaries], w)  # 정규화 좌표 -> 픽셀 변환 후 표시

    # ---- 손바닥이 어느 경계 사이에 있는지 판별 + 전환 횟수 카운트
    if pose is not None and boundaries is not None:
        for side, tracker in trackers.items():  # 왼손 / 오른손 각각 처리
            center = palm_center(pose, side)    # 손바닥 중심 좌표 (정규화)

            if center is None:                  # 손이 안 보이면 이번 프레임은 건너뜀
                info_lines.append((f"{side} palm : not visible", (128, 128, 128)))
                continue

            px, py = center
            point = (int(px * w), int(py * h))  # 화면에 그릴 픽셀 좌표
            tracker.trail.append(point)         # 궤적에 현재 위치 추가

            # 현재 구간 판별 (경계 근처에서는 hysteresis로 이전 구간을 유지해 깜빡임 방지)
            zone = zone_index(py, boundaries)
            if tracker.zone is not None and zone != tracker.zone:
                edge = min(zone, tracker.zone)                    # 두 구간 사이의 경계 index
                if abs(py - boundaries[edge]) < BOUNDARY_MARGIN:  # 경계에서 여유 폭 안쪽이면
                    zone = tracker.zone                           # 아직 넘지 않은 것으로 취급

            event = tracker.update(zone)  # 구간 갱신 -> 전환이 확정되면 이벤트 반환
            if event is not None:
                previous, now, crossed = event  # 이전 구간 / 현재 구간 / 통과한 경계 목록
                counts = ", ".join(f"{n}:{c}" for n, c in zip(BOUNDARY_NAMES, tracker.boundary_counts))
                print(f"[{side}] {ZONE_NAMES[previous]} -> {ZONE_NAMES[now]}"  # 터미널에 전환 로그 출력
                      f" | crossed: {', '.join(crossed)}"
                      f" | transitions: {tracker.transitions} | counts: {counts}")

            shown = tracker.zone if tracker.zone is not None else zone  # 표시할 구간 (확정값 우선)
            color = ZONE_COLORS[shown]                                  # 구간별 색상

            draw_trail(frame, tracker.trail, color)                     # 손바닥 이동 궤적
            cv2.circle(frame, point, 10, color, -1)                     # 손바닥 중심 점
            cv2.circle(frame, point, 12, (255, 255, 255), 2)            # 흰색 테두리
            cv2.putText(frame, f"{side} palm", (point[0] + 14, point[1] - 10),  # 좌/우 손 라벨
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # 패널 출력용 텍스트: 현재 구간 / 경계별 통과 횟수(cross) / 경계별 왕복 횟수(trip)
            info_lines.append((f"{side} palm : {ZONE_NAMES[shown]}", color))
            info_lines.append(("  cross " + " ".join(
                f"{n}:{c}" for n, c in zip(BOUNDARY_NAMES, tracker.boundary_counts)
            ) + f"  (total {tracker.transitions})", (255, 255, 255)))
            info_lines.append(("  trip  " + " ".join(
                f"{n}:{r}" for n, r in zip(BOUNDARY_NAMES, tracker.round_trips())
            ), (180, 180, 180)))

        # ---- 팔 관절 하나하나 조인트 단위 추적 (궤적 + 각도)
        for side, joints in ARM_JOINTS.items():
            for name, idx in joints:           # 어깨 -> 팔꿈치 -> 손목 순서로 조인트별 처리
                if not is_visible(pose[idx]):  # 안 보이는 조인트는 건너뜀
                    continue
                p = (int(pose[idx].x * w), int(pose[idx].y * h))            # 조인트 픽셀 좌표
                joint_trails[side][name].append(p)                          # 조인트별 궤적에 기록
                draw_trail(frame, joint_trails[side][name], (255, 128, 0))  # 조인트 이동 궤적
                cv2.putText(frame, f"{side[0]}-{name}", (p[0] + 10, p[1] + 18),  # 예: "L-ELBOW"
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 0), 1)

            shoulder, elbow, wrist = (idx for _, idx in joints)  # 각도 계산에 사용할 3개 index
            hip = LEFT_HIP if side == "Left" else RIGHT_HIP      # 어깨 각도의 기준이 되는 골반

            elbow_angle = shoulder_angle = None  # 각도를 못 구한 경우를 대비한 초기값

            # 팔꿈치 각도: 어깨-팔꿈치-손목 (팔을 얼마나 굽혔는지)
            if all(is_visible(pose[i]) for i in (shoulder, elbow, wrist)):
                elbow_angle = calculate_angle(pose[shoulder], pose[elbow], pose[wrist])
                cv2.putText(frame, f"{elbow_angle:.0f}",  # 팔꿈치 옆에 각도 표시
                            (int(pose[elbow].x * w) + 10, int(pose[elbow].y * h) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # 어깨 각도: 골반-어깨-팔꿈치 (팔을 얼마나 들어올렸는지)
            if all(is_visible(pose[i]) for i in (hip, shoulder, elbow)):
                shoulder_angle = calculate_angle(pose[hip], pose[shoulder], pose[elbow])
                cv2.putText(frame, f"{shoulder_angle:.0f}",  # 어깨 옆에 각도 표시
                            (int(pose[shoulder].x * w) + 10, int(pose[shoulder].y * h) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # 각도를 못 구한 경우 "--"로 표기해 패널에 추가
            elbow_text = f"{elbow_angle:.0f}" if elbow_angle is not None else "--"
            shoulder_text = f"{shoulder_angle:.0f}" if shoulder_angle is not None else "--"
            info_lines.append((f"{side} arm  : shoulder {shoulder_text} / elbow {elbow_text}",
                               (255, 200, 0)))

    frame = draw_panel(frame, info_lines)          # 좌측 상단 정보 패널 합성
    cv2.imshow("MediaPipe Pose Detection", frame)  # 결과 화면 출력


# ---------------------------------------------------------------- 종료 처리

# 루프가 끝나면 좌/우 손의 최종 통계를 터미널에 정리해서 출력
print("\n===== 최종 결과 =====")
for side, tracker in trackers.items():
    last = ZONE_NAMES[tracker.zone] if tracker.zone is not None else "-"  # 마지막으로 머문 구간
    print(f"[{side}] 마지막 구간: {last} / 총 구간 전환 횟수: {tracker.transitions}")
    for name, count, trip in zip(BOUNDARY_NAMES, tracker.boundary_counts, tracker.round_trips()):
        print(f"    {name:<10} 통과 {count}회 (왕복 {trip}회)")

pose_detector.close()    # 모델 리소스 해제
cap.release()            # 카메라 해제
cv2.destroyAllWindows()  # 열려 있는 OpenCV 창 모두 닫기

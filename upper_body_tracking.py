"""상반신 인식 + 손바닥 경계 판별 + 팔 관절 추적 + 경계 왕복 횟수 카운트

MediaPipe Pose Landmarker로 상반신(머리/귀/어깨/팔/손)을 추적하면서
 1) 손바닥이 [머리 위 / 머리~귀 / 귀~어깨 / 어깨 아래] 중 어느 구간에 있는지 출력
 2) 어깨-팔꿈치-손목 등 팔 관절을 하나하나 조인트 단위로 추적 (좌표, 각도, 이동 궤적)
 3) 손바닥이 경계선을 넘나든 횟수(구간 전환 빈도수)를 좌/우 손 각각 카운트

hand_dectection.py와 동일한 Jetson CSI 카메라(nvarguscamerasrc) 구성으로 바로 실행됩니다.
    python upper_body_tracking.py   (q 키로 종료)
"""

from collections import deque

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ---------------------------------------------------------------- 설정 값

MODEL_PATH = "src/models/MediaPipe/pose_landmarker_full.task"  # Pose Landmarker 모델 경로

VISIBILITY_THRESHOLD = 0.5    # landmark를 신뢰할 최소 visibility
HEAD_TOP_RATIO = 0.35         # 머리 꼭대기 추정 계수 (귀 높이에서 어깨너비 * 계수 만큼 위)
BOUNDARY_MARGIN = 0.02        # 경계 떨림 방지용 여유 폭 (정규화 좌표, hysteresis)
STABLE_FRAMES = 3             # 새 구간으로 인정하기 위해 연속으로 유지되어야 하는 프레임 수
SMOOTHING_ALPHA = 0.4         # 경계선 좌표 EMA 스무딩 계수 (클수록 반응 빠름)
TRAIL_LENGTH = 32             # 조인트 이동 궤적으로 남길 프레임 수

# Pose landmark index (MediaPipe Pose 33 landmarks)
NOSE = 0
LEFT_EAR, RIGHT_EAR = 7, 8
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_PINKY, RIGHT_PINKY = 17, 18
LEFT_INDEX, RIGHT_INDEX = 19, 20
LEFT_THUMB, RIGHT_THUMB = 21, 22
LEFT_HIP, RIGHT_HIP = 23, 24

# 상반신만 그리기 위한 연결선 (얼굴 + 어깨 + 팔 + 몸통)
UPPER_BODY_CONNECTIONS = (
    (NOSE, LEFT_EAR), (NOSE, RIGHT_EAR),
    (LEFT_SHOULDER, RIGHT_SHOULDER),
    (LEFT_SHOULDER, LEFT_ELBOW), (LEFT_ELBOW, LEFT_WRIST),
    (RIGHT_SHOULDER, RIGHT_ELBOW), (RIGHT_ELBOW, RIGHT_WRIST),
    (LEFT_WRIST, LEFT_PINKY), (LEFT_WRIST, LEFT_INDEX), (LEFT_WRIST, LEFT_THUMB),
    (LEFT_PINKY, LEFT_INDEX),
    (RIGHT_WRIST, RIGHT_PINKY), (RIGHT_WRIST, RIGHT_INDEX), (RIGHT_WRIST, RIGHT_THUMB),
    (RIGHT_PINKY, RIGHT_INDEX),
    (LEFT_SHOULDER, LEFT_HIP), (RIGHT_SHOULDER, RIGHT_HIP), (LEFT_HIP, RIGHT_HIP),
)

# 하나하나 추적할 팔 관절 (조인트 이름, landmark index)
ARM_JOINTS = {
    "Left": (("SHOULDER", LEFT_SHOULDER), ("ELBOW", LEFT_ELBOW), ("WRIST", LEFT_WRIST)),
    "Right": (("SHOULDER", RIGHT_SHOULDER), ("ELBOW", RIGHT_ELBOW), ("WRIST", RIGHT_WRIST)),
}

# 손바닥 중심을 계산할 landmark (손목 / 새끼손가락 / 검지)
PALM_POINTS = {
    "Left": (LEFT_WRIST, LEFT_PINKY, LEFT_INDEX),
    "Right": (RIGHT_WRIST, RIGHT_PINKY, RIGHT_INDEX),
}

# 위에서 아래 순서의 구간 이름 (y가 커질수록 화면 아래)
ZONE_NAMES = ("ABOVE HEAD", "HEAD~EAR", "EAR~SHOULDER", "BELOW SHOULDER")
ZONE_COLORS = ((0, 0, 255), (0, 128, 255), (0, 255, 255), (0, 255, 0))

# 구간을 나누는 경계선 이름 (경계 i는 ZONE i 와 ZONE i+1 사이)
BOUNDARY_NAMES = ("HEAD-TOP", "EAR", "SHOULDER")


# ---------------------------------------------------------------- 유틸 함수

def calculate_angle(p1, p2, p3):
    """p2를 중심으로 p1-p2-p3가 이루는 각도(도) 계산"""
    vector1 = np.array([p1.x - p2.x, p1.y - p2.y, p1.z - p2.z])
    vector2 = np.array([p3.x - p2.x, p3.y - p2.y, p3.z - p2.z])

    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)

    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    cosine = np.dot(vector1, vector2) / (magnitude1 * magnitude2)
    cosine = np.clip(cosine, -1.0, 1.0)

    return float(np.degrees(np.arccos(cosine)))


def is_visible(landmark):
    """해당 landmark가 충분히 잘 보이는지 여부"""
    return getattr(landmark, "visibility", 1.0) >= VISIBILITY_THRESHOLD


def mean_y(pose, indices):
    """보이는 landmark들의 평균 y (정규화 좌표). 하나도 안 보이면 None"""
    values = [pose[i].y for i in indices if is_visible(pose[i])]
    return sum(values) / len(values) if values else None


def palm_center(pose, side):
    """손목/새끼/검지 landmark의 평균으로 손바닥 중심 좌표(정규화) 계산"""
    indices = [i for i in PALM_POINTS[side] if is_visible(pose[i])]

    if not indices:
        return None

    x = sum(pose[i].x for i in indices) / len(indices)
    y = sum(pose[i].y for i in indices) / len(indices)
    return x, y


def zone_index(y, boundaries):
    """손바닥 y좌표가 어느 구간에 속하는지 index로 반환 (0: 머리 위 ~ 3: 어깨 아래)"""
    for i, boundary in enumerate(boundaries):
        if y < boundary:
            return i
    return len(boundaries)


def smooth(previous, current):
    """경계선 좌표 EMA 스무딩"""
    if previous is None:
        return current
    return previous * (1 - SMOOTHING_ALPHA) + current * SMOOTHING_ALPHA


# ---------------------------------------------------------------- 상태 클래스

class PalmZoneTracker:
    """한쪽 손바닥의 구간(zone) 상태와 경계 통과 횟수를 관리"""

    def __init__(self, side):
        self.side = side
        self.zone = None            # 현재 확정된 구간 index
        self.candidate = None       # 새로 진입 중인 후보 구간
        self.candidate_count = 0    # 후보 구간이 유지된 프레임 수
        self.transitions = 0        # 총 구간 전환 횟수
        self.boundary_counts = [0] * len(BOUNDARY_NAMES)  # 경계별 통과 횟수
        self.trail = deque(maxlen=TRAIL_LENGTH)           # 손바닥 이동 궤적

    def update(self, new_zone):
        """구간 후보를 갱신하고, 전환이 확정되면 (이전, 현재, 통과한 경계들) 반환"""
        if new_zone == self.zone:
            self.candidate, self.candidate_count = None, 0
            return None

        if new_zone == self.candidate:
            self.candidate_count += 1
        else:
            self.candidate, self.candidate_count = new_zone, 1

        # 같은 구간이 STABLE_FRAMES 이상 유지되어야 실제 전환으로 인정 (떨림 방지)
        if self.candidate_count < STABLE_FRAMES:
            return None

        previous, self.zone = self.zone, new_zone
        self.candidate, self.candidate_count = None, 0

        if previous is None:  # 첫 인식은 전환으로 세지 않음
            return None

        self.transitions += 1

        # 두 구간 사이에 놓인 모든 경계를 통과한 것으로 카운트
        crossed = []
        for b in range(min(previous, new_zone), max(previous, new_zone)):
            self.boundary_counts[b] += 1
            crossed.append(BOUNDARY_NAMES[b])

        return previous, new_zone, crossed

    def round_trips(self):
        """경계별 왕복 횟수 (2번 통과 = 1왕복)"""
        return [count // 2 for count in self.boundary_counts]


# ---------------------------------------------------------------- 화면 그리기

def draw_skeleton(frame, points, visible):
    """상반신 skeleton(선)과 관절(점) 그리기"""
    for start, end in UPPER_BODY_CONNECTIONS:
        if visible[start] and visible[end]:
            cv2.line(frame, points[start], points[end], (0, 255, 0), 2)

    arm_indices = {idx for joints in ARM_JOINTS.values() for _, idx in joints}
    for i, point in enumerate(points):
        if not visible[i] or i > RIGHT_HIP:
            continue
        # 팔 관절은 빨간 큰 점, 나머지 상반신 landmark는 파란 작은 점
        if i in arm_indices:
            cv2.circle(frame, point, 7, (0, 0, 255), -1)
            cv2.circle(frame, point, 9, (255, 255, 255), 1)
        else:
            cv2.circle(frame, point, 4, (255, 0, 0), -1)


def draw_boundaries(frame, boundaries_px, w):
    """머리/귀/어깨 경계선을 화면에 가로선으로 표시"""
    for name, y in zip(BOUNDARY_NAMES, boundaries_px):
        cv2.line(frame, (0, y), (w, y), (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, name, (w - 170, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)


def draw_trail(frame, trail, color):
    """이동 궤적 그리기 (오래된 점일수록 얇게)"""
    for i in range(1, len(trail)):
        thickness = max(1, int(i / len(trail) * 5))
        cv2.line(frame, trail[i - 1], trail[i], color, thickness)


def draw_panel(frame, info_lines):
    """좌측 상단 반투명 정보 패널"""
    if not info_lines:
        return frame

    h = frame.shape[0]
    overlay = frame.copy()
    panel_bottom = min(24 + 22 * len(info_lines), h - 10)
    cv2.rectangle(overlay, (10, 10), (490, panel_bottom), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.45, frame, 0.55, 0)

    y = 34
    for text, color in info_lines:
        cv2.putText(frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
        y += 22

    return frame


# ---------------------------------------------------------------- 초기화

# 모델 경로를 지정해 상반신(pose) 검출 객체 생성
base_option = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.PoseLandmarkerOptions(base_options=base_option, num_poses=1)
pose_detector = vision.PoseLandmarker.create_from_options(options)

pipeline = (
    "nvarguscamerasrc sensor-id=0 ! "
    "video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1 ! "
    "nvvidconv ! "
    "video/x-raw, format=BGRx ! "
    "videoconvert ! "
    "video/x-raw, format=BGR ! "
    "queue leaky=downstream max-size-buffers=1 ! "
    "appsink drop=true max-buffers=1 sync=false"
)

cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

trackers = {side: PalmZoneTracker(side) for side in ("Left", "Right")}
joint_trails = {side: {name: deque(maxlen=TRAIL_LENGTH) for name, _ in joints}
                for side, joints in ARM_JOINTS.items()}
boundaries = None  # 스무딩된 경계선 y (정규화 좌표, 위→아래 순서)


# ---------------------------------------------------------------- 메인 루프

while True:
    ret, frame = cap.read()

    if not ret:
        break
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    # 이미지 좌우 반전 및 RGB로 색공간 변환 (전처리)
    frame = cv2.flip(frame, 0)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    h, w = frame.shape[:2]  # 프레임 높이와 너비

    # 프레임 내 사람(상반신) 탐지
    result = pose_detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

    info_lines = []
    pose = result.pose_landmarks[0] if result.pose_landmarks else None

    if pose is not None:
        points = [(int(p.x * w), int(p.y * h)) for p in pose]  # 프레임 크기 기준 각 landmark 좌표
        visible = [is_visible(p) for p in pose]

        draw_skeleton(frame, points, visible)

        # ---- 머리 / 귀 / 어깨 기준 경계선 계산
        ear_y = mean_y(pose, (LEFT_EAR, RIGHT_EAR))
        shoulder_y = mean_y(pose, (LEFT_SHOULDER, RIGHT_SHOULDER))

        if ear_y is None and visible[NOSE]:
            ear_y = pose[NOSE].y  # 귀가 안 보이면 코 높이로 대체

        if ear_y is not None and shoulder_y is not None:
            # 어깨 너비를 기준 길이로 삼아 머리 꼭대기 위치를 추정
            if visible[LEFT_SHOULDER] and visible[RIGHT_SHOULDER]:
                shoulder_width = abs(pose[LEFT_SHOULDER].x - pose[RIGHT_SHOULDER].x)
            else:
                shoulder_width = abs(shoulder_y - ear_y) * 2
            head_top_y = ear_y - HEAD_TOP_RATIO * shoulder_width

            current = np.sort(np.array([head_top_y, ear_y, shoulder_y]))  # 위→아래 순서 보장
            boundaries = smooth(boundaries, current)
    else:
        info_lines.append(("No person detected", (0, 0, 255)))

    if boundaries is not None:
        draw_boundaries(frame, [int(y * h) for y in boundaries], w)

    # ---- 손바닥이 어느 경계 사이에 있는지 판별 + 전환 횟수 카운트
    if pose is not None and boundaries is not None:
        for side, tracker in trackers.items():
            center = palm_center(pose, side)

            if center is None:
                info_lines.append((f"{side} palm : not visible", (128, 128, 128)))
                continue

            px, py = center
            point = (int(px * w), int(py * h))
            tracker.trail.append(point)

            # 현재 구간 판별 (경계 근처에서는 hysteresis로 이전 구간 유지)
            zone = zone_index(py, boundaries)
            if tracker.zone is not None and zone != tracker.zone:
                edge = min(zone, tracker.zone)
                if abs(py - boundaries[edge]) < BOUNDARY_MARGIN:
                    zone = tracker.zone

            event = tracker.update(zone)
            if event is not None:
                previous, now, crossed = event
                counts = ", ".join(f"{n}:{c}" for n, c in zip(BOUNDARY_NAMES, tracker.boundary_counts))
                print(f"[{side}] {ZONE_NAMES[previous]} -> {ZONE_NAMES[now]}"
                      f" | crossed: {', '.join(crossed)}"
                      f" | transitions: {tracker.transitions} | counts: {counts}")

            shown = tracker.zone if tracker.zone is not None else zone
            color = ZONE_COLORS[shown]

            draw_trail(frame, tracker.trail, color)
            cv2.circle(frame, point, 10, color, -1)
            cv2.circle(frame, point, 12, (255, 255, 255), 2)
            cv2.putText(frame, f"{side} palm", (point[0] + 14, point[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            info_lines.append((f"{side} palm : {ZONE_NAMES[shown]}", color))
            info_lines.append(("  cross " + " ".join(
                f"{n}:{c}" for n, c in zip(BOUNDARY_NAMES, tracker.boundary_counts)
            ) + f"  (total {tracker.transitions})", (255, 255, 255)))
            info_lines.append(("  trip  " + " ".join(
                f"{n}:{r}" for n, r in zip(BOUNDARY_NAMES, tracker.round_trips())
            ), (180, 180, 180)))

        # ---- 팔 관절 하나하나 조인트 단위 추적 (궤적 + 각도)
        for side, joints in ARM_JOINTS.items():
            for name, idx in joints:
                if not is_visible(pose[idx]):
                    continue
                p = (int(pose[idx].x * w), int(pose[idx].y * h))
                joint_trails[side][name].append(p)
                draw_trail(frame, joint_trails[side][name], (255, 128, 0))
                cv2.putText(frame, f"{side[0]}-{name}", (p[0] + 10, p[1] + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 0), 1)

            shoulder, elbow, wrist = (idx for _, idx in joints)
            hip = LEFT_HIP if side == "Left" else RIGHT_HIP

            elbow_angle = shoulder_angle = None

            if all(is_visible(pose[i]) for i in (shoulder, elbow, wrist)):
                elbow_angle = calculate_angle(pose[shoulder], pose[elbow], pose[wrist])
                cv2.putText(frame, f"{elbow_angle:.0f}",
                            (int(pose[elbow].x * w) + 10, int(pose[elbow].y * h) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            if all(is_visible(pose[i]) for i in (hip, shoulder, elbow)):
                shoulder_angle = calculate_angle(pose[hip], pose[shoulder], pose[elbow])
                cv2.putText(frame, f"{shoulder_angle:.0f}",
                            (int(pose[shoulder].x * w) + 10, int(pose[shoulder].y * h) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            elbow_text = f"{elbow_angle:.0f}" if elbow_angle is not None else "--"
            shoulder_text = f"{shoulder_angle:.0f}" if shoulder_angle is not None else "--"
            info_lines.append((f"{side} arm  : shoulder {shoulder_text} / elbow {elbow_text}",
                               (255, 200, 0)))

    frame = draw_panel(frame, info_lines)
    cv2.imshow("MediaPipe Pose Detection", frame)


# ---------------------------------------------------------------- 종료 처리

print("\n===== 최종 결과 =====")
for side, tracker in trackers.items():
    last = ZONE_NAMES[tracker.zone] if tracker.zone is not None else "-"
    print(f"[{side}] 마지막 구간: {last} / 총 구간 전환 횟수: {tracker.transitions}")
    for name, count, trip in zip(BOUNDARY_NAMES, tracker.boundary_counts, tracker.round_trips()):
        print(f"    {name:<10} 통과 {count}회 (왕복 {trip}회)")

pose_detector.close()
cap.release()
cv2.destroyAllWindows()

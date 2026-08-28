"""디비디비딥 가위바위보 승자 판정 프로그램 (실시간 모드)

MediaPipe Pose Landmarker로 두 사람의 상반신 관절(어깨/손목)을 매 프레임 뽑아,
어깨의 수직선(좌우)/수평선(위아래) 기준으로 손 모양을 실시간으로 분류하고 승자를 화면에 표시한다.
[두 손목이 모두 어깨보다 바깥 -> 보 / 두 손목이 모두 어깨보다 아래 -> 바위 / 한쪽만 위 -> 가위]

YOLO 탐지 대기, TTS 안내("디비 디비 딥"), 캡쳐 단계는 Jetson에서 화면이 멈춘 것처럼
느려지는 원인이 되어 뺐다. 튜닝 때 문제없이 잘 동작했던 실시간 루프를 그대로 사용한다.

    python rps_ddobidibidip.py   (q 를 누르면 종료)
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ---------------------------------------------------------------- 설정 값

POSE_MODEL_PATH = "src/models/MediaPipe/pose_landmarker_full.task"

VISIBILITY_THRESHOLD = 0.5  # pose landmark 신뢰 최소 visibility

# Pose landmark index (MediaPipe Pose 33 landmarks)
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_WRIST, RIGHT_WRIST = 15, 16

# 가위바위보 승패 규칙: key가 value를 이긴다
BEATS = {"바위": "가위", "가위": "보", "보": "바위"}

# cv2.putText는 한글 폰트를 지원하지 않아 화면 표시용으로만 영어 라벨을 사용한다 (로그는 한글 그대로)
GESTURE_EN = {"바위": "Rock", "가위": "Scissors", "보": "Paper", None: "?"}

WINDOW_NAME = "Ddobidibidip Rock Paper Scissors"

CAMERA_PIPELINE = (
    "nvarguscamerasrc sensor-id=0 ! "
    "video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1 ! "
    "nvvidconv ! "
    "video/x-raw, format=BGRx ! "
    "videoconvert ! "
    "video/x-raw, format=BGR ! "
    "queue leaky=downstream max-size-buffers=1 ! "
    "appsink drop=true max-buffers=1 sync=false"
)


# ---------------------------------------------------------------- 손 모양 (가위/바위/보) 분류

def is_visible(landmark):
    return getattr(landmark, "visibility", 1.0) >= VISIBILITY_THRESHOLD


def shoulder_center_x(pose):
    """왼쪽/오른쪽 사람 정렬용: 두 어깨의 평균 x좌표"""
    return (pose[LEFT_SHOULDER].x + pose[RIGHT_SHOULDER].x) / 2


def shoulder_line_y(pose):
    """어깨 라인 y좌표 (정규화 좌표, 두 어깨의 평균). 이미지 좌표계라 위로 갈수록 값이 작아짐"""
    return (pose[LEFT_SHOULDER].y + pose[RIGHT_SHOULDER].y) / 2


def is_extended_outward(wrist, shoulder, center_x):
    """손목이 몸 중심 기준으로 같은 쪽 어깨보다 바깥쪽으로 더 나가 있는지"""
    shoulder_offset = shoulder.x - center_x
    wrist_offset = wrist.x - center_x
    return (wrist_offset * shoulder_offset > 0) and (abs(wrist_offset) > abs(shoulder_offset))


def classify_gesture(pose):
    """어깨 수직선(좌우)과 수평선(위아래)을 기준으로 손목 위치로 손 모양을 분류

    - 두 손목이 모두 자기 쪽 어깨 수직선보다 바깥에 있으면 -> 보 (최우선)
    - (보가 아니면서) 두 손목이 모두 어깨 수평선보다 아래 -> 바위
    - (보가 아니면서) 한쪽 손목만 어깨 수평선보다 위 -> 가위
    - 그 외(둘 다 위지만 바깥으로 벌어지지 않음)는 판별 불가(None)
    """
    required = (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_WRIST, RIGHT_WRIST)
    if not all(is_visible(pose[i]) for i in required):
        return None

    center_x = shoulder_center_x(pose)
    left_extended = is_extended_outward(pose[LEFT_WRIST], pose[LEFT_SHOULDER], center_x)
    right_extended = is_extended_outward(pose[RIGHT_WRIST], pose[RIGHT_SHOULDER], center_x)

    if left_extended and right_extended:
        return "보"

    shoulder_y = shoulder_line_y(pose)
    up_count = (pose[LEFT_WRIST].y < shoulder_y) + (pose[RIGHT_WRIST].y < shoulder_y)

    if up_count == 0:
        return "바위"
    if up_count == 1:
        return "가위"
    return None


# ---------------------------------------------------------------- 화면 표시

def draw_shoulder_debug(frame, pose, w, h):
    """어깨 기준선(가로: 위/아래, 세로: 좌우 바깥)과 손목 위치를 그려 분류 기준을 시각화

    손목 색: 자주색 = 어깨보다 바깥(보 우선 조건), 노랑 = 어깨보다 위, 주황 = 어깨보다 아래
    """
    shoulder_y_px = int(shoulder_line_y(pose) * h)
    cv2.line(frame, (0, shoulder_y_px), (w, shoulder_y_px), (0, 200, 200), 1, cv2.LINE_AA)

    for idx in (LEFT_SHOULDER, RIGHT_SHOULDER):
        x = int(pose[idx].x * w)
        cv2.line(frame, (x, 0), (x, h), (0, 200, 200), 1, cv2.LINE_AA)

    center_x = shoulder_center_x(pose)
    shoulder_y = shoulder_line_y(pose)

    for wrist_idx, shoulder_idx in ((LEFT_WRIST, LEFT_SHOULDER), (RIGHT_WRIST, RIGHT_SHOULDER)):
        if not is_visible(pose[wrist_idx]):
            continue
        p = (int(pose[wrist_idx].x * w), int(pose[wrist_idx].y * h))
        if is_extended_outward(pose[wrist_idx], pose[shoulder_idx], center_x):
            color = (255, 0, 255)
        elif pose[wrist_idx].y < shoulder_y:
            color = (0, 255, 255)
        else:
            color = (255, 128, 0)
        cv2.circle(frame, p, 8, color, -1)


def winner_status(gestures):
    """두 손모양으로 상태 문자열과 표시 색을 결정"""
    if None in gestures:
        return "Reading gestures...", (0, 255, 255)

    left_gesture, right_gesture = gestures
    if left_gesture == right_gesture:
        return f"DRAW ({GESTURE_EN[left_gesture]})", (0, 255, 255)

    winner_idx = 0 if BEATS[left_gesture] == right_gesture else 1
    winner_side = "LEFT" if winner_idx == 0 else "RIGHT"
    return f"WINNER: {winner_side} ({GESTURE_EN[gestures[winner_idx]]})", (0, 255, 0)


# ---------------------------------------------------------------- 초기화

base_option = python.BaseOptions(model_asset_path=POSE_MODEL_PATH)
pose_options = vision.PoseLandmarkerOptions(base_options=base_option, num_poses=2)
pose_detector = vision.PoseLandmarker.create_from_options(pose_options)

cap = cv2.VideoCapture(CAMERA_PIPELINE, cv2.CAP_GSTREAMER)
if not cap.isOpened():
    raise RuntimeError("카메라를 열 수 없습니다.")


# ---------------------------------------------------------------- 메인 루프 (실시간)

try:
    print("실시간 가위바위보 승자 판정을 시작합니다. q 를 누르면 종료합니다.")
    last_status = None

    while True:
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("카메라 프레임을 읽을 수 없습니다.")

        frame = cv2.flip(frame, 0)
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_result = pose_detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        poses = sorted(pose_result.pose_landmarks, key=shoulder_center_x)[:2] if pose_result.pose_landmarks else []

        gestures = [classify_gesture(pose) for pose in poses]

        for i, pose in enumerate(poses):
            side = "LEFT" if i == 0 else "RIGHT"
            gesture = gestures[i]
            color = (0, 255, 0) if gesture else (0, 0, 255)

            draw_shoulder_debug(frame, pose, w, h)

            cx = int(shoulder_center_x(pose) * w)
            cy = int(min(pose[LEFT_SHOULDER].y, pose[RIGHT_SHOULDER].y) * h) - 20
            cv2.putText(frame, f"{side}: {GESTURE_EN[gesture]}", (max(10, cx - 70), max(30, cy)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        if len(poses) < 2:
            status, color = "Waiting for 2 players...", (0, 255, 255)
        else:
            status, color = winner_status(gestures)

        if status != last_status:
            print(status)
            last_status = status

        cv2.putText(frame, status, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.imshow(WINDOW_NAME, frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    pose_detector.close()
    cap.release()
    cv2.destroyAllWindows()

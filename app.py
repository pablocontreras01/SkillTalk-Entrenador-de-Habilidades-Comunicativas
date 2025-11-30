import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import tempfile
from typing import List, Dict, Tuple, Optional

# ================== CONFIGURACIÓN ==================
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

CHUNK_SIZE = 30
CLASS_NAMES = ["Beat", "No-Gesture"]
COLORS = {"Beat": (0, 255, 0), "No-Gesture": (255, 0, 0)}

# Puntos del esqueleto
SPINE_BASE = 0; SPINE_MID = 1; NECK = 2; HEAD = 3
SHOULDER_LEFT = 4; ELBOW_LEFT = 5; WRIST_LEFT = 6; HAND_LEFT = 7
SHOULDER_RIGHT = 8; ELBOW_RIGHT = 9; WRIST_RIGHT = 10; HAND_RIGHT = 11
HIP_LEFT = 12; KNEE_LEFT = 13; ANKLE_LEFT = 14; FOOT_LEFT = 15
HIP_RIGHT = 16; KNEE_RIGHT = 17; ANKLE_RIGHT = 18; FOOT_RIGHT = 19
SPINE_SHOULDER = 20; HANDTIP_LEFT = 21; THUMB_LEFT = 22
HANDTIP_RIGHT = 23; THUMB_RIGHT = 24

# ================== FUNCIONES ==================
def normalize_skeleton_sequence(seq: np.ndarray) -> np.ndarray:
    seq = seq.copy().astype(np.float32)
    seq[np.isnan(seq)] = 0.0
    root = seq[:, SPINE_BASE:SPINE_BASE+1, :]
    seq = seq - root
    left_shoulder = seq[:, SHOULDER_LEFT, :]
    right_shoulder = seq[:, SHOULDER_RIGHT, :]
    shoulder_vec = np.mean(left_shoulder - right_shoulder, axis=0)
    shoulder_vec[1] = 0
    norm = np.linalg.norm(shoulder_vec)
    shoulder_vec = shoulder_vec / norm if norm > 1e-6 else np.array([1.0, 0.0, 0.0])
    target = np.array([1.0, 0.0, 0.0])
    v = np.cross(shoulder_vec, target)
    c = np.dot(shoulder_vec, target)
    if np.linalg.norm(v) < 1e-6:
        R = np.eye(3)
    else:
        vx = np.array([[0, -v[2], v[1]],[v[2], 0, -v[0]],[-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * (1 / (1 + c))
    seq = seq @ R.T
    shoulder_dist = np.mean(np.linalg.norm(seq[:, SHOULDER_LEFT, :] - seq[:, SHOULDER_RIGHT, :], axis=1))
    scale = 1.0 / shoulder_dist if shoulder_dist > 1e-6 else 1.0
    seq = seq * scale
    return seq

def create_chunks_from_skeletons(skeletons: List[np.ndarray], chunk_size: int) -> np.ndarray:
    if len(skeletons) == 0:
        return np.zeros((0, chunk_size, 25, 3), dtype=np.float32)
    sk_arr = np.stack(skeletons, axis=0)
    T = sk_arr.shape[0]
    chunks = []
    for start in range(0, T, chunk_size):
        end = start + chunk_size
        chunk = sk_arr[start:end]
        if chunk.shape[0] < chunk_size:
            last = chunk[-1] if chunk.shape[0] > 0 else np.zeros((25,3), dtype=np.float32)
            pad = np.repeat(last[None, :, :], chunk_size - chunk.shape[0], axis=0)
            chunk = np.concatenate([chunk, pad], axis=0)
        chunks.append(chunk)
    return np.stack(chunks, axis=0).astype(np.float32)

def prepare_chunks_for_model(chunks_4d: np.ndarray) -> np.ndarray:
    N, chunk_len, J, C = chunks_4d.shape
    return chunks_4d.reshape(N, chunk_len, J * C)

def extract_kinect25_from_mediapipe(landmarks) -> np.ndarray:
    def to_np(idx):
        lm = landmarks[idx]
        return np.array([lm.x, lm.y, lm.z], dtype=np.float32)
    k = np.zeros((25,3), dtype=np.float32)
    k[0] = (to_np(mp_pose.PoseLandmark.LEFT_HIP) + to_np(mp_pose.PoseLandmark.RIGHT_HIP))/2
    k[1] = (k[0] + (to_np(mp_pose.PoseLandmark.LEFT_SHOULDER)+to_np(mp_pose.PoseLandmark.RIGHT_SHOULDER))/2)/2
    k[2] = (to_np(mp_pose.PoseLandmark.LEFT_SHOULDER)+to_np(mp_pose.PoseLandmark.RIGHT_SHOULDER))/2
    k[3] = to_np(mp_pose.PoseLandmark.NOSE)
    k[4] = to_np(mp_pose.PoseLandmark.LEFT_SHOULDER)
    k[5] = to_np(mp_pose.PoseLandmark.LEFT_ELBOW)
    k[6] = to_np(mp_pose.PoseLandmark.LEFT_WRIST)
    k[7] = to_np(mp_pose.PoseLandmark.LEFT_INDEX)
    k[8] = to_np(mp_pose.PoseLandmark.RIGHT_SHOULDER)
    k[9] = to_np(mp_pose.PoseLandmark.RIGHT_ELBOW)
    k[10] = to_np(mp_pose.PoseLandmark.RIGHT_WRIST)
    k[11] = to_np(mp_pose.PoseLandmark.RIGHT_INDEX)
    k[12] = to_np(mp_pose.PoseLandmark.LEFT_HIP)
    k[13] = to_np(mp_pose.PoseLandmark.LEFT_KNEE)
    k[14] = to_np(mp_pose.PoseLandmark.LEFT_ANKLE)
    k[15] = to_np(mp_pose.PoseLandmark.LEFT_FOOT_INDEX)
    k[16] = to_np(mp_pose.PoseLandmark.RIGHT_HIP)
    k[17] = to_np(mp_pose.PoseLandmark.RIGHT_KNEE)
    k[18] = to_np(mp_pose.PoseLandmark.RIGHT_ANKLE)
    k[19] = to_np(mp_pose.PoseLandmark.RIGHT_FOOT_INDEX)
    k[20] = k[2]
    k[21] = to_np(mp_pose.PoseLandmark.LEFT_INDEX)
    k[22] = to_np(mp_pose.PoseLandmark.LEFT_THUMB)
    k[23] = to_np(mp_pose.PoseLandmark.RIGHT_INDEX)
    k[24] = to_np(mp_pose.PoseLandmark.RIGHT_THUMB)
    return k

def process_video_to_kinect25_with_visuals(video_path: str) -> List[Dict]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {video_path}")
    frame_data = []
    last_valid_k25 = None
    with mp_pose.Pose(static_image_mode=False, model_complexity=1,
                      enable_segmentation=False, min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:
        while True:
            ret, frame_bgr = cap.read()
            if not ret: break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            res = pose.process(frame_rgb)
            current_k25 = None
            if res.pose_landmarks:
                try:
                    current_k25 = extract_kinect25_from_mediapipe(res.pose_landmarks.landmark)
                    last_valid_k25 = current_k25
                except Exception:
                    if last_valid_k25 is not None:
                        current_k25 = last_valid_k25.copy()
                    else:
                        current_k25 = np.zeros((25,3), dtype=np.float32)
            else:
                current_k25 = last_valid_k25.copy() if last_valid_k25 is not None else np.zeros((25,3), dtype=np.float32)
            frame_data.append({'frame': frame_bgr, 'k25': current_k25, 'pose_landmarks': res.pose_landmarks})
    cap.release()
    return frame_data

def draw_skeleton_and_label(image: np.ndarray, pose_landmarks, label: str, color: Tuple) -> np.ndarray:
    if pose_landmarks:
        mp_drawing.draw_landmarks(image, pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  landmark_drawing_spec=mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2),
                                  connection_drawing_spec=mp_drawing.DrawingSpec(color=color, thickness=2))
    cv2.putText(image, f"CLASE: {label}", (10,30), cv2.FONT_HERSHEY_SIMPLEX,1,color,2,cv2.LINE_AA)
    return image

@st.cache_resource
def load_ml_model(model_path):
    abs_path = os.path.join(os.path.dirname(__file__), model_path)
    if not os.path.exists(abs_path):
        st.error(f"❌ Modelo no encontrado en {abs_path}")
        return None
    try:
        model = load_model(abs_path)
        st.success("✅ Modelo cargado correctamente")
        return model
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {e}")
        return None

def run_classification_pipeline(video_path: str, model: tf.keras.Model):
    frame_data_list = process_video_to_kinect25_with_visuals(video_path)
    skeletons = [item['k25'] for item in frame_data_list]
    if len(skeletons) == 0:
        st.warning("⚠️ No se extrajeron esqueletos del video.")
        return None
    chunks_4d = create_chunks_from_skeletons(skeletons, CHUNK_SIZE)
    normalized_chunks = np.stack([normalize_skeleton_sequence(seq) for seq in chunks_4d], axis=0)
    X = prepare_chunks_for_model(normalized_chunks)
    preds = model.predict(X, verbose=0)
    pred_inds = preds.argmax(axis=1)
    visual_frames = []
    T = len(frame_data_list)
    for i in range(preds.shape[0]):
        chunk_start_idx = i*CHUNK_SIZE
        chunk_end_idx = min((i+1)*CHUNK_SIZE, T)
        predicted_label = CLASS_NAMES[pred_inds[i]]
        color = COLORS.get(predicted_label, (255,255,255))
        for j in range(chunk_start_idx, chunk_end_idx):
            if j < T:
                data = frame_data_list[j]
                frame = data['frame'].copy()
                pose_landmarks = data['pose_landmarks']
                visual_frame = draw_skeleton_and_label(frame, pose_landmarks, predicted_label, color)
                visual_frames.append(visual_frame)
    H,W,_ = visual_frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.release()
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_output:
        output_path = temp_output.name
    out = cv2.VideoWriter(output_path, fourcc, fps, (W,H))
    for frame in visual_frames:
        out.write(frame)
    out.release()
    return output_path

def main():
    st.set_page_config(page_title="Clasificador de Gestos SkillTalk", layout="wide")
    st.title("🗣️ Clasificador de Gestos de Presentación")
    st.markdown("Analiza la actividad gestual **Beat** vs. **No-Gesture**:")
    MODEL_PATH = "mlp_lstm_ted.keras"
    model = load_ml_model(MODEL_PATH)
    if model is None: return
    uploaded_file = st.file_uploader("Sube un video (MP4 recomendado)", type=['mp4','mov','avi'])
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_file.read())
            video_path = tmp.name
        col_orig, col_result = st.columns(2)
        with col_orig:
            st.subheader("Video Original")
            st.video(video_path)
        with col_result:
            if st.button("▶️ Ejecutar Clasificación"):
                st.info("Procesando… Esto puede tardar varios minutos.")
                output_video_path = run_classification_pipeline(video_path, model)
                if output_video_path:
                    st.success("✅ Clasificación completada")
                    st.video(output_video_path)
                    try: os.unlink(output_video_path)
                    except: pass
        try: os.unlink(video_path)
        except: pass

if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()


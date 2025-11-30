import streamlit as st
import os
import tempfile
import requests
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from typing import List

# =========================================
# CONFIGURACIÓN
# =========================================
CHUNK_SIZE = 30
CLASS_NAMES = ["Beat", "No-Gesture"]
COLORS = {"Beat": (0, 255, 0), "No-Gesture": (255, 0, 0)}

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Kinect25 joints
SPINE_BASE = 0; SPINE_MID = 1; NECK = 2; HEAD = 3
SHOULDER_LEFT = 4; ELBOW_LEFT = 5; WRIST_LEFT = 6; HAND_LEFT = 7
SHOULDER_RIGHT = 8; ELBOW_RIGHT = 9; WRIST_RIGHT = 10; HAND_RIGHT = 11
HIP_LEFT = 12; KNEE_LEFT = 13; ANKLE_LEFT = 14; FOOT_LEFT = 15
HIP_RIGHT = 16; KNEE_RIGHT = 17; ANKLE_RIGHT = 18; FOOT_RIGHT = 19
SPINE_SHOULDER = 20; HANDTIP_LEFT = 21; THUMB_LEFT = 22
HANDTIP_RIGHT = 23; THUMB_RIGHT = 24

N_FEATURES = 25 * 3

# =========================================
# PREPROCESAMIENTO
# =========================================
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
    shoulder_vec = shoulder_vec / norm if norm>1e-6 else np.array([1.,0.,0.])
    target = np.array([1.,0.,0.])
    v = np.cross(shoulder_vec, target)
    c = np.dot(shoulder_vec, target)
    if np.linalg.norm(v) < 1e-6:
        R = np.eye(3)
    else:
        vx = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
        R = np.eye(3) + vx + vx@vx*(1/(1+c))
    seq = seq @ R.T

    shoulder_dist = np.mean(np.linalg.norm(seq[:, SHOULDER_LEFT,:]-seq[:,SHOULDER_RIGHT,:], axis=1))
    scale = 1.0/shoulder_dist if shoulder_dist>1e-6 else 1.0
    seq *= scale
    return seq

def create_chunks_from_skeletons(skeletons: List[np.ndarray], chunk_size: int) -> np.ndarray:
    if len(skeletons) == 0:
        return np.zeros((0, chunk_size, 25,3), dtype=np.float32)
    sk_arr = np.stack(skeletons, axis=0)
    T = sk_arr.shape[0]
    chunks = []
    for start in range(0,T,chunk_size):
        end = start+chunk_size
        chunk = sk_arr[start:end]
        if chunk.shape[0]<chunk_size:
            last = chunk[-1] if chunk.shape[0]>0 else np.zeros((25,3), dtype=np.float32)
            pad = np.repeat(last[None,:,:], chunk_size-chunk.shape[0], axis=0)
            chunk = np.concatenate([chunk,pad], axis=0)
        chunks.append(chunk)
    return np.stack(chunks, axis=0).astype(np.float32)

def prepare_chunks_for_model(chunks_4d: np.ndarray) -> np.ndarray:
    N, chunk_len, J, C = chunks_4d.shape
    return chunks_4d.reshape(N, chunk_len, J*C)

def extract_skeletons_from_video(video_path: str) -> List[np.ndarray]:
    skeletons = []
    cap = cv2.VideoCapture(video_path)
    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                joints = np.zeros((25,3), dtype=np.float32)
                for idx, lm in enumerate(landmarks[:25]):
                    joints[idx] = [lm.x, lm.y, lm.z]
                skeletons.append(joints)
    cap.release()
    return skeletons

# =========================================
# MODELO `.keras` DESDE DRIVE
MODEL_FILE = "mlp_lstm_ted_final.keras"
DRIVE_FILE_ID = "1n9wuBQPbK_zW_PNbj2BFMGKD8NXGa-XC"
DRIVE_DOWNLOAD_URL = f"https://drive.google.com/uc?export=download&id={DRIVE_FILE_ID}"

@st.cache_resource
def load_my_model(model_file: str):
    if not os.path.exists(model_file):
        st.write("📥 Descargando modelo desde Google Drive...")
        response = requests.get(DRIVE_DOWNLOAD_URL)
        if response.status_code == 200:
            with open(model_file, "wb") as f:
                f.write(response.content)
            st.write("✅ Modelo descargado correctamente.")
        else:
            st.error(f"❌ Error descargando el modelo. Código HTTP: {response.status_code}")
            return None
    model = tf.keras.models.load_model(model_file)
    st.write("✅ Modelo cargado exitosamente.")
    return model

# =========================================
# PREDICCIÓN
# =========================================
def predict_gestures(model, skeletons: List[np.ndarray]):
    seq_norm = normalize_skeleton_sequence(np.stack(skeletons))
    chunks = create_chunks_from_skeletons([seq_norm], CHUNK_SIZE)
    X_model = prepare_chunks_for_model(chunks)
    predictions = model.predict(X_model)
    return predictions.argmax(axis=1)

# =========================================
# STREAMLIT APP
# =========================================
def main():
    st.set_page_config(page_title="Clasificador de Gestos SkillTalk", layout="wide")
    st.title("🗣️ Clasificador de Gestos")
    st.markdown("Sube un video y analiza gestos **Beat** vs **No-Gesture**")

    model = load_my_model(MODEL_FILE)
    if model is None:
        st.stop()

    uploaded_file = st.file_uploader("Sube un video (MP4/AVI/MOV)", type=['mp4','avi','mov'])
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name
        st.video(video_path)
        st.info("Procesando video y extrayendo skeletons...")
        skeletons = extract_skeletons_from_video(video_path)
        if skeletons:
            preds = predict_gestures(model, skeletons)
            st.write(f"Predicciones del video: {preds}")
        else:
            st.warning("No se detectaron skeletons en el video.")

if __name__ == "__main__":
    main()

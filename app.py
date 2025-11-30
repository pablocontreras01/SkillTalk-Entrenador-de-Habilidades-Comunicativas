import streamlit as st
import os
import tempfile
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import requests
from typing import List

# =========================================
# CONFIGURACIÓN
# =========================================
CHUNK_SIZE = 30
CLASS_NAMES = ["Beat", "No-Gesture"]
COLORS = {"Beat": "green", "No-Gesture": "red"}

mp_pose = mp.solutions.pose

SPINE_BASE = 0
SHOULDER_LEFT = 4
SHOULDER_RIGHT = 8
N_FEATURES = 25 * 3

# Modelo desde Google Drive
MODEL_FILE = "mlp_lstm_ted.keras"
DRIVE_URL = "https://drive.google.com/uc?export=download&id=1y1XSOebGIy_Nmr_FNHzX7mHVCvxhwHSL"

# =========================================
# NORMALIZACIÓN DE SKELETONS
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
    shoulder_dist = np.mean(np.linalg.norm(seq[:, SHOULDER_LEFT,:]-seq[:, SHOULDER_RIGHT,:], axis=1))
    scale = 1.0 / shoulder_dist if shoulder_dist>1e-6 else 1.0
    seq *= scale
    return seq

def create_chunks_from_skeletons(skeletons: List[np.ndarray], chunk_size: int) -> np.ndarray:
    if len(skeletons) == 0:
        return np.zeros((0, chunk_size, 25,3), dtype=np.float32)
    sk_arr = np.stack(skeletons, axis=0)
    T = sk_arr.shape[0]
    chunks = []
    for start in range(0, T, chunk_size):
        end = start + chunk_size
        chunk = sk_arr[start:end]
        if chunk.shape[0] < chunk_size:
            last = chunk[-1] if chunk.shape[0] > 0 else np.zeros((25,3), dtype=np.float32)
            pad = np.repeat(last[None,:,:], chunk_size-chunk.shape[0], axis=0)
            chunk = np.concatenate([chunk, pad], axis=0)
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
                sk = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark])
                sk = normalize_skeleton_sequence(sk[None,:,:,:])[0]
                skeletons.append(sk)
    cap.release()
    return skeletons

# =========================================
# CARGA DEL MODELO DESDE DRIVE
# =========================================
@st.cache_resource
def load_model(model_file: str):
    if not os.path.exists(model_file):
        st.info("📥 Descargando modelo desde Google Drive...")
        r = requests.get(DRIVE_URL, allow_redirects=True)
        with open(model_file, "wb") as f:
            f.write(r.content)
        st.success("✅ Modelo descargado.")
    model = tf.keras.models.load_model(model_file)
    return model

def predict_gestures(model, skeletons: List[np.ndarray]):
    chunks = create_chunks_from_skeletons(skeletons, CHUNK_SIZE)
    X_input = prepare_chunks_for_model(chunks)
    preds = model.predict(X_input)
    return preds

# =========================================
# STREAMLIT APP
# =========================================
def main():
    st.set_page_config(page_title="Clasificador de Gestos SkillTalk", layout="wide")
    st.title("🗣️ Clasificador de Gestos")
    st.markdown("Sube un video y analiza gestos **Beat** vs **No-Gesture**")
    
    model = load_model(MODEL_FILE)
    
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
            for i, p in enumerate(preds):
                cls_idx = np.argmax(p)
                cls_name = CLASS_NAMES[cls_idx]
                st.markdown(f"**Chunk {i+1}:** <span style='color:{COLORS[cls_name]}'>{cls_name}</span> - Probabilidad: {p[cls_idx]:.2f}", unsafe_allow_html=True)
        else:
            st.warning("No se detectaron skeletons en el video.")

if __name__ == "__main__":
    main()

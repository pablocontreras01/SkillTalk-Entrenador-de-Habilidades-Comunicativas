import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras import layers, models
import os
from typing import List
import tempfile
import requests

# ====================================================================
# CONFIGURACIÓN

CHUNK_SIZE = 30
CLASS_NAMES = ["Beat", "No-Gesture"]
COLORS = {"Beat": (0,255,0), "No-Gesture": (255,0,0)}

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

N_FEATURES = 25*3

# ====================================================================
# FUNCIONES DE NORMALIZACIÓN Y PREPROCESAMIENTO

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

# ====================================================================
# MODELO MLP-LSTM

def build_mlp_lstm(input_shape=(CHUNK_SIZE, N_FEATURES), num_classes=2):
    model = models.Sequential()
    model.add(layers.TimeDistributed(layers.Dense(128, activation='relu'), input_shape=input_shape))
    model.add(layers.TimeDistributed(layers.Dense(128, activation='relu')))
    model.add(layers.LSTM(128, return_sequences=False))
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model

@st.cache_resource
def load_model_with_weights(weights_path: str):
    # Descargar desde Google Drive si no existe
    if not os.path.exists(weights_path):
        st.write("Descargando pesos del modelo desde Google Drive...")
        url = "https://drive.google.com/uc?export=download&id=1BtLzHd7sD4r0BH4JLj7lMbMs-FcvO318"
        response = requests.get(url)
        with open(weights_path, "wb") as f:
            f.write(response.content)
        st.write("Pesos descargados correctamente.")

    model = build_mlp_lstm()
    model.load_weights(weights_path)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    st.write("Modelo cargado exitosamente.")
    return model

# ====================================================================
# STREAMLIT UI

def main():
    st.set_page_config(page_title="Clasificador de Gestos SkillTalk", layout="wide")
    st.title("🗣️ Clasificador de Gestos")
    st.markdown("Sube un video y analiza gestos **Beat** vs **No-Gesture**")

    MODEL_WEIGHTS_PATH = "mlp_lstm_ted_weights.h5"
    model = load_model_with_weights(MODEL_WEIGHTS_PATH)
    if model is None:
        return

    uploaded_file = st.file_uploader("Sube un video (MP4/AVI/MOV)", type=['mp4','avi','mov'])
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name
        st.video(video_path)
        st.info("Clasificación lista para ejecutarse (agrega tu pipeline aquí)")

if __name__ == "__main__":
    main()

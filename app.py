import streamlit as st
import os
import tempfile
import requests
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
# PREPROCESAMIENTO (placeholders)
# =========================================
def normalize_skeleton_sequence(seq: np.ndarray) -> np.ndarray:
    # Aquí va tu lógica de normalización
    return seq

def create_chunks_from_skeletons(skeletons: List[np.ndarray], chunk_size: int) -> np.ndarray:
    # Aquí va tu lógica de chunking
    return np.zeros((0, chunk_size, 25,3), dtype=np.float32)

def prepare_chunks_for_model(chunks_4d: np.ndarray) -> np.ndarray:
    N, chunk_len, J, C = chunks_4d.shape
    return chunks_4d.reshape(N, chunk_len, J * C)

# =========================================
# CARGA DEL MODELO `.keras` DESDE DRIVE
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
    # Cargar el modelo `.keras`
    model = tf.keras.models.load_model(model_file)
    st.write("✅ Modelo cargado exitosamente.")
    return model

# =========================================
# APP STREAMLIT
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
        st.info("Aquí puedes insertar tu pipeline de extracción de skeletons → predicción con el modelo.")

if __name__ == "__main__":
    main()

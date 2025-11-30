import os
from typing import List, Optional, Dict, Tuple
from collections import Counter
import tempfile # Para manejar archivos temporales de forma segura

# ====================================================================
# ⚙️ PARÁMETROS DE CONFIGURACIÓN (Extraídos de tu script original)


# ====================================================================

# Inicializar MediaPipe y Drawing Utils
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# 🛑 2. PARÁMETROS DEL MODELO Y PROCESAMIENTO 🛑
CHUNK_SIZE = 30 # Tamaño de la secuencia que espera tu modelo (L_MAX).
CLASS_NAMES = ["Beat", "No-Gesture"] # Clases en el orden de salida del modelo (Índice 0, 1)
COLORS = {
    "Beat": (0, 255, 0),    # Verde (Gesto activo) -> BGR
    "No-Gesture": (255, 0, 0) # Azul (No-Gesture) -> BGR
}

# 🛑 3. CONSTANTES DEL ESQUELETO (Kinect v2) 🛑

SPINE_BASE = 0; SPINE_MID = 1; NECK = 2; HEAD = 3
SHOULDER_LEFT = 4; ELBOW_LEFT = 5; WRIST_LEFT = 6; HAND_LEFT = 7
SHOULDER_RIGHT = 8; ELBOW_RIGHT = 9; WRIST_RIGHT = 10; HAND_RIGHT = 11
@@ -34,350 +35,272 @@
SPINE_SHOULDER = 20; HANDTIP_LEFT = 21; THUMB_LEFT = 22
HANDTIP_RIGHT = 23; THUMB_RIGHT = 24



# --------------------------------------------------------------------
# 📌 FUNCIONES DE PREPROCESAMIENTO Y EXTRACCIÓN (Tu lógica original)
# --------------------------------------------------------------------

## 📏 FUNCIÓN DE NORMALIZACIÓN GEOMÉTRICA (IDÉNTICA AL ENTRENAMIENTO)
def normalize_skeleton_sequence(seq: np.ndarray) -> np.ndarray:
    """Normaliza una secuencia completa de esqueletos (T, 25, 3)."""
    seq = seq.copy().astype(np.float32)
    seq[np.isnan(seq)] = 0.0

    # 1. Centrar en pelvis (SPINE_BASE)
    root = seq[:, SPINE_BASE:SPINE_BASE+1, :]
    seq = seq - root

    # 2. Rotación para alinear hombros con eje X
    left_shoulder = seq[:, SHOULDER_LEFT, :]
    right_shoulder = seq[:, SHOULDER_RIGHT, :]
    shoulder_vec = np.mean(left_shoulder - right_shoulder, axis=0)
    shoulder_vec[1] = 0

    norm = np.linalg.norm(shoulder_vec)
    if norm < 1e-6:
        shoulder_vec = np.array([1.0, 0.0, 0.0])
    else:
        shoulder_vec = shoulder_vec / norm

    target = np.array([1.0, 0.0, 0.0])
    v = np.cross(shoulder_vec, target)
    c = np.dot(shoulder_vec, target)

    if np.linalg.norm(v) < 1e-6:
        R = np.eye(3)
    else:
        vx = np.array([[0, -v[2], v[1]],[v[2], 0, -v[0]],[-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * (1 / (1 + c))
    seq = seq @ R.T

    # 3. Escalar por distancia entre hombros
    shoulder_dist = np.mean(np.linalg.norm(seq[:, SHOULDER_LEFT, :] - seq[:, SHOULDER_RIGHT, :], axis=1))
    scale = 1.0 / shoulder_dist if shoulder_dist > 1e-6 else 1.0
    seq = seq * scale

    # 4. Normalización de longitud de huesos (por generalización)
    bones = [(SPINE_BASE, SPINE_MID), (SPINE_MID, SPINE_SHOULDER), (SPINE_SHOULDER, NECK), (NECK, HEAD),
             (SPINE_SHOULDER, SHOULDER_LEFT), (SHOULDER_LEFT, ELBOW_LEFT), (ELBOW_LEFT, WRIST_LEFT), (WRIST_LEFT, HAND_LEFT),
             (SPINE_SHOULDER, SHOULDER_RIGHT), (SHOULDER_RIGHT, ELBOW_RIGHT), (ELBOW_RIGHT, WRIST_RIGHT), (WRIST_RIGHT, HAND_RIGHT),
             (SPINE_BASE, HIP_LEFT), (HIP_LEFT, KNEE_LEFT), (KNEE_LEFT, ANKLE_LEFT), (ANKLE_LEFT, FOOT_LEFT),
             (SPINE_BASE, HIP_RIGHT), (HIP_RIGHT, KNEE_RIGHT), (KNEE_RIGHT, ANKLE_RIGHT), (ANKLE_RIGHT, FOOT_RIGHT)]

    for j1, j2 in bones:
        vec = seq[:, j2] - seq[:, j1]
        avg_len = np.mean(np.linalg.norm(vec, axis=1))
        if avg_len > 1e-6:
            seq[:, j2] = seq[:, j1] + vec / avg_len

    return seq

## 📐 MAPEO DE LANDMARKS (MEDIAPIPE → KINECT25)
def compute_spine_points(landmarks):
    """Calcula puntos sintéticos de la columna."""
    def to_np(idx):
        lm = landmarks[idx]
        return np.array([lm.x, lm.y, lm.z], dtype=np.float32)

    left_hip = to_np(mp_pose.PoseLandmark.LEFT_HIP)
    right_hip = to_np(mp_pose.PoseLandmark.RIGHT_HIP)
    left_sh = to_np(mp_pose.PoseLandmark.LEFT_SHOULDER)
    right_sh = to_np(mp_pose.PoseLandmark.RIGHT_SHOULDER)

    spine_base = (left_hip + right_hip) / 2.0
    spine_shoulder = (left_sh + right_sh) / 2.0
    spine_mid = (spine_base + spine_shoulder) / 2.0


































    return spine_base, spine_mid, spine_shoulder












def extract_kinect25_from_mediapipe(landmarks) -> np.ndarray:
    """Construye un esqueleto Kinect25 (25,3) desde landmarks de MediaPipe."""
    def L(idx):
        lm = landmarks[idx]
        return np.array([lm.x, lm.y, lm.z], dtype=np.float32)













    spine_base, spine_mid, spine_shoulder = compute_spine_points(landmarks)

    k = np.zeros((25, 3), dtype=np.float32)

    k[0] = spine_base; k[1] = spine_mid; k[2] = spine_shoulder; k[3] = L(mp_pose.PoseLandmark.NOSE)
    k[4] = L(mp_pose.PoseLandmark.LEFT_SHOULDER); k[5] = L(mp_pose.PoseLandmark.LEFT_ELBOW); k[6] = L(mp_pose.PoseLandmark.LEFT_WRIST)
    k[7] = L(mp_pose.PoseLandmark.LEFT_INDEX)
    k[8] = L(mp_pose.PoseLandmark.RIGHT_SHOULDER); k[9] = L(mp_pose.PoseLandmark.RIGHT_ELBOW); k[10] = L(mp_pose.PoseLandmark.RIGHT_WRIST)
    k[11] = L(mp_pose.PoseLandmark.RIGHT_INDEX)
    k[12] = L(mp_pose.PoseLandmark.LEFT_HIP); k[13] = L(mp_pose.PoseLandmark.LEFT_KNEE); k[14] = L(mp_pose.PoseLandmark.LEFT_ANKLE); k[15] = L(mp_pose.PoseLandmark.LEFT_FOOT_INDEX)
    k[16] = L(mp_pose.PoseLandmark.RIGHT_HIP); k[17] = L(mp_pose.PoseLandmark.RIGHT_KNEE); k[18] = L(mp_pose.PoseLandmark.RIGHT_ANKLE); k[19] = L(mp_pose.PoseLandmark.RIGHT_FOOT_INDEX)
    k[20] = spine_shoulder
    k[21] = L(mp_pose.PoseLandmark.LEFT_INDEX); k[22] = L(mp_pose.PoseLandmark.LEFT_THUMB); k[23] = L(mp_pose.PoseLandmark.RIGHT_INDEX); k[24] = L(mp_pose.PoseLandmark.RIGHT_THUMB)

    return k

## 📦 CHUNKING Y PREPARACIÓN DE ENTRADAS
def create_chunks_from_skeletons(skeletons: List[np.ndarray], chunk_size: int) -> np.ndarray:
    """Divide la secuencia de esqueletos en chunks y aplica padding por repetición."""
    if len(skeletons) == 0:
        return np.zeros((0, chunk_size, 25, 3), dtype=np.float32)

    sk_arr = np.stack(skeletons, axis=0)
    T = sk_arr.shape[0]

    chunks = []
    for start in range(0, T, chunk_size):
        end = start + chunk_size
        chunk = sk_arr[start:end]
        if chunk.shape[0] < chunk_size:
            # Padding: repetir el último frame válido
            last = chunk[-1] if chunk.shape[0] > 0 else np.zeros((25,3), dtype=np.float32)
            pad = np.repeat(last[None, :, :], chunk_size - chunk.shape[0], axis=0)
            chunk = np.concatenate([chunk, pad], axis=0)
        chunks.append(chunk)

    return np.stack(chunks, axis=0).astype(np.float32)

def prepare_chunks_for_model(chunks_4d: np.ndarray) -> np.ndarray:
    """Input: (N, chunk_size, 25, 3) -> Output: (N, chunk_size, 75)"""
    N, chunk_len, J, C = chunks_4d.shape
    return chunks_4d.reshape(N, chunk_len, J * C)






## 💾 PROCESAMIENTO DE VIDEO Y EXTRACCIÓN
@st.cache_data(show_spinner=False)
def process_video_to_kinect25_with_visuals(video_path: str, repeat_last_valid: bool = True) -> List[Dict]:
    """
    Lee el video, extrae esqueletos K25 y guarda el frame BGR y los pose_landmarks para visualización.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {video_path}")
































    # Es importante inicializar MediaPipe dentro de una función de caché si la usas,
    # o crear una nueva instancia por ejecución. Usaremos un contexto 'with' aquí.
    with mp_pose.Pose(static_image_mode=False, model_complexity=1,
                        enable_segmentation=False, min_detection_confidence=0.5,
                        min_tracking_confidence=0.5) as pose:

        frame_data = []
        last_valid_k25 = None

        while True:
            ret, frame_bgr = cap.read()
            if not ret: break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            res = pose.process(frame_rgb)













            current_pose_landmarks = res.pose_landmarks if res.pose_landmarks else None
            current_k25 = None

            if res.pose_landmarks:
                try:
                    current_k25 = extract_kinect25_from_mediapipe(res.pose_landmarks.landmark)
                    last_valid_k25 = current_k25
                except Exception:
                    if last_valid_k25 is not None and repeat_last_valid:
                        current_k25 = last_valid_k25.copy()
                    else:
                        current_k25 = np.zeros((25,3), dtype=np.float32)
            else:
                if last_valid_k25 is not None and repeat_last_valid:
                    current_k25 = last_valid_k25.copy()
                else:
                    current_k25 = np.zeros((25,3), dtype=np.float32)

            frame_data.append({
                'frame': frame_bgr,
                'k25': current_k25,
                'pose_landmarks': current_pose_landmarks
            })

        cap.release()
        return frame_data

## 🎨 DIBUJO Y ETIQUETADO POR CLASE
def draw_skeleton_and_label(image: np.ndarray, pose_landmarks, label: str, color: Tuple) -> np.ndarray:
    """Dibuja el esqueleto de MediaPipe y la etiqueta de clasificación."""
    if pose_landmarks:
        mp_drawing.draw_landmarks(
            image,
            pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2)
        )

    text = f"CLASE: {label}"
    cv2.putText(image, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
    return image


# --------------------------------------------------------------------
# 🎬 FUNCIONES CENTRALES DE STREAMLIT Y EJECUCIÓN
# --------------------------------------------------------------------

@st.cache_resource # Cargar el modelo solo una vez
def load_ml_model(model_path):
    """Carga el modelo y lo guarda en caché."""
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo desde **{model_path}**. Asegúrate de que el archivo exista en la misma carpeta.")
        return None

def run_classification_pipeline(video_path: str, model: tf.keras.Model,
                                chunk_size: int, class_names: List[str], colors: Dict) -> Optional[str]:
    """
    Ejecuta el pipeline completo de clasificación y genera el video de salida.
    """
    # 1. Extracción y recolección de datos
    with st.spinner('Paso 1/3: Extrayendo esqueletos del video... (Esto puede tardar)'):
        try:
            frame_data_list = process_video_to_kinect25_with_visuals(video_path)
        except RuntimeError as e:
            st.error(f"❌ Error al procesar el video: {e}")
            return None

    skeletons = [item['k25'] for item in frame_data_list]
    if len(skeletons) == 0:
        st.warning("⚠️ No se pudieron extraer esqueletos del video.")






















        return None


















    # 2. Chunking, Normalización y Preparación
    sk_arr = np.stack(skeletons, axis=0)
    T = sk_arr.shape[0]

    chunks_4d = create_chunks_from_skeletons(skeletons, chunk_size=chunk_size)
    normalized_chunks = np.stack([normalize_skeleton_sequence(seq) for seq in chunks_4d], axis=0)
    X = prepare_chunks_for_model(normalized_chunks)

    # 3. Predicción
    with st.spinner('Paso 2/3: Clasificando gestos con el modelo...'):
        preds = model.predict(X, verbose=0)
        pred_inds = preds.argmax(axis=1)

    # 4. Visualización y Escritura del Video
    with st.spinner('Paso 3/3: Generando video de retroalimentación...'):
        visual_frames = []
        for i in range(preds.shape[0]):
            chunk_start_idx = i * chunk_size
            chunk_end_idx = min((i + 1) * chunk_size, T)

            predicted_label = class_names[pred_inds[i]]
            color = colors.get(predicted_label, (255, 255, 255))

            # Aplicar el color a todos los frames dentro del chunk
            for j in range(chunk_start_idx, chunk_end_idx):
                if j < len(frame_data_list): # Evitar error si el padding creó más frames de los que hay
                    data = frame_data_list[j]
                    frame = data['frame'].copy()
                    pose_landmarks = data['pose_landmarks']
                    visual_frame = draw_skeleton_and_label(frame, pose_landmarks, predicted_label, color)
                    visual_frames.append(visual_frame)

        # Escritura a un archivo temporal para Streamlit
        if not visual_frames:
            st.warning("⚠️ No se generaron frames visuales.")
            return None

        H, W, _ = visual_frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        cap.release()

        # Usar un archivo temporal para el resultado
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_output:
            output_path = temp_output.name

        out = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
        for frame in visual_frames:
            out.write(frame)
        out.release()

    return output_path

# --------------------------------------------------------------------
# 💻 INTERFAZ PRINCIPAL DE STREAMLIT
# --------------------------------------------------------------------

def main():
    st.set_page_config(page_title="Clasificación de Gestos SkillTalk", layout="wide")

    st.title("🗣️ Clasificador de Gestos de Presentación")
    st.markdown("Analiza la actividad gestual **Beat** vs. **No-Gesture** en un video:")
    st.markdown(f"**🟢 Beat (Gesto activo)** | **🔵 No-Gesture**")

    # 1. Cargar el Modelo
    MODEL_PATH = "mlp_lstm_ted.h5"
    model = load_ml_model(MODEL_PATH)
    if model is None:
        return

    # 2. Componente de Subida de Archivo
    uploaded_file = st.file_uploader(
        "Sube un archivo de video (MP4 es el más recomendado)",
        type=['mp4', 'mov', 'avi']
    )

    if uploaded_file is not None:
        # Guardar el archivo subido en un path temporal
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name

        col_orig, col_result = st.columns(2)

        with col_orig:
            st.subheader("Video Original")
            st.video(video_path)

        with col_result:
            if st.button("▶️ Ejecutar Clasificación"):
                st.info("Iniciando clasificación. El procesamiento puede tardar varios minutos dependiendo de la longitud del video.")
                
                output_video_path = run_classification_pipeline(
                    video_path, model, CHUNK_SIZE, CLASS_NAMES, COLORS
                )

                if output_video_path:
                    st.success("✅ Clasificación completada. ¡Aquí está el resultado!")
                    st.subheader("Video Clasificado y Visualizado")
                    st.video(output_video_path)

                    # Limpiar archivos temporales
                    try:
                        os.unlink(output_video_path)
                    except Exception as e:
                        st.warning(f"No se pudo eliminar el archivo de salida temporal.")
                
        # Asegurarse de que el video de entrada se elimine al final
        try:
            os.unlink(video_path)
        except Exception as e:
            st.warning(f"No se pudo eliminar el archivo de entrada temporal.")

# Ejecutar la aplicación principal
if __name__ == "__main__":
    main()
if **name** == "**main**":
main()

#!/usr/bin/env bash
set -e

# Variables de entorno para Streamlit
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_SERVER_PORT=${PORT:-8501}
export STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Asegúrate de que el archivo principal se llama modelo_final_skilltalk.py
exec streamlit run app.py --server.port $STREAMLIT_SERVER_PORT --server.address $STREAMLIT_SERVER_ADDRESS

"""Constantes y rutas compartidas del proyecto.

Todas las rutas son relativas a la raíz del repositorio para que los comandos
funcionen en cualquier equipo sin editar rutas personales.
"""
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]

# --- Datos ---------------------------------------------------------------
DATA_RAW = RAIZ / "data" / "raw"
DATA_PREP = RAIZ / "data" / "prepared"
ARCHIVO_TRAIN = DATA_RAW / "UNSW_NB15_training-set.csv"
ARCHIVO_TEST = DATA_RAW / "UNSW_NB15_testing-set.csv"

FUENTE_OFICIAL = "https://research.unsw.edu.au/projects/unsw-nb15-dataset"
MIRROR_BASE = (
    "https://raw.githubusercontent.com/Nir-J/ML-Projects/"
    "44bbd02b9e70f74dd915218d5eb702f125341ffc/UNSW-Network_Packet_Classification/"
)
FECHA_DESCARGA = "2026-10-30T16:40:37Z"   # descarga registrada desde el mirror fijado
# Huellas publicadas en el anexo del enunciado (SHA-256 de los CSV sin comprimir).
SHA256_ESPERADO = {
    "UNSW_NB15_training-set.csv": "bec7dd5ec88dc2a0ccc7a07879d338395ed7421750f675fd0339e07dfe0648fa",
    "UNSW_NB15_testing-set.csv": "734fe6642edf758f7c94d7d9149426b49d202fe8e7bf0bef47392489c3c0a559",
}

# --- Variables -------------------------------------------------------------
OBJETIVO = "label"            # 0 = normal, 1 = ataque (clase positiva)
EXCLUIDAS = ["id", "attack_cat", "label"]   # nunca entran al modelo
CATEGORICAS = ["proto", "service", "state"]
# El resto de columnas (39) son numéricas; se derivan al cargar.

# --- Particiones -------------------------------------------------------------
SEMILLA = 42
N_FOLDS_SPLIT = 5   # StratifiedGroupKFold: 1 de 5 grupos de pliegues = 20 % validación

# --- Evidencias ----------------------------------------------------------------
EVID = RAIZ / "evidencias"
# Backend local SQLite (el file store de MLflow 3 está en mantenimiento) y artefactos en mlruns/.
MLFLOW_URI = f"sqlite:///{(RAIZ / 'mlflow.db').as_posix()}"
MLFLOW_ARTEFACTOS = (RAIZ / "mlruns").as_uri()
EXPERIMENTO_MLFLOW = "defiende-el-servidor"

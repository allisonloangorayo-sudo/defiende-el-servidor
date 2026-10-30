"""Predicción con el modelo publicado (uso: python predict.py entrada.csv salida.csv).

La entrada debe tener las 42 variables listadas en config.json["variables"]; columnas
extra (row_id, id, label, attack_cat) se ignoran. La regla de decisión es:
    y_pred = 1 (ataque) si score_attack >= config.json["umbral"], si no 0 (normal).
"""
import hashlib
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

AQUI = Path(__file__).resolve().parent


def cargar():
    cfg = json.loads((AQUI / "config.json").read_text())
    ruta = AQUI / "model.joblib"
    sha = hashlib.sha256(ruta.read_bytes()).hexdigest()
    # joblib usa pickle: solo se carga si el archivo es exactamente el publicado.
    if sha != cfg["sha256_model_joblib"]:
        raise ValueError(f"SHA-256 inesperado para model.joblib: {sha}")
    return joblib.load(ruta), cfg


def predecir(df, modelo, cfg):
    score = modelo.predict_proba(df[cfg["variables"]])[:, 1]
    out = pd.DataFrame({"score_attack": score,
                        "y_pred": (score >= cfg["umbral"]).astype(int)})
    if "row_id" in df.columns:
        out.insert(0, "row_id", df["row_id"].to_numpy())
    return out


if __name__ == "__main__":
    modelo, cfg = cargar()
    entrada = pd.read_csv(sys.argv[1])
    salida = predecir(entrada, modelo, cfg)
    salida.to_csv(sys.argv[2] if len(sys.argv) > 2 else sys.stdout, index=False,
                  float_format="%.17g")

"""Evaluación final ÚNICA de la versión congelada sobre toda la partición de test.

Uso:  python -m src.evaluar_test

* Lee la decisión de evidencias/seleccion.json (modelo, umbral, hash) y se niega a
  ejecutar si metricas-test.json ya existe: el test se evalúa una sola vez.
* Verifica que models/v2/model.joblib tenga el SHA-256 congelado y que el CSV de test
  sea el auditado.
* Escribe evidencias/predicciones-test.csv (una fila por registro original) y
  evidencias/metricas-test.json. No modifica el modelo ni el umbral.
"""
from __future__ import annotations

import datetime as dt
import json
import subprocess

import joblib
import pandas as pd

from src.config import ARCHIVO_TEST, DATA_PREP, EVID, OBJETIVO, RAIZ
from src.metricas import evaluar
from src.utils import guardar_json, sha256_archivo

SALIDA_METRICAS = EVID / "metricas-test.json"
SALIDA_PRED = EVID / "predicciones-test.csv"


def main() -> None:
    if SALIDA_METRICAS.exists():
        raise SystemExit("metricas-test.json ya existe: la evaluación final solo se hace una vez.")
    sel = json.loads((EVID / "seleccion.json").read_text())
    cfg = json.loads((RAIZ / "models" / sel["version_elegida"] / "config.json").read_text())
    ruta_modelo = RAIZ / "models" / sel["version_elegida"] / "model.joblib"
    sha = sha256_archivo(ruta_modelo)
    assert sha == sel["sha256_model_joblib"] == cfg["sha256_model_joblib"], "Modelo distinto al congelado"
    assert cfg["umbral"] == sel["umbral"], "Umbral distinto al congelado"

    # El commit que congeló la selección debe existir antes de esta evaluación.
    commit_congelacion = subprocess.check_output(
        ["git", "log", "-1", "--format=%H", "--", "evidencias/seleccion.json"], cwd=RAIZ, text=True).strip()
    commit_actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=RAIZ, text=True).strip()

    test = pd.read_csv(DATA_PREP / "test.csv")
    modelo = joblib.load(ruta_modelo)
    score = modelo.predict_proba(test[cfg["variables"]])[:, 1]
    y_pred = (score >= cfg["umbral"]).astype(int)
    pred = pd.DataFrame({"row_id": test.row_id, "y_true": test[OBJETIVO],
                         "score_attack": score, "y_pred": y_pred})
    pred.to_csv(SALIDA_PRED, index=False, float_format="%.17g")

    m = evaluar(test[OBJETIVO], score, cfg["umbral"])
    por_categoria = (pd.DataFrame({"cat": test.attack_cat, "pred": y_pred})
                     .groupby("cat").pred.agg(["size", "mean"]).round(4))
    resultado = {
        "version": sel["version_elegida"],
        "experimento": cfg["experimento"],
        "mlflow_run_id": cfg["mlflow_run_id"],
        "sha256_model_joblib": sha,
        "hugging_face": sel["hugging_face"],
        "umbral": cfg["umbral"],
        "regla_decision": cfg["regla_decision"],
        "clase_positiva": "1 = ataque",
        "dataset": {
            "archivo_original": f"data/raw/{ARCHIVO_TEST.name}",
            "sha256_original": sha256_archivo(ARCHIVO_TEST),
            "archivo_preparado": "data/prepared/test.csv",
            "sha256_preparado": sha256_archivo(DATA_PREP / "test.csv"),
        },
        "filas_evaluadas": int(len(test)),
        "filas_archivo_publicado": 82332,
        "matriz_confusion": m["matriz_confusion"],
        "metricas": {k: m[k] for k in ("recall_ataque", "precision_ataque", "fpr", "macro_f1",
                                        "f1_ataque", "f1_normal", "accuracy")},
        "denominadores_cero": m["denominadores_cero"],
        "tasa_prediccion_ataque_por_attack_cat": {
            c: {"filas": int(r["size"]), "proporcion_predicha_ataque": float(r["mean"])}
            for c, r in por_categoria.iterrows()},
        "predicciones": {"archivo": "evidencias/predicciones-test.csv",
                         "sha256": sha256_archivo(SALIDA_PRED),
                         "columnas": "row_id (test:<id del CSV>), y_true, score_attack, y_pred",
                         "regla": "y_pred = 1 si score_attack >= umbral"},
        "commit_congelacion_seleccion": commit_congelacion,
        "commit_codigo_evaluacion": commit_actual,
        "fecha_evaluacion_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "evaluaciones_previas_en_test": 0,
    }
    guardar_json(resultado, SALIDA_METRICAS)
    print(json.dumps({k: resultado[k] for k in ("matriz_confusion", "metricas")}, indent=2))


if __name__ == "__main__":
    main()

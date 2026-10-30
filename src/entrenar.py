"""Entrena una configuración, la evalúa en validación y la registra en MLflow.

Uso:  python -m src.entrenar configs/experimentos/E1_rl_base.yaml

* Ajusta el pipeline completo solo con data/prepared/train.csv.
* Evalúa en data/prepared/val.csv (misma partición para los cuatro experimentos).
* Registra en MLflow: hipótesis, parámetros, semilla, commit Git, md5 DVC de los datos,
  métricas a umbral 0,5 y al umbral operativo, matriz de confusión y el modelo completo.
* Nunca lee el archivo de test.
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import yaml

from src.config import DATA_PREP, EXPERIMENTO_MLFLOW, MLFLOW_URI, OBJETIVO, RAIZ
from src.metricas import evaluar
from src.modelos import construir_pipeline
from src.utils import (columnas_numericas, columnas_predictoras, dvc_md5, git_commit,
                       sha256_archivo)


def cargar_regla() -> dict:
    with open(RAIZ / "configs" / "seleccion.yaml") as f:
        return yaml.safe_load(f)


def umbral_operativo(y, score, regla: dict) -> dict:
    """Mayor recall con FPR <= fpr_maximo; desempate por macro-F1 y luego umbral más alto."""
    r = regla["rejilla_umbrales"]
    candidatos = np.round(np.arange(r["inicio"], r["fin"] + 1e-9, r["paso"]), 2)
    filas = [evaluar(y, score, float(u)) for u in candidatos]
    validos = [m for m in filas if m["fpr"] <= regla["fpr_maximo"]]
    if not validos:   # ningún umbral cumple: se toma el de menor FPR y se avisa
        mejor = min(filas, key=lambda m: (m["fpr"], -m["recall_ataque"]))
        mejor["cumple_fpr_maximo"] = False
    else:
        mejor = max(validos, key=lambda m: (m["recall_ataque"], m["macro_f1"], m["umbral"]))
        mejor["cumple_fpr_maximo"] = True
    return mejor, filas


def main(ruta_cfg: str) -> None:
    with open(ruta_cfg) as f:
        cfg = yaml.safe_load(f)
    regla = cargar_regla()
    commit = git_commit()
    np.random.seed(cfg["semilla"])

    train = pd.read_csv(DATA_PREP / "train.csv")
    val = pd.read_csv(DATA_PREP / "val.csv")
    X_cols = columnas_predictoras(train)
    assert len(X_cols) == 42
    num_cols = columnas_numericas(train)

    pipe = construir_pipeline(cfg, num_cols)
    t0 = time.perf_counter()
    pipe.fit(train[X_cols], train[OBJETIVO])
    seg_entrenamiento = time.perf_counter() - t0

    score_val = pipe.predict_proba(val[X_cols])[:, 1]
    y_val = val[OBJETIVO].to_numpy()
    m_defecto = evaluar(y_val, score_val, regla["umbral_por_defecto"])
    m_oper, barrido = umbral_operativo(y_val, score_val, regla)

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENTO_MLFLOW)
    with mlflow.start_run(run_name=f"{cfg['id']}_{cfg['nombre']}") as run, \
            tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        modelo_path = tmp / "model.joblib"
        joblib.dump(pipe, modelo_path)
        sha_modelo = sha256_archivo(modelo_path)

        mlflow.set_tags({
            "experimento_id": cfg["id"],
            "familia": cfg["familia"],
            "hipotesis": cfg["hipotesis"].strip(),
            "cambio": cfg["cambio"].strip(),
            "git_commit": commit,
            "dvc_raw_md5": dvc_md5("data/raw.dvc"),
            "dvc_prepared_md5": dvc_md5("data/prepared.dvc"),
            "sha256_train_csv": sha256_archivo(DATA_PREP / "train.csv"),
            "sha256_val_csv": sha256_archivo(DATA_PREP / "val.csv"),
            "sha256_model_joblib": sha_modelo,
            "config_path": str(Path(ruta_cfg).as_posix()),
            "regla_seleccion": f"max recall_ataque s.a. fpr<={regla['fpr_maximo']}",
        })
        params = {"semilla": cfg["semilla"], "estimador": cfg["modelo"]["clase"],
                  "pre_numericas": cfg["preprocesamiento"]["numericas"],
                  "pre_categoricas": cfg["preprocesamiento"]["categoricas"],
                  "n_train": len(train), "n_val": len(val), "n_variables": len(X_cols)}
        params.update({f"modelo__{k}": v for k, v in cfg["modelo"].get("parametros", {}).items()})
        mlflow.log_params(params)

        def log_bloque(prefijo: str, m: dict) -> None:
            mlflow.log_metrics({f"{prefijo}_{k}": m[k] for k in
                                ("umbral", "recall_ataque", "precision_ataque", "fpr",
                                 "macro_f1", "accuracy")})
            mlflow.log_metrics({f"{prefijo}_{k}": v for k, v in m["matriz_confusion"].items()})

        log_bloque("val_u050", m_defecto)
        log_bloque("val_oper", m_oper)
        mlflow.log_metric("segundos_entrenamiento", round(seg_entrenamiento, 2))

        resultado = {"experimento": cfg["id"], "run_id": run.info.run_id, "git_commit": commit,
                     "sha256_model_joblib": sha_modelo, "validacion_umbral_050": m_defecto,
                     "validacion_umbral_operativo": m_oper}
        (tmp / "metricas_validacion.json").write_text(json.dumps(resultado, indent=2))
        pd.DataFrame(barrido).drop(columns=["matriz_confusion", "denominadores_cero"]) \
            .to_csv(tmp / "barrido_umbrales_val.csv", index=False)
        pd.DataFrame({"row_id": val.row_id, "y_true": y_val, "score_attack": score_val}) \
            .to_csv(tmp / "scores_validacion.csv", index=False, float_format="%.10f")
        (tmp / "config.yaml").write_text(Path(ruta_cfg).read_text())
        for archivo in ["model.joblib", "metricas_validacion.json", "barrido_umbrales_val.csv",
                        "scores_validacion.csv", "config.yaml"]:
            mlflow.log_artifact(str(tmp / archivo))

    print(json.dumps({"run_id": run.info.run_id, "u050": {k: round(m_defecto[k], 4) for k in
                      ("recall_ataque", "precision_ataque", "fpr", "macro_f1")},
                      "oper": {k: round(m_oper[k], 4) for k in
                               ("umbral", "recall_ataque", "precision_ataque", "fpr", "macro_f1")},
                      "seg": round(seg_entrenamiento, 1)}))


if __name__ == "__main__":
    main(sys.argv[1])

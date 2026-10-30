"""Construye la carpeta publicable de una versión del modelo (v1 o v2).

Uso:  python -m src.empaquetar configs/v1.yaml

Toma el artefacto `model.joblib` del run de MLflow indicado en la config (el mismo
archivo, byte a byte, verificado por SHA-256) y genera en models/<version>/:

  model.joblib          pipeline completo: preprocesamiento + estimador (no va a Git)
  config.json           umbral, variables, hashes y referencias Git/DVC/MLflow
  metricas_validacion.json
  requirements.txt      dependencias mínimas de inferencia
  predict.py            script de predicción autocontenido
  README.md             model card

La carpeta se sube tal cual a Hugging Face Hub (scripts/publicar.sh).
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import mlflow
import sklearn
import yaml
from mlflow.tracking import MlflowClient

from src.config import (CATEGORICAS, EXCLUIDAS, FUENTE_OFICIAL, MLFLOW_URI, RAIZ)
from src.metricas import evaluar
from src.utils import git_commit, guardar_json, sha256_archivo

REQ_INFERENCIA = """numpy==2.5.3
pandas==3.0.5
scipy==1.18.1
scikit-learn==1.9.1
joblib==1.6.0
threadpoolctl==3.7.0
"""

PREDICT_PY = '''"""Predicción con el modelo publicado (uso: python predict.py entrada.csv salida.csv).

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
                  float_format="%.10f")
'''


def tabla_metricas(m: dict) -> str:
    cm = m["matriz_confusion"]
    return (
        "| Métrica (validación, ataque = positivo) | Valor |\n|---|---|\n"
        f"| Umbral | {m['umbral']:.2f} |\n"
        f"| Recall de ataque | {m['recall_ataque']:.4f} |\n"
        f"| Precision de ataque | {m['precision_ataque']:.4f} |\n"
        f"| Tasa de falsos positivos (FPR) | {m['fpr']:.4f} |\n"
        f"| Macro-F1 | {m['macro_f1']:.4f} |\n"
        f"| Accuracy (complementaria) | {m['accuracy']:.4f} |\n\n"
        "Matriz de confusión (filas = real, columnas = predicho):\n\n"
        "| | Pred. normal | Pred. ataque |\n|---|---|---|\n"
        f"| Real normal | {cm['tn']} (TN) | {cm['fp']} (FP) |\n"
        f"| Real ataque | {cm['fn']} (FN) | {cm['tp']} (TP) |\n"
    )


def model_card(cfg_v: dict, info: dict, m: dict) -> str:
    v = cfg_v["version"]
    repo = cfg_v["hf_repo"]
    return f"""---
language: es
license: cc-by-4.0
library_name: sklearn
pipeline_tag: tabular-classification
tags:
  - tabular-classification
  - network-intrusion-detection
  - unsw-nb15
  - mlops
  - academic
---

# Defiende el servidor · {v} — {cfg_v['titulo']}

Clasificador binario de conexiones de red (UNSW-NB15): **0 = normal, 1 = ataque**.
Proyecto académico del curso MLOps 2026-II (Universidad Externado de Colombia),
taller *Defiende el servidor*. Autora: Allison Loango.

## Propósito

{cfg_v['proposito'].strip()}

**Uso fuera de alcance:** no es un IDS para producción. El dataset es histórico (2015,
tráfico sintético del laboratorio UNSW Canberra) y no demuestra eficacia sobre una red actual.

## Qué contiene esta revisión

| Archivo | Contenido |
|---|---|
| `model.joblib` | `Pipeline` de scikit-learn completo: preprocesamiento + `{info['estimador']}` |
| `config.json` | Umbral, variables de entrada, hashes y referencias de trazabilidad |
| `metricas_validacion.json` | Métricas y matriz de confusión en validación |
| `predict.py` | Predicción autocontenida con verificación SHA-256 del artefacto |
| `requirements.txt` | Dependencias de inferencia fijadas |

- **SHA-256 de `model.joblib`:** `{info['sha256_model_joblib']}`
- **Umbral de decisión:** `{m['umbral']:.2f}` → ataque si `score_attack >= {m['umbral']:.2f}`
- **Preprocesamiento:** {info['preprocesamiento']}

## Datos

- Fuente: [UNSW-NB15]({FUENTE_OFICIAL}) (Moustafa & Slay, 2015), particiones publicadas
  `UNSW_NB15_training-set.csv` (175.341 filas) y `UNSW_NB15_testing-set.csv` (82.332 filas).
- Entrenamiento y validación salen **solo** del archivo de entrenamiento con
  `StratifiedGroupKFold` (semilla 42): las filas con entradas idénticas forman un grupo
  que nunca se reparte entre ambas particiones. Train = 140.274 filas, validación = 35.067.
- Variables: 42 predictoras (3 categóricas: `proto`, `service`, `state`; 39 numéricas).
  Excluidas: `{'`, `'.join(EXCLUIDAS)}` (`attack_cat` revela la respuesta, `id` es identificador).
- El archivo de test no participó en el entrenamiento ni en la elección del modelo o umbral.

## Métricas de validación

{tabla_metricas(m)}
{cfg_v.get('comparacion', '').strip()}

## Limitaciones

- Dataset histórico y de laboratorio; la distribución de test difiere de la de entrenamiento
  (68 % de ataques en entrenamiento frente a 55 % en test).
- 42 % de las filas de entrenamiento repiten entradas; el split agrupado evita fuga, pero las
  métricas dependen de esa estructura de duplicados.
- 229 grupos de entradas idénticas tienen etiquetas contradictorias (940 filas): ruido irreducible.
- El umbral se eligió en validación; las métricas al umbral elegido son algo optimistas.
- `joblib` usa pickle: cargue solo archivos cuyo SHA-256 coincida con el publicado.

## Uso

```bash
pip install -r requirements.txt huggingface_hub
```

```python
from huggingface_hub import snapshot_download
import sys
ruta = snapshot_download("{repo}", revision="<SHA completo de la revisión {v}>")
sys.path.insert(0, ruta)
import pandas as pd
from predict import cargar, predecir
modelo, cfg = cargar()                      # verifica SHA-256 antes de cargar
df = pd.read_csv("conexiones.csv")          # 42 variables de config.json["variables"]
print(predecir(df, modelo, cfg).head())     # row_id, score_attack, y_pred
```

## Trazabilidad

| Pieza | Referencia |
|---|---|
| Código que entrenó el modelo (Git) | `{info['git_commit_entrenamiento']}` |
| Config de versión `configs/{v}.yaml` confirmada en (Git) | `{info['git_commit_empaquetado']}` |
| Tag Git de la versión | `{v}` (commit que añade `models/{v}/` con esta model card) |
| Repositorio de código | {cfg_v['github_repo']} |
| Datos originales (DVC, `data/raw.dvc`) | md5 `{info['dvc_raw_md5']}` |
| Datos preparados (DVC, `data/prepared.dvc`) | md5 `{info['dvc_prepared_md5']}` |
| Almacenamiento DVC | {cfg_v['dvc_store']} |
| Experimento MLflow | `{info['experimento']}` · run ID `{info['mlflow_run_id']}` |
| Config del experimento | `{info['config_experimento']}` |

## Referencias

- Moustafa, N. & Slay, J. (2015). UNSW-NB15: a comprehensive data set for network intrusion
  detection systems. *MilCIS 2015*, IEEE.
- Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR* 12.
"""


def main(ruta_cfg: str) -> None:
    with open(ruta_cfg) as f:
        cfg_v = yaml.safe_load(f)
    version = cfg_v["version"]
    commit_empaquetado = git_commit(ruta_cfg)

    mlflow.set_tracking_uri(MLFLOW_URI)
    run = MlflowClient().get_run(cfg_v["mlflow_run_id"])
    tags = run.data.tags

    destino = RAIZ / "models" / version
    destino.mkdir(parents=True, exist_ok=True)
    origen = Path(mlflow.artifacts.download_artifacts(run_id=run.info.run_id,
                                                      artifact_path="model.joblib"))
    shutil.copyfile(origen, destino / "model.joblib")
    sha_modelo = sha256_archivo(destino / "model.joblib")
    assert sha_modelo == tags["sha256_model_joblib"], "El artefacto no coincide con el run"

    # Métricas de validación recalculadas desde los scores guardados en el run.
    import pandas as pd
    scores = pd.read_csv(mlflow.artifacts.download_artifacts(
        run_id=run.info.run_id, artifact_path="scores_validacion.csv"))
    m = evaluar(scores.y_true, scores.score_attack, float(cfg_v["umbral"]))

    with open(RAIZ / tags["config_path"]) as f:
        cfg_exp = yaml.safe_load(f)
    pre = cfg_exp["preprocesamiento"]
    info = {
        "version": version,
        "experimento": tags["experimento_id"],
        "mlflow_run_id": run.info.run_id,
        "estimador": run.data.params["estimador"],
        "preprocesamiento": f"numéricas `{pre['numericas']}`, categóricas `{pre['categoricas']}`",
        "config_experimento": tags["config_path"],
        "git_commit_entrenamiento": tags["git_commit"],
        "git_commit_empaquetado": commit_empaquetado,
        "dvc_raw_md5": tags["dvc_raw_md5"],
        "dvc_prepared_md5": tags["dvc_prepared_md5"],
        "sha256_model_joblib": sha_modelo,
    }
    cols = pd.read_csv(RAIZ / "data" / "prepared" / "val.csv", nrows=1).columns
    variables = [c for c in cols if c not in set(EXCLUIDAS) | {"row_id", "grupo"}]
    config_json = {
        **info,
        "umbral": float(cfg_v["umbral"]),
        "regla_decision": "y_pred = 1 (ataque) si score_attack >= umbral; score_attack = predict_proba[:, 1]",
        "clase_positiva": "1 = ataque",
        "variables": variables,
        "categoricas": CATEGORICAS,
        "sklearn_version": sklearn.__version__,
        "razon_version": cfg_v["razon"].strip(),
    }
    guardar_json(config_json, destino / "config.json")
    guardar_json({"version": version, "mlflow_run_id": run.info.run_id, "validacion": m},
                 destino / "metricas_validacion.json")
    (destino / "requirements.txt").write_text(REQ_INFERENCIA)
    (destino / "predict.py").write_text(PREDICT_PY)
    (destino / "README.md").write_text(model_card(cfg_v, info, m))
    print(json.dumps({"version": version, "sha256_model_joblib": sha_modelo,
                      "umbral": cfg_v["umbral"], "run": run.info.run_id}, indent=2))


if __name__ == "__main__":
    main(sys.argv[1])

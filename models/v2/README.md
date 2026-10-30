---
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

# Defiende el servidor · v2 — gradient boosting (E4) con umbral 0,66 elegido en validación

Clasificador binario de conexiones de red (UNSW-NB15): **0 = normal, 1 = ataque**.
Proyecto académico del curso MLOps 2026-II (Universidad Externado de Colombia),
taller *Defiende el servidor*. Autora: Allison Loango.

## Propósito

Versión revisada para un equipo de seguridad con presupuesto de falsas alarmas: maximiza la detección de ataques manteniendo la tasa de falsos positivos en validación por debajo de 5 %.

**Uso fuera de alcance:** no es un IDS para producción. El dataset es histórico (2015,
tráfico sintético del laboratorio UNSW Canberra) y no demuestra eficacia sobre una red actual.

## Qué contiene esta revisión

| Archivo | Contenido |
|---|---|
| `model.joblib` | `Pipeline` de scikit-learn completo: preprocesamiento + `HistGradientBoostingClassifier` |
| `config.json` | Umbral, variables de entrada, hashes y referencias de trazabilidad |
| `metricas_validacion.json` | Métricas y matriz de confusión en validación |
| `predict.py` | Predicción autocontenida con verificación SHA-256 del artefacto |
| `requirements.txt` | Dependencias de inferencia fijadas |

- **SHA-256 de `model.joblib`:** `23c42b34e7729858b5a809fc7dde153d2ca108179db7ad3d745297ddca125671`
- **Umbral de decisión:** `0.66` → ataque si `score_attack >= 0.66`
- **Preprocesamiento:** numéricas `ninguno`, categóricas `ordinal`

## Datos

- Fuente: [UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset) (Moustafa & Slay, 2015), particiones publicadas
  `UNSW_NB15_training-set.csv` (175.341 filas) y `UNSW_NB15_testing-set.csv` (82.332 filas).
- Entrenamiento y validación salen **solo** del archivo de entrenamiento con
  `StratifiedGroupKFold` (semilla 42): las filas con entradas idénticas forman un grupo
  que nunca se reparte entre ambas particiones. Train = 140.274 filas, validación = 35.067.
- Variables: 42 predictoras (3 categóricas: `proto`, `service`, `state`; 39 numéricas).
  Excluidas: `id`, `attack_cat`, `label` (`attack_cat` revela la respuesta, `id` es identificador).
- El archivo de test no participó en el entrenamiento ni en la elección del modelo o umbral.

## Métricas de validación

| Métrica (validación, ataque = positivo) | Valor |
|---|---|
| Umbral | 0.66 |
| Recall de ataque | 0.9532 |
| Precision de ataque | 0.9769 |
| Tasa de falsos positivos (FPR) | 0.0480 |
| Macro-F1 | 0.9464 |
| Accuracy (complementaria) | 0.9528 |

Matriz de confusión (filas = real, columnas = predicho):

| | Pred. normal | Pred. ataque |
|---|---|---|
| Real normal | 10662 (TN) | 538 (FP) |
| Real ataque | 1117 (FN) | 22750 (TP) |

## Comparación con v1 (validación)

| | v1 (E1, u=0,50) | v2 (E4, u=0,66) | Cambio |
|---|---|---|---|
| Recall de ataque | 0.9876 | 0.9532 | -3.4 pp |
| Precision de ataque | 0.9230 | 0.9769 | +5.4 pp |
| FPR | 0.1755 | 0.0480 | -12.8 pp |
| Macro-F1 | 0.9225 | 0.9464 | +2.4 pp |
| Falsos positivos | 1966 | 538 | |
| Ataques no detectados (FN) | 296 | 1117 | |

**Qué mejora:** las falsas alarmas bajan de 1966 a 538 y la precision y el macro-F1 suben. **Qué empeora:** el recall baja 3.4 pp: se aceptan 821 ataques no detectados más (sobre todo Fuzzers) a cambio de respetar el presupuesto de FPR <= 5 %.

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
ruta = snapshot_download("allisonloango/defiende-el-servidor-unsw-nb15", revision="<SHA completo de la revisión v2>")
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
| Código que entrenó el modelo (Git) | `0bf39dff1c2f7858f905ee8213512ce872f6b671` |
| Config de versión `configs/v2.yaml` confirmada en (Git) | `283a9eb680cdee4d4f1bbe3c35e3d38942601428` |
| Tag Git de la versión | `v2` (commit que añade `models/v2/` con esta model card) |
| Repositorio de código | https://github.com/allisonloangorayo-sudo/defiende-el-servidor |
| Datos originales (DVC, `data/raw.dvc`) | md5 `fe93051ae209ffb0128a31e7a478c009.dir` |
| Datos preparados (DVC, `data/prepared.dvc`) | md5 `5ae30cc8b0b02e4fe567608df0b93638.dir` |
| Almacenamiento DVC | dataset de Hugging Face `allisonloango/defiende-el-servidor-dvc` (remote DVC `hf`, HTTP de solo lectura) |
| Experimento MLflow | `E4` · run ID `68d9aeb74e0d402d9209c10ca1a735da` |
| Config del experimento | `configs/experimentos/E4_boosting.yaml` |

## Referencias

- Moustafa, N. & Slay, J. (2015). UNSW-NB15: a comprehensive data set for network intrusion
  detection systems. *MilCIS 2015*, IEEE.
- Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR* 12.

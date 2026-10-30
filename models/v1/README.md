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

# Defiende el servidor · v1 — referencia regresión logística (umbral 0,50)

Clasificador binario de conexiones de red (UNSW-NB15): **0 = normal, 1 = ataque**.
Proyecto académico del curso MLOps 2026-II (Universidad Externado de Colombia),
taller *Defiende el servidor*. Autora: Allison Loango.

## Propósito

Línea base interpretable para separar conexiones normales de ataques en UNSW-NB15. Sirve como punto de comparación para la versión 2 y como primera solución funcional.

**Uso fuera de alcance:** no es un IDS para producción. El dataset es histórico (2015,
tráfico sintético del laboratorio UNSW Canberra) y no demuestra eficacia sobre una red actual.

## Qué contiene esta revisión

| Archivo | Contenido |
|---|---|
| `model.joblib` | `Pipeline` de scikit-learn completo: preprocesamiento + `LogisticRegression` |
| `config.json` | Umbral, variables de entrada, hashes y referencias de trazabilidad |
| `metricas_validacion.json` | Métricas y matriz de confusión en validación |
| `predict.py` | Predicción autocontenida con verificación SHA-256 del artefacto |
| `requirements.txt` | Dependencias de inferencia fijadas |

- **SHA-256 de `model.joblib`:** `36648d8096669151f881ffa0d00f32b07daab389cd8927b0898d4fe81a33a7f3`
- **Umbral de decisión:** `0.50` → ataque si `score_attack >= 0.50`
- **Preprocesamiento:** numéricas `estandar`, categóricas `onehot`

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
| Umbral | 0.50 |
| Recall de ataque | 0.9876 |
| Precision de ataque | 0.9230 |
| Tasa de falsos positivos (FPR) | 0.1755 |
| Macro-F1 | 0.9225 |
| Accuracy (complementaria) | 0.9355 |

Matriz de confusión (filas = real, columnas = predicho):

| | Pred. normal | Pred. ataque |
|---|---|---|
| Real normal | 9234 (TN) | 1966 (FP) |
| Real ataque | 296 (FN) | 23571 (TP) |



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
ruta = snapshot_download("allisonloango/defiende-el-servidor-unsw-nb15", revision="<SHA completo de la revisión v1>")
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
| Código que entrenó el modelo (Git) | `720e330562248d9b1b5c68a4c647d12aad1abfea` |
| Config de versión `configs/v1.yaml` confirmada en (Git) | `8ee1114479ae66dd2a346faedff632073136a23f` |
| Tag Git de la versión | `v1` (commit que añade `models/v1/` con esta model card) |
| Repositorio de código | https://github.com/allisonloangorayo-sudo/defiende-el-servidor |
| Datos originales (DVC, `data/raw.dvc`) | md5 `fe93051ae209ffb0128a31e7a478c009.dir` |
| Datos preparados (DVC, `data/prepared.dvc`) | md5 `5ae30cc8b0b02e4fe567608df0b93638.dir` |
| Almacenamiento DVC | dataset de Hugging Face `allisonloango/defiende-el-servidor-dvc` (remote DVC `hf`, HTTP de solo lectura) |
| Experimento MLflow | `E1` · run ID `48de31fc1e57432194b41f4bc9e4a3a5` |
| Config del experimento | `configs/experimentos/E1_rl_base.yaml` |

## Referencias

- Moustafa, N. & Slay, J. (2015). UNSW-NB15: a comprehensive data set for network intrusion
  detection systems. *MilCIS 2015*, IEEE.
- Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR* 12.

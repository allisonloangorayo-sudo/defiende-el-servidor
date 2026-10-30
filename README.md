# Defiende el servidor · versionado de código, datos, experimentos y modelos

Taller evaluable de **MLOps 2026-II** (Universidad Externado de Colombia). Clasificación
binaria de conexiones de red de **UNSW-NB15**: `0 = normal`, `1 = ataque` (clase positiva).

Pregunta central del taller: si se pierde la carpeta de trabajo, ¿se puede recuperar una
versión concreta, explicar de qué datos y experimento viene y usarla para predecir? Este
repositorio responde con una cadena de identificadores verificables:

```
Código: commit Git ──► Datos: md5 DVC ──► Experimento: run ID MLflow ──► Modelo: revisión SHA en Hugging Face
                                                                          └─ SHA-256 de model.joblib en cada eslabón
```

| Pieza | Dónde |
|---|---|
| Código, configs, historial, tags `v1`/`v2` | GitHub · https://github.com/allisonloangorayo-sudo/defiende-el-servidor |
| Datos originales y preparados | DVC (`data/raw.dvc`, `data/prepared.dvc`) · remote `hf` en https://huggingface.co/datasets/allisonloango/defiende-el-servidor-dvc |
| Experimentos | MLflow local · exportación en `evidencias/mlflow/` |
| Modelos v1 y v2 | Hugging Face Hub · https://huggingface.co/allisonloango/defiende-el-servidor-unsw-nb15 (SHA en `configs/referencias.json`) |
| Índice de la entrega | [`entrega.md`](entrega.md) · informe: `informe.pdf` |

## Resultados en una mirada

| | Familia | Umbral | Recall ataque | Precision ataque | FPR | Macro-F1 |
|---|---|---|---|---|---|---|
| **v1** (E1, validación) | Regresión logística | 0,50 | 0,9876 | 0,9230 | 0,1755 | 0,9225 |
| **v2** (E4, validación) | HistGradientBoosting | 0,66 | 0,9532 | 0,9769 | 0,0480 | 0,9464 |
| **v2** (test, una sola vez) | | 0,66 | 0,9681 | 0,8712 | 0,1754 | 0,9010 |

v2 se eligió con la regla fijada en `configs/seleccion.yaml`: máximo recall de ataque con
FPR ≤ 5 % en validación. En test la FPR sube a 17,5 %: el tráfico normal de test se
distribuye distinto; se reporta sin modificar modelo ni umbral (ver `informe.pdf`).

## Estructura

```
├── README.md · entrega.md · informe.pdf · requirements.txt
├── configs/
│   ├── experimentos/E1..E4.yaml   hipótesis, preprocesamiento, modelo, semilla
│   ├── seleccion.yaml             regla de selección fijada antes de ver resultados
│   ├── conclusiones.yaml          interpretación de cada experimento en validación
│   ├── v1.yaml · v2.yaml          definición de cada versión publicada
│   └── referencias.json           SHA de Hugging Face (lo escribe scripts/publicar.sh)
├── data/
│   ├── raw.dvc                    versión DVC de los CSV recibidos
│   └── prepared.dvc               versión DVC de las particiones train/val/test
├── src/
│   ├── preparar.py                auditoría + split agrupado sin fuga
│   ├── modelos.py · entrenar.py   pipelines y registro en MLflow
│   ├── seleccionar.py             comparación, umbral y congelación de v2
│   ├── empaquetar.py              carpeta publicable de v1/v2 con model card
│   ├── evaluar_test.py            evaluación final única
│   ├── recuperacion.py · verificar_datos.py   prueba de recuperación
│   ├── exportar_mlflow.py · registrar_referencias.py · publicar_hf.py · informe.py
│   └── config.py · utils.py · metricas.py
├── models/v1 · models/v2          model card, config, predict.py (el binario se publica en HF)
├── scripts/publicar.sh · recuperar.sh · empaquetar_entrega.sh
└── evidencias/                    auditoria.json, experimentos.csv, seleccion.json,
                                   metricas-test.json, predicciones-test.csv, mlflow/,
                                   recuperacion/, figuras/
```

## Instalación

Requiere **Python 3.13** (probado con 3.13.18), `git` y [`uv`](https://docs.astral.sh/uv/)
(`curl -LsSf https://astral.sh/uv/install.sh | sh`). Las versiones están fijadas porque
`model.joblib` debe cargarse con scikit-learn 1.9.1.

```bash
git clone https://github.com/allisonloangorayo-sudo/defiende-el-servidor.git
cd defiende-el-servidor
uv venv --python 3.13
uv pip install -r requirements.txt
source .venv/bin/activate          # Windows: .venv\Scripts\activate
```

## Recuperar los datos

```bash
dvc pull                           # remote "hf": HTTP de solo lectura, sin credenciales
python -m src.verificar_datos      # SHA-256 frente al anexo del enunciado y a splits.json
```

Datos originales: `UNSW_NB15_training-set.csv` (175.341 filas, SHA-256 `bec7dd5e…48fa`) y
`UNSW_NB15_testing-set.csv` (82.332 filas, `734fe664…c559`), del mirror fijado que indica el
enunciado. Fuente oficial: https://research.unsw.edu.au/projects/unsw-nb15-dataset.

## Reproducir el pipeline

```bash
python -m src.preparar                                   # auditoría + particiones (determinista)
python -m src.entrenar configs/experimentos/E1_rl_base.yaml
python -m src.entrenar configs/experimentos/E4_boosting.yaml   # (E2, E3 igual)
mlflow ui --backend-store-uri sqlite:///mlflow.db       # consultar los runs nuevos
```

`src.entrenar` exige que el código y la config estén confirmados en Git: cada run registra
el commit, el md5 DVC de los datos, la hipótesis, los parámetros, la semilla, las métricas de
validación y el modelo completo. `src.seleccionar` y `src.evaluar_test` no se vuelven a
ejecutar: la decisión está congelada y el test se evaluó una sola vez.

## Descargar un modelo por SHA y predecir

```bash
SHA_V2=$(python -c "import json;print(json.load(open('configs/referencias.json'))['v2']['hf_revision_sha'])")
hf download allisonloango/defiende-el-servidor-unsw-nb15 --revision "$SHA_V2" --local-dir modelo_v2
python modelo_v2/predict.py evidencias/recuperacion/entrada_20_val.csv predicciones_v2.csv
```

`predict.py` verifica el SHA-256 de `model.joblib` antes de cargarlo, aplica
`y_pred = 1 si score_attack >= umbral` con el umbral de `config.json` y devuelve
`row_id, score_attack, y_pred`. Para v1 se usa `['v1']['hf_revision_sha']`.

## Prueba de recuperación

```bash
bash scripts/recuperar.sh                    # carpeta y entorno nuevos, cachés vacías
```

Clona el código desde GitHub, crea un entorno nuevo, ejecuta `dvc pull` desde el
almacenamiento entregado, verifica las huellas, descarga v1 y v2 por SHA completo y compara
sus predicciones sobre 20 filas fijas de validación con las guardadas (tolerancia 1e-9 en
`score_attack`, coincidencia exacta en `y_pred`). Evidencias en
`evidencias/recuperacion/entorno_nuevo/`.

## Publicación (solo la autora)

Requiere sesión en GitHub (`gh auth login`) y en Hugging Face (`hf auth login`, token con
permiso *write*). `bash scripts/publicar.sh` sube el historial con sus tags, el
almacenamiento DVC y las dos revisiones del modelo; verifica cada descarga por SHA y escribe
`configs/referencias.json`. Ningún script lee ni guarda tokens.

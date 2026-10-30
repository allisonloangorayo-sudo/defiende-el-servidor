# Entrega · MLOps — Defiende el servidor — Grupo A — Allison Loango y Kevin Nieto

## Integrantes y repositorio

- **Integrantes:** Allison Loango y Kevin Nieto · Grupo A (entrega en pareja)
- **Repositorio de código (GitHub):** https://github.com/allisonloangorayo-sudo/defiende-el-servidor
- **Commit final de la entrega:** el último commit de `main` (se informa completo en el mensaje de entrega; `git rev-parse HEAD`).
- **Repositorio de modelos (Hugging Face):** https://huggingface.co/allisonloango/defiende-el-servidor-unsw-nb15
- **Almacenamiento DVC:** https://huggingface.co/datasets/allisonloango/defiende-el-servidor-dvc · revisión `a93baca2bcf5601887d764dd10960f2f70a591f1`
  - Remote `hf` en `.dvc/config` (HTTP de solo lectura): `https://huggingface.co/datasets/allisonloango/defiende-el-servidor-dvc/resolve/main`
  - Acceso: repositorios públicos; `dvc pull` no requiere credenciales.

## Versiones del modelo

### v1 — referencia regresión logística (umbral 0,50)

- Commit del código que entrenó el modelo: `720e330562248d9b1b5c68a4c647d12aad1abfea`
- Commit con la config de la versión: `8ee1114479ae66dd2a346faedff632073136a23f` · tag Git `v1` → `af4cb0beb3a823ad404397a9eb193b8aa3417a96`
- Datos DVC: original `data/raw.dvc` md5 `fe93051ae209ffb0128a31e7a478c009.dir` (commit `87763e7cec12e97865eeaabe95c46ab539c710a3`); preparados `data/prepared.dvc` md5 `5ae30cc8b0b02e4fe567608df0b93638.dir` (commit `162eec5de623834b569fbd8854425e9fdfc72ff3`)
- Experimento y run de MLflow: E1 · `48de31fc1e57432194b41f4bc9e4a3a5`
- Hugging Face: https://huggingface.co/allisonloango/defiende-el-servidor-unsw-nb15 · revisión `403d252959457cb15cb470e83262d4734dabda33` (tag `v1`)
- SHA-256 de `model.joblib`: `36648d8096669151f881ffa0d00f32b07daab389cd8927b0898d4fe81a33a7f3`
- Umbral: `0.5` (ataque si `score_attack >= umbral`)

### v2 — gradient boosting (E4) con umbral 0,66 elegido en validación

- Commit del código que entrenó el modelo: `0bf39dff1c2f7858f905ee8213512ce872f6b671`
- Commit con la config de la versión: `283a9eb680cdee4d4f1bbe3c35e3d38942601428` · tag Git `v2` → `693d2ef5c21e0452f969f2e29471fe84509980bb`
- Datos DVC: original `data/raw.dvc` md5 `fe93051ae209ffb0128a31e7a478c009.dir` (commit `87763e7cec12e97865eeaabe95c46ab539c710a3`); preparados `data/prepared.dvc` md5 `5ae30cc8b0b02e4fe567608df0b93638.dir` (commit `162eec5de623834b569fbd8854425e9fdfc72ff3`)
- Experimento y run de MLflow: E4 · `68d9aeb74e0d402d9209c10ca1a735da`
- Hugging Face: https://huggingface.co/allisonloango/defiende-el-servidor-unsw-nb15 · revisión `c95111ff150a57f2666c77f0d4b220aaec801ada` (tag `v2`)
- SHA-256 de `model.joblib`: `23c42b34e7729858b5a809fc7dde153d2ca108179db7ad3d745297ddca125671`
- Umbral: `0.66` (ataque si `score_attack >= umbral`)

La model card de v2 se actualizó después de test en la revisión `c274d191303cba18bda196fe9618e1861106e463` (tag `v2-test`);
esa revisión solo cambia `README.md` y conserva el mismo `model.joblib` (SHA-256 verificado al publicar).

## Experimentos (validación, misma partición)

| ID | Familia | Run ID MLflow | Umbral 0,50 / operativo | Recall | Precision | FPR | Macro-F1 |
|---|---|---|---|---|---|---|---|
| E1 | regresion_logistica | `48de31fc1e57432194b41f4bc9e4a3a5` | 0.50 / 0.78 | 0.8950 | 0.9746 | 0.0496 | 0.9037 |
| E2 | regresion_logistica | `4f70fc75168a4bdda8145fa0396f5458` | 0.50 / 0.78 | 0.9132 | 0.9754 | 0.0491 | 0.9168 |
| E3 | arboles | `d7c8bfee8d2f476784a576304928ea04` | 0.50 / 0.70 | 0.9325 | 0.9762 | 0.0485 | 0.9310 |
| E4 | arboles | `68d9aeb74e0d402d9209c10ca1a735da` | 0.50 / 0.66 | 0.9532 | 0.9769 | 0.0480 | 0.9464 |

Métricas al umbral operativo (máximo recall con FPR <= 5 %). Las métricas a 0,50 están en `evidencias/experimentos.csv`.

## Decisión y test

- Elegida: **v2 = E4** con umbral **0.66**, solo con validación (`evidencias/seleccion.json`).
- Congelación: commit de selección `283a9eb680cdee4d4f1bbe3c35e3d38942601428` y tag `v2` → `693d2ef5c21e0452f969f2e29471fe84509980bb`, ambos anteriores al commit de evaluación `34e606d024a71c1408bd8f2c425105088b5e2a51`.
- Test (una sola vez, 82332 filas): recall 0.9681 · precision 0.8712 · FPR 0.1754 · macro-F1 0.9010.

## Ubicación de las evidencias

| Evidencia | Ruta |
|---|---|
| Auditoría de datos y particiones | `evidencias/auditoria.json` |
| Tabla de experimentos | `evidencias/experimentos.csv` |
| Selección congelada | `evidencias/seleccion.json` |
| Métricas de test | `evidencias/metricas-test.json` |
| Predicciones de test por fila | `evidencias/predicciones-test.csv` |
| Exportación de MLflow (4 runs, artefactos, `mlflow.db`) | `evidencias/mlflow/` (ver `evidencias/mlflow/README.md`) |
| Prueba de recuperación | `evidencias/recuperacion/` (entrada, salidas originales; `entorno_nuevo/` con log y comparaciones) |
| Figuras | `evidencias/figuras/` |
| Model cards versionadas | `models/v1/README.md`, `models/v2/README.md` |
| Referencias de publicación | `configs/referencias.json` |
| Informe | `informe.pdf` |

## Comandos

```bash
# Instalación (Python 3.13 con uv)
git clone https://github.com/allisonloangorayo-sudo/defiende-el-servidor.git && cd defiende-el-servidor
uv venv --python 3.13 && uv pip install -r requirements.txt
# Recuperar datos y verificar huellas
.venv/bin/dvc pull && .venv/bin/python -m src.verificar_datos
# Reproducir un experimento (registra un run nuevo en MLflow local)
.venv/bin/python -m src.entrenar configs/experimentos/E4_boosting.yaml
# Descargar v2 por SHA y predecir
.venv/bin/hf download allisonloango/defiende-el-servidor-unsw-nb15 --revision <SHA v2> --local-dir modelo_v2
.venv/bin/python modelo_v2/predict.py evidencias/recuperacion/entrada_20_val.csv predicciones.csv
# Prueba completa de recuperación en carpeta y entorno nuevos
bash scripts/recuperar.sh
```

**Tiempo y recursos medidos** (contenedor Linux x86-64, 2 vCPU, 7 GB RAM, sin GPU):
auditoría y particiones ≈ 30 s; entrenamiento E1 14 s, E2 14 s, E3 3 s, E4 7 s;
evaluación en test < 10 s. La prueba de recuperación registra su duración en `evidencias/recuperacion/entorno_nuevo/log_recuperacion.txt`.

## Limitaciones conocidas

- UNSW-NB15 es histórico y de laboratorio; no demuestra eficacia sobre una red actual.
- La FPR de test (17,5 %) supera ampliamente la de validación (4,8 %): el tráfico normal de test se distribuye distinto del de entrenamiento, y una validación extraída del mismo archivo no lo anticipa.
- 10,4 % de las filas de test repiten entradas vistas en train/val; las conclusiones sobre generalización se limitan en consecuencia.
- El umbral se eligió en validación, así que las métricas de validación al umbral elegido son optimistas.
- `model.joblib` es un pickle de scikit-learn 1.9.1: requiere esa versión y solo debe cargarse tras verificar su SHA-256 (lo hace `predict.py`).

## Declaración de uso de IA, código externo y contribución

- **IA:** se usó Claude (Anthropic) como asistente para estructurar el proyecto, escribir el código y la documentación y ejecutar los experimentos. Los commits generados con asistencia llevan la línea `Co-Authored-By: Claude`. Verifiqué: huellas SHA-256 frente al anexo, ausencia de grupos compartidos entre train y val, que test no participó en la selección (orden de commits), coincidencia de hashes entre MLflow, `models/` y Hugging Face, recálculo de métricas desde `predicciones-test.csv` y la prueba de recuperación en entorno nuevo.
- **Código externo:** solo bibliotecas listadas en `requirements.txt` (scikit-learn, pandas, MLflow, DVC, huggingface_hub, matplotlib, reportlab). No se adaptó código de terceros.
- **Datos:** UNSW-NB15 (Moustafa & Slay, 2015), obtenido del mirror fijado indicado en el enunciado.
- **Contribución:** entrega en pareja; Allison Loango y Kevin Nieto son responsables de las decisiones y del contenido.

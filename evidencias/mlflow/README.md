# Exportación de MLflow

Cuatro runs del experimento `defiende-el-servidor` (E1–E4), exportados con
`python -m src.exportar_mlflow` para consultarlos sin el servidor original.

| Archivo | Contenido |
|---|---|
| `runs.csv`, `runs.json` | Una fila por run: run ID, parámetros, métricas y tags (hipótesis, commit Git, md5 DVC, SHA-256 del modelo, conclusión) |
| `<run_id>/run.json` | Registro completo del run con el SHA-256 de cada artefacto |
| `<run_id>/artifacts/model.joblib` | Pipeline completo (preprocesamiento + estimador) registrado en el run |
| `<run_id>/artifacts/config.yaml` | Configuración exacta del experimento |
| `<run_id>/artifacts/scores_validacion.csv` | `row_id`, `y_true`, `score_attack` de las 35.067 filas de validación |
| `<run_id>/artifacts/barrido_umbrales_val.csv` | Métricas para umbrales 0,01–0,99 |
| `<run_id>/artifacts/metricas_validacion.json` | Métricas y matriz de confusión a 0,50 y al umbral operativo |
| `mlflow.db` | Copia del backend SQLite de MLflow |

## Cómo consultar

```bash
# Tabla en Python
python -c "import pandas as pd; print(pd.read_csv('evidencias/mlflow/runs.csv').filter(regex='run_id|experimento_id|val_oper').T)"

# Interfaz de MLflow (parámetros, métricas y tags; los artefactos están en las carpetas <run_id>/)
mlflow ui --backend-store-uri sqlite:///evidencias/mlflow/mlflow.db

# Recalcular las métricas de un run desde sus scores
python -c "
import pandas as pd; from src.metricas import evaluar
s = pd.read_csv('evidencias/mlflow/68d9aeb74e0d402d9209c10ca1a735da/artifacts/scores_validacion.csv')
print(evaluar(s.y_true, s.score_attack, 0.66))"
```

`mlflow.db` guarda la ubicación original de los artefactos (ruta del contenedor donde se
entrenó); por eso los artefactos se entregan copiados en `<run_id>/artifacts/`, con su SHA-256
en `run.json`. El `model.joblib` del run E1 es el mismo archivo publicado como v1 y el del run
E4 es el publicado como v2 (mismo SHA-256).

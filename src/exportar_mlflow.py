"""Exporta los runs de MLflow a evidencias/mlflow/ (consultable sin servidor).

Uso:  python -m src.exportar_mlflow

Estructura:
  evidencias/mlflow/runs.csv            una fila por run: info, params, métricas y tags
  evidencias/mlflow/runs.json           lo mismo en JSON
  evidencias/mlflow/<run_id>/run.json   info, params, métricas y tags del run
  evidencias/mlflow/<run_id>/artifacts/ artefactos registrados (modelo completo incluido)
  evidencias/mlflow/mlflow.db           copia del backend SQLite (para `mlflow ui`)
"""
from __future__ import annotations

import json
import shutil

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

from src.config import EVID, EXPERIMENTO_MLFLOW, MLFLOW_URI, RAIZ
from src.utils import sha256_archivo


def main() -> None:
    mlflow.set_tracking_uri(MLFLOW_URI)
    cliente = MlflowClient()
    exp = cliente.get_experiment_by_name(EXPERIMENTO_MLFLOW)
    runs = cliente.search_runs([exp.experiment_id], order_by=["tags.experimento_id ASC"])
    destino = EVID / "mlflow"
    destino.mkdir(parents=True, exist_ok=True)

    filas = []
    for r in runs:
        carpeta = destino / r.info.run_id
        if carpeta.exists():
            shutil.rmtree(carpeta)
        (carpeta / "artifacts").mkdir(parents=True)
        local = mlflow.artifacts.download_artifacts(run_id=r.info.run_id, dst_path=str(carpeta / "tmp"))
        for f in sorted((carpeta / "tmp").iterdir()):
            shutil.move(str(f), carpeta / "artifacts" / f.name)
        shutil.rmtree(carpeta / "tmp")
        artefactos = {f.name: sha256_archivo(f) for f in sorted((carpeta / "artifacts").iterdir())}
        registro = {
            "run_id": r.info.run_id,
            "run_name": r.info.run_name,
            "experiment_name": EXPERIMENTO_MLFLOW,
            "experiment_id": r.info.experiment_id,
            "status": r.info.status,
            "start_time_ms": r.info.start_time,
            "end_time_ms": r.info.end_time,
            "params": dict(sorted(r.data.params.items())),
            "metrics": dict(sorted(r.data.metrics.items())),
            "tags": {k: v for k, v in sorted(r.data.tags.items()) if not k.startswith("mlflow.")},
            "artifacts_sha256": artefactos,
        }
        (carpeta / "run.json").write_text(json.dumps(registro, indent=2, ensure_ascii=False) + "\n")
        fila = {k: registro[k] for k in ("run_id", "run_name", "status", "start_time_ms", "end_time_ms")}
        fila.update({f"param.{k}": v for k, v in registro["params"].items()})
        fila.update({f"metric.{k}": v for k, v in registro["metrics"].items()})
        fila.update({f"tag.{k}": v for k, v in registro["tags"].items()})
        filas.append(fila)
        print(r.data.tags["experimento_id"], r.info.run_id, len(artefactos), "artefactos")

    pd.DataFrame(filas).to_csv(destino / "runs.csv", index=False)
    (destino / "runs.json").write_text(json.dumps(filas, indent=2, ensure_ascii=False) + "\n")
    shutil.copyfile(RAIZ / "mlflow.db", destino / "mlflow.db")


if __name__ == "__main__":
    main()

"""Muestra fija de validación y comprobación de la recuperación de v1/v2.

  python -m src.recuperacion muestra
      Elige 20 filas fijas de validación (10 normales + 10 ataques, semilla 42) y las
      guarda en evidencias/recuperacion/entrada_20_val.csv.

  python -m src.recuperacion originales v1
      Predice esas filas con models/v1 (artefacto local antes de publicar) y guarda
      evidencias/recuperacion/predicciones_originales_v1.csv.

  python -m src.recuperacion verificar v1 <carpeta_descargada_de_HF> <carpeta_salida>
      Carga el modelo descargado de Hugging Face (por SHA), vuelve a extraer las 20 filas
      desde data/prepared/val.csv recuperado con DVC, predice y compara con las
      predicciones originales. Escribe predicciones_recuperadas_<v>.csv y
      comparacion_<v>.json. Sale con código 1 si algo no coincide.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DATA_PREP, EVID, OBJETIVO, RAIZ, SEMILLA
from src.utils import columnas_predictoras, guardar_json, sha256_archivo

DIR = EVID / "recuperacion"
ENTRADA = DIR / "entrada_20_val.csv"
TOLERANCIA = 1e-9   # diferencia absoluta máxima admitida en score_attack


def _predict_mod(carpeta: Path):
    spec = importlib.util.spec_from_file_location(f"predict_{carpeta.name}", carpeta / "predict.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _predecir(carpeta: Path, entrada: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    mod = _predict_mod(carpeta)
    modelo, cfg = mod.cargar()           # verifica SHA-256 de model.joblib
    salida = mod.predecir(entrada, modelo, cfg)
    salida.insert(1, "y_true", entrada[OBJETIVO].to_numpy())
    return salida, cfg


def muestra() -> None:
    val = pd.read_csv(DATA_PREP / "val.csv")
    rng = np.random.default_rng(SEMILLA)
    idx = []
    for clase in (0, 1):
        candidatos = val.index[val[OBJETIVO] == clase].to_numpy()
        idx.extend(rng.choice(candidatos, size=10, replace=False))
    sel = val.loc[sorted(idx), ["row_id"] + columnas_predictoras(val) + [OBJETIVO]]
    DIR.mkdir(parents=True, exist_ok=True)
    sel.to_csv(ENTRADA, index=False)
    print(f"{len(sel)} filas -> {ENTRADA.relative_to(RAIZ)}  sha256={sha256_archivo(ENTRADA)}")


def originales(version: str) -> None:
    entrada = pd.read_csv(ENTRADA)
    salida, cfg = _predecir(RAIZ / "models" / version, entrada)
    ruta = DIR / f"predicciones_originales_{version}.csv"
    salida.to_csv(ruta, index=False, float_format="%.17g")
    print(salida.to_string(index=False))


def verificar(version: str, carpeta_hf: str, carpeta_salida: str) -> None:
    carpeta_hf, out = Path(carpeta_hf), Path(carpeta_salida)
    out.mkdir(parents=True, exist_ok=True)
    guardada = pd.read_csv(ENTRADA)
    # Las filas se vuelven a extraer del val.csv recuperado con DVC (no de la copia guardada).
    val = pd.read_csv(DATA_PREP / "val.csv")
    entrada = val.set_index("row_id").loc[guardada.row_id].reset_index()
    entrada = entrada[guardada.columns]
    entradas_iguales = bool(entrada.reset_index(drop=True).equals(guardada))

    recuperada, cfg = _predecir(carpeta_hf, entrada)
    recuperada.to_csv(out / f"predicciones_recuperadas_{version}.csv", index=False,
                      float_format="%.17g")
    original = pd.read_csv(DIR / f"predicciones_originales_{version}.csv")
    diff = np.abs(original.score_attack.to_numpy() - recuperada.score_attack.to_numpy())
    resultado = {
        "version": version,
        "filas": int(len(recuperada)),
        "row_id_coinciden": bool((original.row_id.to_numpy() == recuperada.row_id.to_numpy()).all()),
        "entradas_dvc_coinciden_con_guardadas": entradas_iguales,
        "sha256_model_joblib_descargado": sha256_archivo(carpeta_hf / "model.joblib"),
        "sha256_model_joblib_esperado": cfg["sha256_model_joblib"],
        "umbral": cfg["umbral"],
        "mlflow_run_id": cfg["mlflow_run_id"],
        "y_pred_coinciden": int((original.y_pred.to_numpy() == recuperada.y_pred.to_numpy()).sum()),
        "max_abs_diff_score": float(diff.max()),
        "tolerancia": TOLERANCIA,
        "predicciones_ataque": int(recuperada.y_pred.sum()),
    }
    resultado["sha256_coincide"] = (resultado["sha256_model_joblib_descargado"]
                                    == resultado["sha256_model_joblib_esperado"])
    resultado["resultado"] = "OK" if (
        resultado["row_id_coinciden"] and entradas_iguales and resultado["sha256_coincide"]
        and resultado["y_pred_coinciden"] == len(recuperada)
        and resultado["max_abs_diff_score"] <= TOLERANCIA) else "FALLO"
    guardar_json(resultado, out / f"comparacion_{version}.json")
    print(json.dumps(resultado, indent=2))
    if resultado["resultado"] != "OK":
        sys.exit(1)


if __name__ == "__main__":
    accion = sys.argv[1]
    if accion == "muestra":
        muestra()
    elif accion == "originales":
        originales(sys.argv[2])
    elif accion == "verificar":
        verificar(*sys.argv[2:5])
    else:
        raise SystemExit(__doc__)

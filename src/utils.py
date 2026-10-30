"""Utilidades de trazabilidad: hashes, lectura de datos, Git y DVC."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd
import yaml

from src.config import CATEGORICAS, EXCLUIDAS, RAIZ


def sha256_archivo(ruta: Path | str, bloque: int = 1 << 20) -> str:
    """SHA-256 de un archivo leído por bloques."""
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        while chunk := f.read(bloque):
            h.update(chunk)
    return h.hexdigest()


def sha256_texto(texto: str) -> str:
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()


def leer_csv(ruta: Path | str) -> pd.DataFrame:
    # Los CSV del mirror empiezan con BOM; utf-8-sig lo elimina del nombre de 'id'.
    return pd.read_csv(ruta, encoding="utf-8-sig", low_memory=False)


def columnas_predictoras(df: pd.DataFrame) -> list[str]:
    """Variables de entrada: todo excepto id, attack_cat, label y columnas de auditoría."""
    auxiliares = set(EXCLUIDAS) | {"row_id", "grupo", "split"}
    return [c for c in df.columns if c not in auxiliares]


def columnas_numericas(df: pd.DataFrame) -> list[str]:
    return [c for c in columnas_predictoras(df) if c not in CATEGORICAS]


def git_commit(*rutas_requeridas: str) -> str:
    """Commit completo de HEAD.

    Falla si hay cambios sin confirmar en archivos versionados de src/, configs/ o
    requirements.txt, o si alguna ruta requerida (p. ej. la config del experimento)
    no está confirmada en Git. Así cada run queda ligado a código real del historial.
    """
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=RAIZ, text=True).strip()
    sucio = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no", "--",
         "src", "configs", "requirements.txt"],
        cwd=RAIZ, text=True,
    ).strip()
    for ruta in rutas_requeridas:
        if subprocess.run(["git", "ls-files", "--error-unmatch", ruta], cwd=RAIZ,
                          capture_output=True).returncode != 0:
            sucio += f"\n?? {ruta} (no está confirmado en Git)"
    sucio = sucio.strip()
    if sucio:
        raise RuntimeError(
            "Hay cambios sin confirmar en src/, configs/ o requirements.txt. "
            "Confirma antes de entrenar para que el run quede ligado a un commit real:\n" + sucio
        )
    return sha


def dvc_md5(archivo_dvc: str) -> str:
    """md5 que DVC registra en un archivo .dvc (identifica la versión de datos)."""
    with open(RAIZ / archivo_dvc) as f:
        meta = yaml.safe_load(f)
    return meta["outs"][0]["md5"]


def guardar_json(obj, ruta: Path | str) -> None:
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_serializar)
        f.write("\n")


def _serializar(o):
    if hasattr(o, "item"):
        return o.item()
    if isinstance(o, Path):
        return str(o.relative_to(RAIZ)) if o.is_absolute() else str(o)
    raise TypeError(f"No serializable: {type(o)}")

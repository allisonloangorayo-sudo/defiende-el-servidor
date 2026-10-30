"""Verifica que los datos recuperados con DVC tengan las huellas registradas.

Uso:  python -m src.verificar_datos
Compara el SHA-256 de los CSV originales con auditoria.json (y el anexo del enunciado)
y el de las particiones preparadas con data/prepared/splits.json. Sale con código 1 si
alguna huella no coincide.
"""
from __future__ import annotations

import json
import sys

from src.config import DATA_PREP, DATA_RAW, EVID, SHA256_ESPERADO
from src.utils import sha256_archivo


def main() -> None:
    aud = json.loads((EVID / "auditoria.json").read_text())
    splits = aud["particiones"]
    filas = []
    for nombre, esperado in SHA256_ESPERADO.items():
        filas.append({"archivo": f"data/raw/{nombre}", "esperado": esperado,
                      "obtenido": sha256_archivo(DATA_RAW / nombre)})
    for parte in ("train", "val", "test"):
        filas.append({"archivo": f"data/prepared/{parte}.csv",
                      "esperado": splits[parte]["sha256_archivo"],
                      "obtenido": sha256_archivo(DATA_PREP / f"{parte}.csv")})
    for f in filas:
        f["coincide"] = f["esperado"] == f["obtenido"]
    ok = all(f["coincide"] for f in filas)
    print(json.dumps({"archivos": filas, "resultado": "OK" if ok else "FALLO"}, indent=2))
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()

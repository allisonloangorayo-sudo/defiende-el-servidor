#!/usr/bin/env bash
# Prueba de recuperación en una carpeta y un entorno NUEVOS.
#
# No reutiliza nada del proyecto original: clona el código desde GitHub, crea un entorno
# virtual nuevo, recupera los datos con `dvc pull` desde el almacenamiento entregado,
# descarga v1 y v2 de Hugging Face por su SHA completo con una caché vacía y compara las
# predicciones sobre las 20 filas fijas de validación con las guardadas.
#
# Uso:  bash scripts/recuperar.sh [carpeta_destino] [revision_git]
#   carpeta_destino  por defecto ~/recuperacion-defiende-<fecha>
#   revision_git     commit/tag/rama a recuperar (por defecto main)
# Requiere: git y uv (https://docs.astral.sh/uv/). No requiere sesión iniciada si los
# repositorios son públicos.
set -euo pipefail

GH_URL="https://github.com/allisonloangorayo-sudo/defiende-el-servidor.git"
DEST="${1:-$HOME/recuperacion-defiende-$(date +%Y%m%d-%H%M%S)}"
REV="${2:-main}"
[ -e "$DEST" ] && { echo "La carpeta $DEST ya existe; usa una nueva."; exit 1; }
mkdir -p "$DEST"
DEST="$(cd "$DEST" && pwd)"
OUT="$DEST/salida"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/log_recuperacion.txt") 2>&1
set -x

# Cachés vacías y exclusivas de esta prueba (no se reutiliza ~/.cache ni la caché DVC original).
export HF_HOME="$DEST/hf-home"
export HF_HUB_CACHE="$DEST/hf-cache"
export UV_CACHE_DIR="$DEST/uv-cache"
export MLFLOW_DISABLE_AGENT_HINT=1
export PIP_NO_CACHE_DIR=1
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN || true
date -u +%Y-%m-%dT%H:%M:%SZ
uname -a

# 1. Código desde GitHub en la revisión elegida
git clone "$GH_URL" "$DEST/repo"
cd "$DEST/repo"
git checkout "$REV"
git log -1 --format='commit %H%n%s'
git tag --list

# 2. Entorno nuevo con dependencias fijadas
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python --version
.venv/bin/python -c "import sklearn, pandas, numpy; print('sklearn', sklearn.__version__, 'pandas', pandas.__version__, 'numpy', numpy.__version__)"
PY=.venv/bin/python

# 3. Datos con DVC desde el almacenamiento entregado (dataset de Hugging Face)
.venv/bin/dvc config --local cache.dir "$DEST/dvc-cache"
if ! .venv/bin/dvc pull -r hf; then
  # Alternativa documentada: descargar el almacenamiento fijado por SHA y usarlo como remote local.
  STORE_REPO=$($PY -c "import json;print(json.load(open('configs/referencias.json'))['dvc_remote']['hf_dataset_repo'].split('/datasets/')[1])")
  STORE_SHA=$($PY -c "import json;print(json.load(open('configs/referencias.json'))['dvc_remote']['hf_revision_sha'])")
  .venv/bin/hf download "$STORE_REPO" --repo-type dataset --revision "$STORE_SHA" --local-dir "$DEST/dvc-store"
  .venv/bin/dvc remote add --local store_descargado "$DEST/dvc-store"
  .venv/bin/dvc pull -r store_descargado
fi
.venv/bin/dvc status
$PY -m src.verificar_datos | tee "$OUT/verificacion_datos.json"

# 4-5. Descargar v1 y v2 por SHA completo, cargar modelo + umbral y predecir las 20 filas
for V in v1 v2; do
  SHA=$($PY -c "import json;print(json.load(open('configs/referencias.json'))['$V']['hf_revision_sha'])")
  REPO=$($PY -c "import json;print(json.load(open('configs/referencias.json'))['hf_model_repo'].split('huggingface.co/')[1])")
  echo "Descargando $REPO @ $SHA ($V)"
  .venv/bin/hf download "$REPO" --revision "$SHA" --local-dir "$DEST/modelos/$V"
  ls -la "$DEST/modelos/$V"
  $PY -m src.recuperacion verificar "$V" "$DEST/modelos/$V" "$OUT"
done

# 6. Resumen con hashes de entradas y salidas
$PY - "$OUT" <<'EOF'
import hashlib, json, pathlib, subprocess, sys
out = pathlib.Path(sys.argv[1])
h = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
res = {v: json.loads((out / f"comparacion_{v}.json").read_text()) for v in ("v1", "v2")}
resumen = {
    "commit_recuperado": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "entrada": {"archivo": "evidencias/recuperacion/entrada_20_val.csv",
                "sha256": h("evidencias/recuperacion/entrada_20_val.csv")},
    "salidas": {f.name: h(f) for f in sorted(out.glob("predicciones_recuperadas_*.csv"))},
    "originales": {f"predicciones_originales_{v}.csv": h(f"evidencias/recuperacion/predicciones_originales_{v}.csv") for v in ("v1", "v2")},
    "resultado_v1": res["v1"]["resultado"], "resultado_v2": res["v2"]["resultado"],
    "resultado_global": "OK" if all(r["resultado"] == "OK" for r in res.values()) else "FALLO",
}
(out / "resumen_recuperacion.json").write_text(json.dumps(resumen, indent=2) + "\n")
print(json.dumps(resumen, indent=2))
EOF
set +x
echo "Evidencias en: $OUT"
echo "Cópialas al proyecto con: cp -R \"$OUT\"/. evidencias/recuperacion/entorno_nuevo/"

#!/usr/bin/env bash
# Publica el proyecto: código en GitHub, datos en el remote DVC (dataset de Hugging Face)
# y las revisiones v1/v2 del modelo en Hugging Face Hub.
#
# Requisitos (ver README, sección "Publicación"):
#   - entorno creado con:  uv venv --python 3.13 && uv pip install -r requirements.txt
#   - sesión iniciada:     gh auth login     y     hf auth login   (token con permiso de escritura)
#   - repositorio vacío creado en GitHub (este script lo crea con gh si no existe)
#
# Uso, desde la raíz del repositorio:   bash scripts/publicar.sh
# No lee, imprime ni guarda tokens.
set -euo pipefail
cd "$(dirname "$0")/.."

GH_REPO="allisonloangorayo-sudo/defiende-el-servidor"
PY=".venv/bin/python"
[ -x "$PY" ] || { echo "Falta .venv: ejecuta 'uv venv --python 3.13 && uv pip install -r requirements.txt'"; exit 1; }
export MLFLOW_DISABLE_AGENT_HINT=1
LOG="build/publicacion.log"; mkdir -p build
exec > >(tee -a "$LOG") 2>&1
echo "== Publicación $(date -u +%Y-%m-%dT%H:%M:%SZ) =="

echo "-- 1. Comprobaciones previas"
.venv/bin/hf auth whoami
gh auth status >/dev/null 2>&1 || { echo "Inicia sesión en GitHub: gh auth login"; exit 1; }
for v in v1 v2; do
  "$PY" - "$v" <<'EOF'
import json, sys, hashlib, pathlib
v = sys.argv[1]
cfg = json.loads(pathlib.Path(f"models/{v}/config.json").read_text())
sha = hashlib.sha256(pathlib.Path(f"models/{v}/model.joblib").read_bytes()).hexdigest()
assert sha == cfg["sha256_model_joblib"], f"{v}: model.joblib no coincide con config.json"
print(f"{v}: model.joblib OK ({sha})")
EOF
done
.venv/bin/dvc checkout -q

echo "-- 2. Código en GitHub (historial completo y tags v1/v2)"
if ! gh repo view "$GH_REPO" >/dev/null 2>&1; then
  gh repo create "$GH_REPO" --public --description "MLOps 2026-II · Defiende el servidor (UNSW-NB15) · Git, DVC, MLflow y Hugging Face"
fi
git remote get-url origin >/dev/null 2>&1 || git remote add origin "https://github.com/$GH_REPO.git"
git push -u origin main --follow-tags

echo "-- 3. Datos: dvc push a una carpeta temporal y subida al dataset de Hugging Face"
STAGE="$(mktemp -d)"
.venv/bin/dvc remote add --local -f staging "$STAGE"
.venv/bin/dvc push -r staging
"$PY" -m src.publicar_hf dvc "$STAGE"
.venv/bin/dvc remote remove --local staging

echo "-- 4. Modelos v1 y v2 en Hugging Face (cada uno es un commit; se verifica la descarga por SHA)"
"$PY" -m src.publicar_hf modelos

echo "-- 5. Model card de v2 con el resultado de test (no cambia model.joblib)"
"$PY" -m src.publicar_hf card_test

echo "-- 6. Registrar las referencias en Git"
"$PY" -m src.registrar_referencias
git add configs/referencias.json evidencias/seleccion.json evidencias/metricas-test.json entrega.md
git --no-pager diff --cached --stat
read -r -p "¿Confirmar el commit con las referencias de Hugging Face? [s/N] " ok
if [[ "$ok" =~ ^[sS]$ ]]; then
  git commit -m "Referencias de publicación: SHA de Hugging Face (v1, v2) y almacenamiento DVC"
  git push origin main
fi
echo "== Listo. Siguiente paso: bash scripts/recuperar.sh =="

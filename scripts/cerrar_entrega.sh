#!/usr/bin/env bash
# Cierra la entrega después de la prueba de recuperación:
#   1. copia las evidencias de recuperación al proyecto
#   2. regenera entrega.md e informe.pdf con los SHA reales
#   3. confirma y sube el commit final (pide confirmación)
#   4. arma build/defiende-el-servidor-entrega.zip desde ese commit (git archive: sin datos,
#      entornos, cachés ni credenciales) e imprime el texto para el mensaje de entrega
#
# Uso, desde la raíz del repositorio:  bash scripts/cerrar_entrega.sh <carpeta_salida_de_recuperar.sh>
set -euo pipefail
cd "$(dirname "$0")/.."
SALIDA="${1:?Indica la carpeta 'salida' que imprimió scripts/recuperar.sh}"
PY=".venv/bin/python"
export MLFLOW_DISABLE_AGENT_HINT=1

grep -q '"resultado_global": "OK"' "$SALIDA/resumen_recuperacion.json" || {
  echo "La recuperación no terminó en OK; revisa $SALIDA/log_recuperacion.txt"; exit 1; }
mkdir -p evidencias/recuperacion/entorno_nuevo
cp -R "$SALIDA"/. evidencias/recuperacion/entorno_nuevo/

"$PY" -m src.registrar_referencias
"$PY" -m src.informe

git add evidencias/recuperacion/entorno_nuevo entrega.md informe.pdf evidencias/figuras
git --no-pager diff --cached --stat
read -r -p "¿Confirmar el commit final de la entrega? [s/N] " ok
if [[ "$ok" =~ ^[sS]$ ]]; then
  git commit -m "Prueba de recuperación en entorno nuevo (v1 y v2 OK) e informe final"
  git push origin main
fi

mkdir -p build
rm -f build/defiende-el-servidor-entrega.zip
git archive --format=zip --prefix=proyecto/ -o build/defiende-el-servidor-entrega.zip HEAD
FINAL=$(git rev-parse HEAD)
REFS=configs/referencias.json
cat <<EOF

================ Para el mensaje de entrega ================
Asunto: MLOps — Defiende el servidor — Grupo A — Allison Loango y Kevin Nieto
Integrantes: Allison Loango y Kevin Nieto · Grupo A
Repositorio de código: https://github.com/allisonloangorayo-sudo/defiende-el-servidor
Commit final: $FINAL
Repositorio de modelos: $("$PY" -c "import json;print(json.load(open('$REFS'))['hf_model_repo'])")
SHA v1: $("$PY" -c "import json;print(json.load(open('$REFS'))['v1']['hf_revision_sha'])")
SHA v2: $("$PY" -c "import json;print(json.load(open('$REFS'))['v2']['hf_revision_sha'])")
Almacenamiento DVC: $("$PY" -c "import json;d=json.load(open('$REFS'))['dvc_remote'];print(d['hf_dataset_repo'], '· revisión', d['hf_revision_sha'])")
Acceso: repositorios públicos; dvc pull no requiere credenciales.
Adjuntos: build/defiende-el-servidor-entrega.zip e informe.pdf
============================================================
EOF

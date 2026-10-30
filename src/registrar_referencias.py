"""Genera entrega.md (índice de la entrega) desde las evidencias y configs/referencias.json.

Uso:  python -m src.registrar_referencias
Se ejecuta de nuevo después de publicar para incluir los SHA de Hugging Face.
"""
from __future__ import annotations

import json
import subprocess

import pandas as pd
import yaml

from src.config import EVID, RAIZ

PENDIENTE = "PENDIENTE (se completa al ejecutar scripts/publicar.sh)"


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=RAIZ, text=True).strip()


def main() -> None:
    refs_path = RAIZ / "configs" / "referencias.json"
    refs = json.loads(refs_path.read_text()) if refs_path.exists() else {}
    exp = pd.read_csv(EVID / "experimentos.csv").set_index("experimento_id")
    sel = json.loads((EVID / "seleccion.json").read_text())
    test = json.loads((EVID / "metricas-test.json").read_text())
    cfg_v1 = yaml.safe_load((RAIZ / "configs" / "v1.yaml").read_text())
    gh = cfg_v1["github_repo"]
    commit_raw = git("log", "-1", "--format=%H", "--", "data/raw.dvc")
    commit_prep = git("log", "-1", "--format=%H", "--", "data/prepared.dvc")
    md5_raw = yaml.safe_load((RAIZ / "data" / "raw.dvc").read_text())["outs"][0]["md5"]
    md5_prep = yaml.safe_load((RAIZ / "data" / "prepared.dvc").read_text())["outs"][0]["md5"]
    hf_repo = refs.get("hf_model_repo", "https://huggingface.co/" + cfg_v1["hf_repo"])
    dvc = refs.get("dvc_remote", {})

    bloques = []
    for v in ("v1", "v2"):
        c = json.loads((RAIZ / "models" / v / "config.json").read_text())
        r = refs.get(v, {})
        bloques.append(f"""### {v} — {yaml.safe_load((RAIZ / 'configs' / f'{v}.yaml').read_text())['titulo']}

- Commit del código que entrenó el modelo: `{c['git_commit_entrenamiento']}`
- Commit con la config de la versión: `{c['git_commit_empaquetado']}` · tag Git `{v}` → `{git('rev-list', '-n', '1', v)}`
- Datos DVC: original `data/raw.dvc` md5 `{c['dvc_raw_md5']}` (commit `{commit_raw}`); preparados `data/prepared.dvc` md5 `{c['dvc_prepared_md5']}` (commit `{commit_prep}`)
- Experimento y run de MLflow: {c['experimento']} · `{c['mlflow_run_id']}`
- Hugging Face: {hf_repo} · revisión `{r.get('hf_revision_sha', PENDIENTE)}` (tag `{v}`)
- SHA-256 de `model.joblib`: `{c['sha256_model_joblib']}`
- Umbral: `{c['umbral']}` (ataque si `score_attack >= umbral`)
""")
    card_test = refs.get("v2", {}).get("hf_revision_card_test_sha")
    filas_exp = "\n".join(
        f"| {i} | {r.familia} | `{r.mlflow_run_id}` | {r.umbral_u050:.2f} / {r.umbral_oper:.2f} | "
        f"{r.recall_ataque_oper:.4f} | {r.precision_ataque_oper:.4f} | {r.fpr_oper:.4f} | {r.macro_f1_oper:.4f} |"
        for i, r in exp.iterrows())
    seg = exp.segundos_entrenamiento.to_dict()
    texto = f"""# Entrega · MLOps — Defiende el servidor — Grupo A — Allison Loango

## Integrantes y repositorio

- **Integrante:** Allison Loango · Grupo A (entrega individual)
- **Repositorio de código (GitHub):** {gh}
- **Commit final de la entrega:** el último commit de `main` (se informa completo en el mensaje de entrega; `git rev-parse HEAD`).
- **Repositorio de modelos (Hugging Face):** {hf_repo}
- **Almacenamiento DVC:** {dvc.get('hf_dataset_repo', 'https://huggingface.co/datasets/allisonloango/defiende-el-servidor-dvc')} · revisión `{dvc.get('hf_revision_sha', PENDIENTE)}`
  - Remote `hf` en `.dvc/config` (HTTP de solo lectura): `{dvc.get('url', 'https://huggingface.co/datasets/allisonloango/defiende-el-servidor-dvc/resolve/main')}`
  - Acceso: repositorios públicos; `dvc pull` no requiere credenciales.

## Versiones del modelo

{chr(10).join(bloques)}
La model card de v2 se actualizó después de test en la revisión `{card_test or PENDIENTE}` (tag `v2-test`);
esa revisión solo cambia `README.md` y conserva el mismo `model.joblib` (SHA-256 verificado al publicar).

## Experimentos (validación, misma partición)

| ID | Familia | Run ID MLflow | Umbral 0,50 / operativo | Recall | Precision | FPR | Macro-F1 |
|---|---|---|---|---|---|---|---|
{filas_exp}

Métricas al umbral operativo (máximo recall con FPR <= 5 %). Las métricas a 0,50 están en `evidencias/experimentos.csv`.

## Decisión y test

- Elegida: **v2 = {sel['experimento']}** con umbral **{sel['umbral']}**, solo con validación (`evidencias/seleccion.json`).
- Congelación: commit de selección `{git('log', '-1', '--format=%H', '--', 'configs/v2.yaml')}` y tag `v2` → `{git('rev-list', '-n', '1', 'v2')}`, ambos anteriores al commit de evaluación `{git('log', '-1', '--format=%H', '--', 'evidencias/predicciones-test.csv')}`.
- Test (una sola vez, {test['filas_evaluadas']} filas): recall {test['metricas']['recall_ataque']:.4f} · precision {test['metricas']['precision_ataque']:.4f} · FPR {test['metricas']['fpr']:.4f} · macro-F1 {test['metricas']['macro_f1']:.4f}.

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
git clone {gh}.git && cd defiende-el-servidor
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
auditoría y particiones ≈ 30 s; entrenamiento E1 {seg['E1']:.0f} s, E2 {seg['E2']:.0f} s, E3 {seg['E3']:.0f} s, E4 {seg['E4']:.0f} s;
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
- **Contribución:** entrega individual; Allison Loango es responsable de todas las decisiones y del contenido.
"""
    (RAIZ / "entrega.md").write_text(texto)
    print("entrega.md actualizado")


if __name__ == "__main__":
    main()

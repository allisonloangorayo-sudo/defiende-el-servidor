"""Publicación en Hugging Face Hub de v1, v2 y del almacenamiento DVC.

Se ejecuta desde scripts/publicar.sh con una sesión iniciada (`hf auth login`).
Nunca lee ni escribe tokens: usa la credencial que huggingface_hub ya tiene guardada.

  python -m src.publicar_hf modelos   -> sube models/v1 y models/v2 (dos commits), crea tags,
                                         verifica la descarga por SHA y escribe
                                         configs/referencias.json
  python -m src.publicar_hf dvc <dir> -> sube el almacenamiento DVC (contenido de `dvc push`)
                                         a un repositorio de tipo dataset
  python -m src.publicar_hf card_test -> añade a la model card de v2 el resultado de test,
                                         citando el SHA evaluado (model.joblib no cambia)
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import yaml
from huggingface_hub import HfApi, hf_hub_download

from src.config import EVID, RAIZ
from src.utils import guardar_json, sha256_archivo

REFS = RAIZ / "configs" / "referencias.json"


def _cfg(version: str) -> dict:
    with open(RAIZ / "configs" / f"{version}.yaml") as f:
        return yaml.safe_load(f)


def _refs() -> dict:
    return json.loads(REFS.read_text()) if REFS.exists() else {}


def _verificar(repo: str, sha: str, esperado: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        ruta = hf_hub_download(repo, "model.joblib", revision=sha, cache_dir=tmp)
        obtenido = sha256_archivo(ruta)
    if obtenido != esperado:
        raise SystemExit(f"Descarga de {repo}@{sha}: SHA-256 {obtenido} != {esperado}")


def modelos() -> None:
    api = HfApi()
    usuario = api.whoami()["name"]
    repo = _cfg("v1")["hf_repo"]
    if not repo.startswith(usuario + "/"):
        raise SystemExit(f"La sesión de Hugging Face es '{usuario}', pero el repo es {repo}.")
    api.create_repo(repo, repo_type="model", exist_ok=True)
    refs = _refs()
    refs["hf_model_repo"] = f"https://huggingface.co/{repo}"
    for version in ("v1", "v2"):
        carpeta = RAIZ / "models" / version
        cfg = json.loads((carpeta / "config.json").read_text())
        assert sha256_archivo(carpeta / "model.joblib") == cfg["sha256_model_joblib"]
        info = api.upload_folder(
            repo_id=repo, folder_path=str(carpeta), repo_type="model",
            commit_message=f"{version}: {cfg['experimento']} · run {cfg['mlflow_run_id']} · "
                           f"umbral {cfg['umbral']} · sha256 {cfg['sha256_model_joblib'][:12]}",
        )
        sha = info.oid
        _verificar(repo, sha, cfg["sha256_model_joblib"])
        api.create_tag(repo, tag=version, revision=sha, exist_ok=True)
        refs[version] = {"hf_revision_sha": sha, "sha256_model_joblib": cfg["sha256_model_joblib"],
                         "umbral": cfg["umbral"], "mlflow_run_id": cfg["mlflow_run_id"],
                         "git_tag": version}
        print(f"{version}: {repo}@{sha}  (descarga verificada)")
    guardar_json(refs, REFS)

    # Completar las referencias de Hugging Face en la decisión congelada (sin tocar modelo/umbral).
    for nombre in ("seleccion.json", "metricas-test.json"):
        ruta = EVID / nombre
        d = json.loads(ruta.read_text())
        d["hugging_face"] = {"repo": repo, "revision_v2_sha": refs["v2"]["hf_revision_sha"],
                             "tag": "v2", "nota": "Completado tras la publicación; el artefacto "
                             "es el mismo (sha256_model_joblib)."}
        guardar_json(d, ruta)


def dvc(carpeta: str) -> None:
    api = HfApi()
    with open(RAIZ / ".dvc" / "config") as f:
        url = next(l.split("=", 1)[1].strip() for l in f if l.strip().startswith("url"))
    repo = url.split("/datasets/")[1].split("/resolve/")[0]
    api.create_repo(repo, repo_type="dataset", exist_ok=True)
    readme = Path(carpeta) / "README.md"
    readme.write_text(
        "---\nlicense: other\n---\n# Almacenamiento DVC · Defiende el servidor\n\n"
        "Contenido de `dvc push` (direccionado por md5) para el proyecto académico "
        "*Defiende el servidor*. No se usa directamente: se recupera con `dvc pull` desde el "
        "repositorio de código.\n\nDatos: UNSW-NB15 (Moustafa & Slay, 2015), "
        "https://research.unsw.edu.au/projects/unsw-nb15-dataset — uso académico.\n")
    info = api.upload_folder(repo_id=repo, folder_path=carpeta, repo_type="dataset",
                             commit_message="Almacenamiento DVC: datos originales y preparados")
    refs = _refs()
    refs["dvc_remote"] = {"url": url, "hf_dataset_repo": f"https://huggingface.co/datasets/{repo}",
                          "hf_revision_sha": info.oid}
    guardar_json(refs, REFS)
    print(f"DVC store: {repo}@{info.oid}")


def card_test() -> None:
    api = HfApi()
    refs = _refs()
    repo = _cfg("v2")["hf_repo"]
    sha_v2 = refs["v2"]["hf_revision_sha"]
    m = json.loads((EVID / "metricas-test.json").read_text())
    cm, met = m["matriz_confusion"], m["metricas"]
    texto = (
        "\n## Evaluación final en test (una sola vez)\n\n"
        f"Revisión evaluada: `{sha_v2}` · SHA-256 de `model.joblib`: `{m['sha256_model_joblib']}` "
        f"· umbral `{m['umbral']}`. Esta actualización solo cambia la documentación; el modelo "
        "congelado es el mismo archivo.\n\n"
        f"Test completo: {m['filas_evaluadas']} filas (`UNSW_NB15_testing-set.csv`, sha256 "
        f"`{m['dataset']['sha256_original']}`).\n\n"
        "| Métrica | Test |\n|---|---|\n"
        f"| Recall de ataque | {met['recall_ataque']:.4f} |\n"
        f"| Precision de ataque | {met['precision_ataque']:.4f} |\n"
        f"| FPR | {met['fpr']:.4f} |\n| Macro-F1 | {met['macro_f1']:.4f} |\n\n"
        f"Matriz de confusión: TN {cm['tn']} · FP {cm['fp']} · FN {cm['fn']} · TP {cm['tp']}.\n\n"
        "La FPR en test es mucho mayor que en validación (0,048): el tráfico normal del archivo "
        "de test difiere del de entrenamiento, y el umbral calibrado en validación no conserva el "
        "presupuesto de falsas alarmas. Se reporta sin ajustar el modelo ni el umbral.\n"
    )
    with tempfile.TemporaryDirectory() as tmp:
        card = Path(hf_hub_download(repo, "README.md", revision=sha_v2, cache_dir=tmp)).read_text()
        if "## Evaluación final en test" in card:
            raise SystemExit("La model card ya tiene el resultado de test.")
        nueva = card.replace("\n## Limitaciones", texto + "\n## Limitaciones", 1)
        ruta = Path(tmp) / "README.md"
        ruta.write_text(nueva)
        info = api.upload_file(path_or_fileobj=str(ruta), path_in_repo="README.md", repo_id=repo,
                               commit_message=f"Model card v2: resultado de test de la revisión {sha_v2[:12]}")
    _verificar(repo, info.oid, refs["v2"]["sha256_model_joblib"])
    api.create_tag(repo, tag="v2-test", revision=info.oid, exist_ok=True)
    refs["v2"]["hf_revision_card_test_sha"] = info.oid
    guardar_json(refs, REFS)
    print(f"Model card v2 actualizada: {repo}@{info.oid} (model.joblib sin cambios)")


if __name__ == "__main__":
    accion = sys.argv[1]
    if accion == "modelos":
        modelos()
    elif accion == "dvc":
        dvc(sys.argv[2])
    elif accion == "card_test":
        card_test()
    else:
        raise SystemExit(__doc__)

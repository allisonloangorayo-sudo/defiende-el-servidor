"""Compara los cuatro runs en validación, aplica la regla fijada y congela v2.

Uso:  python -m src.seleccionar

Salidas:
  evidencias/experimentos.csv     una fila por configuración (ambos umbrales)
  evidencias/seleccion.json       decisión congelada antes de evaluar test
  evidencias/figuras/*.png        curvas recall–FPR y comparación de métricas
  configs/v2.yaml                 config de la versión 2 (modelo + umbral elegidos)

Solo usa los scores de validación guardados en cada run. Nunca lee test.
"""
from __future__ import annotations

import datetime as dt
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import yaml
from mlflow.tracking import MlflowClient

from src.config import EVID, EXPERIMENTO_MLFLOW, MLFLOW_URI, RAIZ
from src.metricas import evaluar

COLORES = {"E1": "#9aa5b1", "E2": "#52606d", "E3": "#3f7cac", "E4": "#c2410c"}


def cargar_runs() -> list:
    mlflow.set_tracking_uri(MLFLOW_URI)
    exp = mlflow.get_experiment_by_name(EXPERIMENTO_MLFLOW)
    runs = MlflowClient().search_runs([exp.experiment_id], order_by=["tags.experimento_id ASC"])
    ids = [r.data.tags["experimento_id"] for r in runs]
    assert sorted(ids) == ["E1", "E2", "E3", "E4"], f"Se esperaban 4 runs únicos, hay {ids}"
    return sorted(runs, key=lambda r: r.data.tags["experimento_id"])


def tabla(runs, conclusiones) -> pd.DataFrame:
    filas = []
    for r in runs:
        t, p, m = r.data.tags, r.data.params, r.data.metrics
        params_modelo = {k.replace("modelo__", ""): v for k, v in p.items() if k.startswith("modelo__")}
        fila = {
            "experimento_id": t["experimento_id"],
            "hipotesis": t["hipotesis"],
            "cambio": t["cambio"],
            "familia": t["familia"],
            "estimador": p["estimador"],
            "preprocesamiento": f"num={p['pre_numericas']}; cat={p['pre_categoricas']}",
            "parametros": json.dumps(params_modelo, ensure_ascii=False),
            "config": t["config_path"],
            "semilla": int(p["semilla"]),
            "git_commit": t["git_commit"],
            "dvc_raw_md5": t["dvc_raw_md5"],
            "dvc_prepared_md5": t["dvc_prepared_md5"],
            "mlflow_run_id": r.info.run_id,
            "sha256_model_joblib": t["sha256_model_joblib"],
        }
        for pref, nombre in (("val_u050", "u050"), ("val_oper", "oper")):
            fila[f"umbral_{nombre}"] = m[f"{pref}_umbral"]
            for k in ("recall_ataque", "precision_ataque", "fpr", "macro_f1"):
                fila[f"{k}_{nombre}"] = round(m[f"{pref}_{k}"], 6)
            for k in ("tn", "fp", "fn", "tp"):
                fila[f"{k}_{nombre}"] = int(m[f"{pref}_{k}"])
        fila["segundos_entrenamiento"] = m["segundos_entrenamiento"]
        fila["conclusion_validacion"] = " ".join(conclusiones[t["experimento_id"]].split())
        filas.append(fila)
    return pd.DataFrame(filas)


def figuras(runs, regla, elegido, v1) -> None:
    carpeta = EVID / "figuras"
    carpeta.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=200)
    for r in runs:
        eid = r.data.tags["experimento_id"]
        b = pd.read_csv(mlflow.artifacts.download_artifacts(run_id=r.info.run_id,
                                                            artifact_path="barrido_umbrales_val.csv"))
        ax.plot(b.fpr, b.recall_ataque, color=COLORES[eid], lw=1.6,
                label=f"{eid} · {r.data.params['estimador'].replace('Classifier', '')}")
    ax.axvline(regla["fpr_maximo"], color="#111827", ls="--", lw=0.9)
    ax.text(regla["fpr_maximo"] + 0.003, 0.80, "FPR máx. 5 %", fontsize=8)
    ax.scatter([v1["fpr"]], [v1["recall_ataque"]], marker="o", s=40, color=COLORES["E1"],
               edgecolor="black", zorder=5)
    ax.annotate("v1 (E1, u=0,50)", (v1["fpr"], v1["recall_ataque"]), xytext=(-10, -18),
                textcoords="offset points", fontsize=8, ha="right")
    ax.scatter([elegido["fpr"]], [elegido["recall_ataque"]], marker="*", s=120,
               color=COLORES["E4"], edgecolor="black", zorder=5)
    ax.annotate(f"v2 (E4, u={elegido['umbral']:.2f})".replace(".", ","),
                (elegido["fpr"], elegido["recall_ataque"]), xytext=(8, -14),
                textcoords="offset points", fontsize=8)
    ax.set_xlim(0, 0.25)
    ax.set_ylim(0.78, 1.0)
    ax.set_xlabel("Tasa de falsos positivos (validación)")
    ax.set_ylabel("Recall de ataque (validación)")
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(carpeta / "curvas_recall_fpr_validacion.png")
    plt.close(fig)


def main() -> None:
    with open(RAIZ / "configs" / "seleccion.yaml") as f:
        regla = yaml.safe_load(f)
    with open(RAIZ / "configs" / "conclusiones.yaml") as f:
        conclusiones = yaml.safe_load(f)
    with open(RAIZ / "configs" / "v1.yaml") as f:
        cfg_v1 = yaml.safe_load(f)

    runs = cargar_runs()
    df = tabla(runs, conclusiones)
    df.to_csv(EVID / "experimentos.csv", index=False)

    # Regla: cada run aporta su mejor punto con FPR <= máx.; gana el mayor recall, luego macro-F1.
    candidatos = df[df.fpr_oper <= regla["fpr_maximo"]]
    ganador = candidatos.sort_values(["recall_ataque_oper", "macro_f1_oper"],
                                     ascending=False).iloc[0]
    run_g = next(r for r in runs if r.info.run_id == ganador.mlflow_run_id)
    scores = pd.read_csv(mlflow.artifacts.download_artifacts(
        run_id=run_g.info.run_id, artifact_path="scores_validacion.csv"))
    m_v2 = evaluar(scores.y_true, scores.score_attack, float(ganador.umbral_oper))
    run_v1 = next(r for r in runs if r.info.run_id == cfg_v1["mlflow_run_id"])
    s_v1 = pd.read_csv(mlflow.artifacts.download_artifacts(
        run_id=run_v1.info.run_id, artifact_path="scores_validacion.csv"))
    m_v1 = evaluar(s_v1.y_true, s_v1.score_attack, float(cfg_v1["umbral"]))

    # Registrar la conclusión en cada run de MLflow.
    cliente = MlflowClient()
    for r in runs:
        cliente.set_tag(r.info.run_id, "conclusion_validacion",
                        " ".join(conclusiones[r.data.tags["experimento_id"]].split()))
        cliente.set_tag(r.info.run_id, "seleccionado_v2",
                        str(r.info.run_id == ganador.mlflow_run_id).lower())

    def d(a, b):  # diferencia en puntos porcentuales con signo
        return f"{(b - a) * 100:+.1f} pp"

    comparacion = (
        "## Comparación con v1 (validación)\n\n"
        "| | v1 (E1, u=0,50) | v2 (E4, u={u}) | Cambio |\n|---|---|---|---|\n"
        f"| Recall de ataque | {m_v1['recall_ataque']:.4f} | {m_v2['recall_ataque']:.4f} | {d(m_v1['recall_ataque'], m_v2['recall_ataque'])} |\n"
        f"| Precision de ataque | {m_v1['precision_ataque']:.4f} | {m_v2['precision_ataque']:.4f} | {d(m_v1['precision_ataque'], m_v2['precision_ataque'])} |\n"
        f"| FPR | {m_v1['fpr']:.4f} | {m_v2['fpr']:.4f} | {d(m_v1['fpr'], m_v2['fpr'])} |\n"
        f"| Macro-F1 | {m_v1['macro_f1']:.4f} | {m_v2['macro_f1']:.4f} | {d(m_v1['macro_f1'], m_v2['macro_f1'])} |\n"
        f"| Falsos positivos | {m_v1['matriz_confusion']['fp']} | {m_v2['matriz_confusion']['fp']} | |\n"
        f"| Ataques no detectados (FN) | {m_v1['matriz_confusion']['fn']} | {m_v2['matriz_confusion']['fn']} | |\n\n"
        "**Qué mejora:** las falsas alarmas bajan de "
        f"{m_v1['matriz_confusion']['fp']} a {m_v2['matriz_confusion']['fp']} y la precision y el macro-F1 suben. "
        "**Qué empeora:** el recall baja "
        f"{abs(m_v2['recall_ataque'] - m_v1['recall_ataque']) * 100:.1f} pp: se aceptan "
        f"{m_v2['matriz_confusion']['fn'] - m_v1['matriz_confusion']['fn']} ataques no detectados más "
        "(sobre todo Fuzzers) a cambio de respetar el presupuesto de FPR <= 5 %."
    ).replace("{u}", f"{ganador.umbral_oper:.2f}".replace(".", ","))

    cfg_v2 = {
        "version": "v2",
        "titulo": f"gradient boosting (E4) con umbral {ganador.umbral_oper:.2f} elegido en validación".replace(".", ","),
        "experimento": ganador.experimento_id,
        "mlflow_run_id": ganador.mlflow_run_id,
        "umbral": float(ganador.umbral_oper),
        "razon": (
            "Cambio de modelo (regresión logística -> HistGradientBoosting, preprocesamiento "
            "ordinal sin escalado) y de umbral (0,50 -> "
            f"{ganador.umbral_oper:.2f}".replace(".", ",") + "). Aplicando la regla fijada en configs/seleccion.yaml "
            "(máximo recall de ataque con FPR <= 5 % en validación), E4 obtiene el mayor recall "
            "entre las cuatro configuraciones."),
        "proposito": (
            "Versión revisada para un equipo de seguridad con presupuesto de falsas alarmas: "
            "maximiza la detección de ataques manteniendo la tasa de falsos positivos en "
            "validación por debajo de 5 %."),
        "comparacion": comparacion,
        "hf_repo": cfg_v1["hf_repo"],
        "github_repo": cfg_v1["github_repo"],
        "dvc_store": cfg_v1["dvc_store"],
    }
    with open(RAIZ / "configs" / "v2.yaml", "w") as f:
        f.write("# Versión 2: generada por src/seleccionar.py a partir de validación (no editar a mano).\n")
        yaml.safe_dump(cfg_v2, f, allow_unicode=True, sort_keys=False, width=100)

    seleccion = {
        "version_elegida": "v2",
        "fecha_decision_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "regla": regla,
        "experimento": ganador.experimento_id,
        "mlflow_run_id": ganador.mlflow_run_id,
        "estimador": ganador.estimador,
        "umbral": float(ganador.umbral_oper),
        "sha256_model_joblib": ganador.sha256_model_joblib,
        "git_commit_entrenamiento": ganador.git_commit,
        "dvc": {"data/raw.dvc": ganador.dvc_raw_md5, "data/prepared.dvc": ganador.dvc_prepared_md5},
        "hugging_face": {"repo": cfg_v1["hf_repo"], "revision_v2_sha": "PENDIENTE_PUBLICACION",
                         "nota": "Se completa con scripts/publicar.sh; el artefacto se identifica por sha256_model_joblib."},
        "metricas_validacion": m_v2,
        "metricas_validacion_v1": m_v1,
        "candidatos": df[["experimento_id", "umbral_oper", "recall_ataque_oper", "fpr_oper",
                          "macro_f1_oper"]].to_dict("records"),
        "razon": (
            "Ataque no detectado (FN) es el error más costoso; las falsas alarmas tienen un "
            f"presupuesto (FPR <= {regla['fpr_maximo']:.0%}). Entre los puntos que cumplen el "
            f"presupuesto, E4 con umbral {ganador.umbral_oper:.2f} maximiza el recall "
            f"({m_v2['recall_ataque']:.4f}) y además tiene el mayor macro-F1."),
        "congelacion": ("Este archivo y configs/v2.yaml se confirman en Git (tag v2) antes del "
                        "commit que ejecuta src/evaluar_test.py. El modelo y el umbral no se "
                        "modifican después de ver test."),
        "test_usado_para_decidir": False,
    }
    with open(EVID / "seleccion.json", "w") as f:
        json.dump(seleccion, f, indent=2, ensure_ascii=False)
        f.write("\n")
    figuras(runs, regla, m_v2, m_v1)
    print(df[["experimento_id", "umbral_oper", "recall_ataque_oper", "fpr_oper", "macro_f1_oper"]])
    print("Elegido:", ganador.experimento_id, ganador.umbral_oper)


if __name__ == "__main__":
    main()

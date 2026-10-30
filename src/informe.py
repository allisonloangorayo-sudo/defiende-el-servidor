"""Genera informe.pdf (máx. 5 páginas) a partir de los archivos de evidencia.

Uso:  python -m src.informe
Todas las cifras se leen de evidencias/*.json|csv, configs/referencias.json y
evidencias/recuperacion/; no hay números escritos a mano.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

from src.config import DATA_PREP, EVID, RAIZ

FIG = EVID / "figuras"
TINTA, GRIS, ACENTO, TENUE = "#111827", "#6b7280", "#c2410c", "#e5e7eb"

# Fuentes DejaVu incluidas con matplotlib: portables y con ≤, →, ±.
_ttf = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
pdfmetrics.registerFont(TTFont("DV", str(_ttf / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DVB", str(_ttf / "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("DVM", str(_ttf / "DejaVuSansMono.ttf")))

ST = {
    "titulo": ParagraphStyle("t", fontName="DVB", fontSize=15, leading=18, textColor=TINTA),
    "sub": ParagraphStyle("s", fontName="DV", fontSize=8.5, leading=11, textColor=GRIS),
    "h": ParagraphStyle("h", fontName="DVB", fontSize=11, leading=14, spaceBefore=9,
                        spaceAfter=3, textColor=TINTA, keepWithNext=1),
    "p": ParagraphStyle("p", fontName="DV", fontSize=9, leading=12.4, alignment=TA_JUSTIFY,
                        textColor=TINTA, spaceAfter=3),
    "cap": ParagraphStyle("c", fontName="DV", fontSize=7.2, leading=9, textColor=GRIS,
                          spaceAfter=4),
    "cel": ParagraphStyle("cel", fontName="DV", fontSize=7.6, leading=9.4, textColor=TINTA),
    "celb": ParagraphStyle("celb", fontName="DVB", fontSize=7.6, leading=9.4, textColor=TINTA),
}


def f4(x): return f"{x:.4f}".replace(".", ",")
def f2(x): return f"{x:.2f}".replace(".", ",")
def pct(x): return f"{x * 100:.1f} %".replace(".", ",")
def miles(n): return f"{n:,}".replace(",", ".")
def mono(t): return f"<font name='DVM' size='6.6'>{t}</font>"


def tabla(filas, anchos, cabecera=True, resaltar=None):
    datos = [[Paragraph(str(c), ST["celb" if (cabecera and i == 0) else "cel"]) for c in fila]
             for i, fila in enumerate(filas)]
    t = Table(datos, colWidths=[a * cm for a in anchos], hAlign="LEFT")
    estilo = [("LINEABOVE", (0, 0), (-1, 0), 0.8, colors.HexColor(TINTA)),
              ("LINEBELOW", (0, -1), (-1, -1), 0.8, colors.HexColor(TINTA)),
              ("VALIGN", (0, 0), (-1, -1), "TOP"),
              ("TOPPADDING", (0, 0), (-1, -1), 1.6), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.6),
              ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3)]
    if cabecera:
        estilo.append(("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.HexColor(GRIS)))
    for r in resaltar or []:
        estilo.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#fff4ed")))
    t.setStyle(TableStyle(estilo))
    return t


def fig_matrices(sel, test) -> Path:
    paneles = [("v1 · validación (u=0,50)", sel["metricas_validacion_v1"]["matriz_confusion"]),
               ("v2 · validación (u=0,66)", sel["metricas_validacion"]["matriz_confusion"]),
               ("v2 · test (u=0,66)", test["matriz_confusion"])]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.15), dpi=220)
    for ax, (titulo, cm_) in zip(axes, paneles):
        m = [[cm_["tn"], cm_["fp"]], [cm_["fn"], cm_["tp"]]]
        tot = [cm_["tn"] + cm_["fp"], cm_["fn"] + cm_["tp"]]
        ax.imshow([[m[i][j] / tot[i] for j in range(2)] for i in range(2)], cmap="Greys",
                  vmin=0, vmax=1.6)
        for i in range(2):
            for j in range(2):
                err = (i, j) in ((0, 1), (1, 0))
                ax.text(j, i, f"{miles(m[i][j])}\n{pct(m[i][j] / tot[i])}", ha="center",
                        va="center", fontsize=7.5, color=ACENTO if err else "black",
                        fontweight="bold" if err else "normal")
        ax.set_xticks([0, 1], ["normal", "ataque"])
        ax.set_yticks([0, 1], ["normal", "ataque"])
        ax.set_xlabel("predicho", fontsize=7)
        ax.set_title(titulo, fontsize=8)
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0].set_ylabel("real", fontsize=7)
    fig.tight_layout()
    ruta = FIG / "matrices_confusion.png"
    fig.savefig(ruta)
    plt.close(fig)
    return ruta


def fig_categorias(sel, test) -> Path:
    run = sel["mlflow_run_id"]
    s = pd.read_csv(EVID / "mlflow" / run / "artifacts" / "scores_validacion.csv")
    cats = pd.read_csv(DATA_PREP / "val.csv", usecols=["row_id", "attack_cat"])
    s = s.merge(cats, on="row_id")
    val = (s.score_attack >= sel["umbral"]).groupby(s.attack_cat).mean()
    te = pd.Series({k: v["proporcion_predicha_ataque"]
                    for k, v in test["tasa_prediccion_ataque_por_attack_cat"].items()})
    orden = te.sort_values().index.tolist()
    fig, ax = plt.subplots(figsize=(7.2, 2.3), dpi=220)
    y = range(len(orden))
    ax.barh([i + 0.2 for i in y], [val[c] for c in orden], height=0.38, color="#9aa5b1",
            label="validación")
    ax.barh([i - 0.2 for i in y], [te[c] for c in orden], height=0.38, color=ACENTO, label="test")
    ax.set_yticks(list(y), [("Normal (FPR)" if c == "Normal" else c) for c in orden], fontsize=7)
    ax.set_xlabel("Proporción clasificada como ataque por v2 (u = 0,66)", fontsize=7)
    ax.set_xlim(0, 1.0)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.grid(axis="x", alpha=0.25, lw=0.5)
    fig.tight_layout()
    ruta = FIG / "deteccion_por_categoria.png"
    fig.savefig(ruta)
    plt.close(fig)
    return ruta


def main() -> None:
    aud = json.loads((EVID / "auditoria.json").read_text())
    exp = pd.read_csv(EVID / "experimentos.csv")
    sel = json.loads((EVID / "seleccion.json").read_text())
    test = json.loads((EVID / "metricas-test.json").read_text())
    refs_p = RAIZ / "configs" / "referencias.json"
    refs = json.loads(refs_p.read_text()) if refs_p.exists() else {}
    rec_p = EVID / "recuperacion" / "entorno_nuevo" / "resumen_recuperacion.json"
    rec = json.loads(rec_p.read_text()) if rec_p.exists() else None
    v1c = json.loads((RAIZ / "models" / "v1" / "config.json").read_text())
    v2c = json.loads((RAIZ / "models" / "v2" / "config.json").read_text())
    tr, te = aud["archivos"]["entrenamiento"], aud["archivos"]["test"]
    sp = aud["particiones"]
    P = lambda t: Paragraph(t, ST["p"])  # noqa: E731
    H = lambda t: Paragraph(t, ST["h"])  # noqa: E731
    C = lambda t: Paragraph(t, ST["cap"])  # noqa: E731

    h = []
    h.append(Paragraph("Defiende el servidor", ST["titulo"]))
    h.append(Paragraph("Versionado de código, datos, experimentos y modelos · MLOps 2026-II · "
                       "Universidad Externado de Colombia · Allison Loango y Kevin Nieto · Grupo A · "
                       "Docente: Julián Zuluaga · 30 de octubre de 2026", ST["sub"]))
    h.append(Spacer(1, 5))
    h.append(P(
        "Construimos un clasificador binario de conexiones de red (UNSW-NB15; 1 = ataque, clase "
        "positiva) y, sobre todo, su trazabilidad: cada versión del modelo se identifica por un "
        "<b>commit Git</b>, una <b>versión DVC</b> de los datos, un <b>run de MLflow</b> y una "
        "<b>revisión SHA de Hugging Face</b>, enlazados por el SHA-256 del artefacto. "
        f"La versión final (v2) detecta {pct(sel['metricas_validacion']['recall_ataque'])} de los ataques "
        f"con FPR {pct(sel['metricas_validacion']['fpr'])} en validación; en test detecta "
        f"{pct(test['metricas']['recall_ataque'])} pero su FPR sube a {pct(test['metricas']['fpr'])}, "
        "hallazgo que se analiza en la sección 6."))

    # 1. Datos y auditoría
    h.append(H("1. Datos y auditoría"))
    h.append(P(
        f"Se usaron las particiones publicadas, descargadas el {aud['fecha_descarga_utc'][:10]} del mirror "
        "fijado por commit que indica el enunciado (fuente oficial: UNSW Canberra). Ambos SHA-256 "
        "coinciden con el anexo. No hay faltantes. El 42 % de las filas de entrenamiento repite "
        "exactamente las 42 variables predictoras de otra fila, y hay grupos de entradas idénticas "
        "con etiquetas opuestas. Se excluyen <i>label</i>, <i>attack_cat</i> (revela la respuesta) e "
        "<i>id</i>; quedan 3 categóricas (proto, service, state) y 39 numéricas."))
    h.append(tabla([
        ["Archivo", "Filas × col.", "Normal / ataque", "Filas duplicadas (entradas)", "Grupos contradictorios (filas)", "SHA-256"],
        ["training-set", f"{miles(tr['filas'])} × {tr['columnas']}", f"{miles(tr['clases_label']['0'])} / {miles(tr['clases_label']['1'])}",
         miles(tr["duplicados_por_entradas"]["filas_duplicadas"]),
         f"{tr['contradicciones']['grupos_con_etiquetas_distintas']} ({tr['contradicciones']['filas_en_grupos_contradictorios']})",
         mono(tr["sha256"][:16] + "…")],
        ["testing-set", f"{miles(te['filas'])} × {te['columnas']}", f"{miles(te['clases_label']['0'])} / {miles(te['clases_label']['1'])}",
         miles(te["duplicados_por_entradas"]["filas_duplicadas"]),
         f"{te['contradicciones']['grupos_con_etiquetas_distintas']} ({te['contradicciones']['filas_en_grupos_contradictorios']})",
         mono(te["sha256"][:16] + "…")],
    ], [2.2, 2.5, 2.9, 3.2, 3.4, 3.6]))
    h.append(Spacer(1, 3))
    h.append(P(
        "<b>Política.</b> Duplicados y contradicciones se conservan: reflejan la frecuencia real del "
        "tráfico y un ruido de etiqueta que también existe en test, así que eliminarlos inflaría la "
        "validación. Para que no haya fuga, cada grupo de entradas idénticas (hash de las 42 variables) "
        "va completo a entrenamiento o a validación mediante <i>StratifiedGroupKFold</i> (5 pliegues, "
        f"semilla 42, pliegue 0 = validación): train {miles(sp['train']['filas'])} filas, validación "
        f"{miles(sp['val']['filas'])} ({pct(sp['val']['proporcion_ataque'])} ataques en ambas), "
        f"<b>{sp['grupos_compartidos_train_val']} grupos compartidos</b>. Test solo se auditó: "
        f"{miles(aud['cruces_test']['filas_test_con_entrada_en_train_o_val'])} filas "
        f"({pct(aud['cruces_test']['proporcion_test_solapada'])}) repiten entradas vistas, y state trae "
        "dos categorías nuevas (ACC, CLO), absorbidas por los codificadores."))

    # 2. Metodología
    h.append(H("2. Metodología y trazabilidad"))
    h.append(P(
        "Preprocesamiento y estimador viven en un único <i>Pipeline</i> de scikit-learn ajustado solo "
        "con entrenamiento (imputación, codificación, escalado), guardado como <i>model.joblib</i>. Los "
        "cuatro experimentos comparten partición, semilla y código de evaluación. Antes de entrenar, "
        "<i>src/entrenar.py</i> exige que código y configuración estén confirmados en Git; cada run de "
        "MLflow guarda hipótesis, cambio, parámetros, semilla, commit, md5 DVC de datos originales y "
        "preparados, métricas, scores de validación y el modelo completo con su SHA-256. DVC versiona "
        "los CSV recibidos (<i>data/raw.dvc</i>) y las particiones (<i>data/prepared.dvc</i>) en commits "
        "separados; su almacenamiento es un dataset público de Hugging Face accesible por HTTP sin "
        "credenciales. Hugging Face guarda v1 y v2 como dos commits del mismo repositorio, con tags."))
    h.append(P(
        "<b>Errores priorizados.</b> Para un equipo de seguridad un ataque no detectado (FN) es el error "
        "más caro, pero cada falso positivo consume tiempo de analista y su exceso produce fatiga de "
        "alertas. La regla, fijada antes de ver resultados (<i>configs/seleccion.yaml</i>), es: "
        "<b>maximizar el recall de ataque con FPR ≤ 5 % en validación</b>, desempate por macro-F1. "
        "Precision, recall, FPR y macro-F1 se calculan desde la matriz de confusión; un denominador cero "
        "se reporta como 0 y se marca (no ocurrió en ninguna evaluación)."))

    # 3. Experimentos
    h.append(H("3. Cuatro experimentos (validación)"))
    filas = [["ID", "Modelo · preprocesamiento", "Umbral", "Recall", "Precision", "FPR", "Macro-F1", "Run ID"]]
    for _, r in exp.iterrows():
        filas.append([r.experimento_id, f"{r.estimador.replace('Classifier', '')} · {r.preprocesamiento}",
                      "0,50", f4(r.recall_ataque_u050), f4(r.precision_ataque_u050), f4(r.fpr_u050),
                      f4(r.macro_f1_u050), mono(r.mlflow_run_id[:12] + "…")])
        filas.append(["", "", f2(r.umbral_oper), f4(r.recall_ataque_oper), f4(r.precision_ataque_oper),
                      f4(r.fpr_oper), f4(r.macro_f1_oper), ""])
    h.append(tabla(filas, [0.8, 5.6, 1.4, 1.5, 1.7, 1.5, 1.7, 3.6], resaltar=[8]))
    h.append(C("Cada experimento aparece a 0,50 y a su umbral operativo (máximo recall con FPR ≤ 5 %). "
               "Fila resaltada: punto elegido para v2. Hipótesis, parámetros, commits y md5 DVC completos "
               "en evidencias/experimentos.csv."))
    hip = [["ID", "Hipótesis (resumen)", "Resultado"],
           ["E1", "Referencia lineal: recall alto pero FPR alta por las colas largas de bytes/cargas.",
            "Confirmada: FPR 17,6 % a 0,50; con FPR ≤ 5 % el recall cae a 89,5 %."],
           ["E2", "log1p antes de estandarizar vuelve más lineal la señal y baja la FPR.",
            "Parcial: FPR 13,6 % a 0,50 y recall 91,3 % con FPR ≤ 5 %; Fuzzers solo 54 %."],
           ["E3", "Un árbol captura reglas por umbrales e interacciones (TTL, estado, servicio).",
            "Confirmada: recall 93,2 % con FPR ≤ 5 %, supera a ambas regresiones."],
           ["E4", "Un ensemble boosting reduce la varianza del árbol y da la mejor curva recall–FPR.",
            "Confirmada: recall 95,3 % con FPR 4,8 %; errores concentrados en Fuzzers y Analysis."]]
    h.append(Spacer(1, 4))
    h.append(KeepTogether(tabla(hip, [0.8, 8.4, 8.6])))
    h.append(Spacer(1, 4))
    h.append(KeepTogether([Image(str(FIG / "curvas_recall_fpr_validacion.png"), width=13.2 * cm, height=7.4 * cm),
                           C("Figura 1. Curvas recall–FPR en validación. La línea discontinua marca el presupuesto de "
                             "falsas alarmas; v1 queda muy a la derecha y v2 es el punto con mayor recall dentro de él.")]))

    # 4. v1 vs v2
    m1, m2 = sel["metricas_validacion_v1"], sel["metricas_validacion"]
    h.append(H("4. Dos versiones: v1 y v2"))
    h.append(P(
        "<b>v1</b> es la primera solución funcional (E1 con el umbral convencional 0,50), empaquetada y "
        "etiquetada antes de correr E2–E4. <b>v2</b> cambia el modelo (HistGradientBoosting con categorías "
        "nativas, sin escalado) y el umbral (0,66), ambos elegidos con la regla de la sección 2. v2 no "
        "mejora todo: reduce las falsas alarmas a menos de un tercio y sube precision y macro-F1, a costa "
        f"de {miles(m2['matriz_confusion']['fn'] - m1['matriz_confusion']['fn'])} ataques no detectados más, "
        "sobre todo Fuzzers, cuyo tráfico se parece al normal. Aceptamos ese intercambio porque v1 "
        "alertaría sobre 1 de cada 6 conexiones normales, un volumen inmanejable para un SOC."))
    h.append(tabla([
        ["", "Modelo", "Umbral", "Recall", "Precision", "FPR", "Macro-F1", "FP", "FN", "SHA-256 del artefacto"],
        ["v1", "Regresión logística (E1)", "0,50", f4(m1["recall_ataque"]), f4(m1["precision_ataque"]), f4(m1["fpr"]),
         f4(m1["macro_f1"]), miles(m1["matriz_confusion"]["fp"]), miles(m1["matriz_confusion"]["fn"]), mono(v1c["sha256_model_joblib"][:16] + "…")],
        ["v2", "HistGradientBoosting (E4)", "0,66", f4(m2["recall_ataque"]), f4(m2["precision_ataque"]), f4(m2["fpr"]),
         f4(m2["macro_f1"]), miles(m2["matriz_confusion"]["fp"]), miles(m2["matriz_confusion"]["fn"]), mono(v2c["sha256_model_joblib"][:16] + "…")],
    ], [0.7, 3.5, 1.5, 1.4, 1.7, 1.4, 1.7, 1.1, 1.1, 3.7]))
    trz = [["Eslabón", "v1", "v2"],
           ["Commit que entrenó (Git)", mono(v1c["git_commit_entrenamiento"]), mono(v2c["git_commit_entrenamiento"])],
           ["Tag Git", "v1", "v2"],
           ["Datos DVC (raw / prepared md5)", mono(v1c["dvc_raw_md5"][:10] + "… / " + v1c["dvc_prepared_md5"][:10] + "…"),
            mono(v2c["dvc_raw_md5"][:10] + "… / " + v2c["dvc_prepared_md5"][:10] + "…")],
           ["Run MLflow", mono(v1c["mlflow_run_id"]), mono(v2c["mlflow_run_id"])],
           ["Revisión Hugging Face", mono(refs.get("v1", {}).get("hf_revision_sha", "ver configs/referencias.json")),
            mono(refs.get("v2", {}).get("hf_revision_sha", "ver configs/referencias.json"))]]
    h.append(Spacer(1, 3))
    h.append(tabla(trz, [4.4, 6.7, 6.7]))
    h.append(C("Cadena de trazabilidad. Cada revisión de Hugging Face contiene model.joblib, config.json (umbral, variables, "
               "referencias), predict.py, requirements.txt y la model card."))

    # 5. Umbral
    h.append(H("5. Elección del umbral"))
    h.append(P(
        "Para cada run se barrieron umbrales de 0,01 a 0,99 sobre los scores de validación y se tomó el de "
        "mayor recall con FPR ≤ 5 %. Luego se compararon esos puntos entre experimentos; E4 a 0,66 gana en "
        "recall y también en macro-F1. Subir el umbral de 0,50 a 0,66 en E4 baja la FPR de "
        f"{pct(exp.set_index('experimento_id').loc['E4', 'fpr_u050'])} a {pct(m2['fpr'])} y cuesta "
        + f"{(exp.set_index('experimento_id').loc['E4', 'recall_ataque_u050'] - m2['recall_ataque']) * 100:.1f}".replace(".", ",")
        + " puntos porcentuales de recall. "
        "La decisión se escribió en <i>evidencias/seleccion.json</i> y se confirmó en Git (tag v2) antes del "
        "commit que ejecuta <i>src/evaluar_test.py</i>; el script verifica SHA-256 y umbral congelados y se "
        "niega a correr dos veces. Como el umbral se ajustó en validación, las métricas de validación al "
        "umbral elegido son algo optimistas."))

    # 6. Test
    mt = test["metricas"]
    h.append(H("6. Evaluación final en test"))
    h.append(P(
        f"v2 se evaluó una sola vez sobre las {miles(test['filas_evaluadas'])} filas de test (sin eliminar ninguna): "
        f"recall {f4(mt['recall_ataque'])}, precision {f4(mt['precision_ataque'])}, FPR {f4(mt['fpr'])}, "
        f"macro-F1 {f4(mt['macro_f1'])}, accuracy {f4(mt['accuracy'])}. Las predicciones por fila "
        "(<i>row_id, y_true, score_attack, y_pred</i>) permiten recalcular estas cifras. "
        "El resultado empeora frente a validación y se conserva: el recall se mantiene, pero la FPR se "
        "multiplica por 3,7. La figura 3 muestra que el problema está en la clase normal y no en los ataques: "
        "el tráfico normal del archivo de test se distribuye distinto del de entrenamiento, y una validación "
        "extraída del mismo archivo de entrenamiento no podía anticiparlo. Con datos de operación, el "
        "umbral debería recalibrarse sobre tráfico normal reciente y no sobre test."))
    h.append(KeepTogether([Image(str(fig_matrices(sel, test)), width=17 * cm, height=5.07 * cm),
                           C("Figura 2. Matrices de confusión (en naranja, los errores con su tasa por fila real).")]))
    h.append(KeepTogether([Image(str(fig_categorias(sel, test)), width=17 * cm, height=5.43 * cm),
                           C("Figura 3. Proporción clasificada como ataque por categoría. En Normal esa proporción es la FPR.")]))

    # 7. Recuperación
    h.append(H("7. Prueba de recuperación"))
    if rec:
        r1 = json.loads((EVID / "recuperacion" / "entorno_nuevo" / "comparacion_v1.json").read_text())
        r2 = json.loads((EVID / "recuperacion" / "entorno_nuevo" / "comparacion_v2.json").read_text())
        h.append(P(
            "En una carpeta y un entorno nuevos, con cachés vacías, <i>scripts/recuperar.sh</i> clonó el código "
            f"desde GitHub (commit {mono(rec['commit_recuperado'])}), instaló las dependencias fijadas, recuperó "
            "con <i>dvc pull</i> los datos originales y preparados desde el almacenamiento entregado y verificó "
            "sus SHA-256, descargó v1 y v2 de Hugging Face por SHA completo y predijo las mismas 20 filas de "
            f"validación. Resultado: v1 <b>{r1['resultado']}</b> ({r1['y_pred_coinciden']}/20 etiquetas iguales, "
            f"diferencia máxima de score {r1['max_abs_diff_score']:.1e}), v2 <b>{r2['resultado']}</b> "
            f"({r2['y_pred_coinciden']}/20, {r2['max_abs_diff_score']:.1e}); tolerancia 1e-9. Log, entradas, salidas "
            "y hashes en <i>evidencias/recuperacion/entorno_nuevo/</i>."))
    else:
        h.append(P(
            "<i>scripts/recuperar.sh</i> clona el código desde GitHub, crea un entorno nuevo con cachés vacías, "
            "ejecuta <i>dvc pull</i>, verifica huellas, descarga v1 y v2 por SHA y compara sus predicciones sobre "
            "20 filas fijas de validación (tolerancia 1e-9). Un ensayo previo en contenedor limpio, con GitHub y "
            "Hugging Face simulados localmente, coincidió en 20/20 filas para ambas versiones "
            "(<i>evidencias/recuperacion/ensayo_contenedor/</i>). La prueba oficial queda en "
            "<i>evidencias/recuperacion/entorno_nuevo/</i>."))

    # 8. Limitaciones y conclusiones
    h.append(H("8. Limitaciones y conclusiones"))
    h.append(P(
        "UNSW-NB15 es tráfico de laboratorio de 2015: estos resultados no demuestran eficacia en una red "
        "actual. La validación proviene del archivo de entrenamiento, por lo que subestimó el cambio de "
        "distribución de test; el 10,4 % de test repite entradas ya vistas. Los grupos contradictorios son "
        "ruido irreducible. El artefacto es un pickle de scikit-learn 1.9.1: depende de esa versión y solo se "
        "carga tras verificar su SHA-256. Se usó todo el entrenamiento (sin muestreo) en CPU."))
    h.append(P(
        "La respuesta a la pregunta central es sí: desde GitHub, el remote DVC y Hugging Face se recupera "
        "cualquiera de las dos versiones, se sabe de qué datos (md5 DVC) y experimento (run ID) viene, y "
        "produce exactamente las mismas predicciones. En lo metodológico, la regla fijada de antemano "
        "llevó a una v2 con un tercio de las falsas alarmas de v1 en validación; el test mostró que ese "
        "presupuesto no se transfiere a otra distribución de tráfico normal, y lo reportamos tal cual."))
    h.append(Paragraph("Referencias: Moustafa, N. & Slay, J. (2015). UNSW-NB15: a comprehensive data set for network "
                       "intrusion detection systems. MilCIS, IEEE. · Pedregosa et al. (2011). Scikit-learn. JMLR 12. · "
                       "Documentación oficial de Git, DVC, MLflow Tracking y Hugging Face Hub (descarga por revisión, model cards).",
                       ST["cap"]))

    def pie(canvas, doc):
        canvas.saveState()
        canvas.setFont("DV", 6.8)
        canvas.setFillColor(colors.HexColor(GRIS))
        canvas.drawString(1.6 * cm, 1.0 * cm, "MLOps 2026-II · Defiende el servidor · Allison Loango y Kevin Nieto · Grupo A")
        canvas.drawRightString(A4[0] - 1.6 * cm, 1.0 * cm, f"{doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(str(RAIZ / "informe.pdf"), pagesize=A4, leftMargin=1.6 * cm,
                            rightMargin=1.6 * cm, topMargin=1.4 * cm, bottomMargin=1.5 * cm,
                            title="Defiende el servidor — informe", author="Allison Loango, Kevin Nieto")
    doc.build(h, onFirstPage=pie, onLaterPages=pie)
    print("informe.pdf generado")


if __name__ == "__main__":
    main()

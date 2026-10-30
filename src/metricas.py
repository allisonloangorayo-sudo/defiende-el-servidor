"""Métricas con ataque (1) como clase positiva.

Denominadores cero: si un cociente no está definido (p. ej. ninguna predicción
positiva para precision), la métrica se reporta como 0.0 y se marca en
`denominadores_cero`. Así una configuración degenerada nunca parece buena.
"""
from __future__ import annotations

import numpy as np


def _div(num: int, den: int, nombre: str, avisos: list[str]) -> float:
    if den == 0:
        avisos.append(nombre)
        return 0.0
    return num / den


def matriz_confusion(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    return {
        "tn": int(((y_true == 0) & (y_pred == 0)).sum()),
        "fp": int(((y_true == 0) & (y_pred == 1)).sum()),
        "fn": int(((y_true == 1) & (y_pred == 0)).sum()),
        "tp": int(((y_true == 1) & (y_pred == 1)).sum()),
    }


def metricas_desde_matriz(cm: dict) -> dict:
    tn, fp, fn, tp = cm["tn"], cm["fp"], cm["fn"], cm["tp"]
    avisos: list[str] = []
    recall = _div(tp, tp + fn, "recall_ataque", avisos)
    precision = _div(tp, tp + fp, "precision_ataque", avisos)
    fpr = _div(fp, fp + tn, "fpr", avisos)
    # F1 por clase y macro-F1 (promedio simple de F1 normal y F1 ataque).
    f1_ataque = _div(2 * tp, 2 * tp + fp + fn, "f1_ataque", avisos)
    f1_normal = _div(2 * tn, 2 * tn + fn + fp, "f1_normal", avisos)
    n = tn + fp + fn + tp
    return {
        "recall_ataque": recall,
        "precision_ataque": precision,
        "fpr": fpr,
        "macro_f1": (f1_ataque + f1_normal) / 2,
        "f1_ataque": f1_ataque,
        "f1_normal": f1_normal,
        "accuracy": _div(tp + tn, n, "accuracy", avisos),
        "denominadores_cero": avisos,
    }


def evaluar(y_true, score, umbral: float) -> dict:
    """Regla de decisión: ataque si score_attack >= umbral."""
    y_pred = (np.asarray(score) >= umbral).astype(int)
    cm = matriz_confusion(y_true, y_pred)
    return {"umbral": umbral, "matriz_confusion": cm, **metricas_desde_matriz(cm)}


def barrido_umbrales(y_true, score, umbrales=None) -> list[dict]:
    if umbrales is None:
        umbrales = np.round(np.arange(0.05, 0.951, 0.01), 2)
    return [evaluar(y_true, score, float(u)) for u in umbrales]

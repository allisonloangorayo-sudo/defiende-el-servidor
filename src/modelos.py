"""Construcción de la solución completa (preprocesamiento + estimador) desde una config.

Toda la transformación vive dentro de un `Pipeline` de scikit-learn: se ajusta solo
con entrenamiento y se guarda/carga como un único artefacto (model.joblib).
Solo se usan funciones de scikit-learn y numpy para que el artefacto cargue sin el
código de este repositorio.
"""
from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (FunctionTransformer, OneHotEncoder, OrdinalEncoder,
                                   StandardScaler)
from sklearn.tree import DecisionTreeClassifier

from src.config import CATEGORICAS

ESTIMADORES = {
    "LogisticRegression": LogisticRegression,
    "DecisionTreeClassifier": DecisionTreeClassifier,
    "HistGradientBoostingClassifier": HistGradientBoostingClassifier,
}


def _rama_numerica(tipo: str) -> Pipeline:
    pasos = [("imputar", SimpleImputer(strategy="median"))]
    if tipo == "log1p_estandar":
        # Todas las numéricas son >= 0 en entrenamiento (ver auditoría); log1p comprime colas.
        pasos.append(("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")))
    if tipo in ("estandar", "log1p_estandar"):
        pasos.append(("escalar", StandardScaler()))
    elif tipo != "ninguno":
        raise ValueError(f"Preprocesamiento numérico desconocido: {tipo}")
    return Pipeline(pasos)


def _rama_categorica(tipo: str, min_frecuencia) -> Pipeline:
    imputar = ("imputar", SimpleImputer(strategy="most_frequent"))
    if tipo == "onehot":
        cod = OneHotEncoder(handle_unknown="infrequent_if_exist" if min_frecuencia else "ignore",
                            min_frequency=min_frecuencia, sparse_output=False)
    elif tipo == "ordinal":
        cod = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                             encoded_missing_value=-1)
    else:
        raise ValueError(f"Codificación categórica desconocida: {tipo}")
    return Pipeline([imputar, ("codificar", cod)])


def construir_pipeline(cfg: dict, numericas: list[str]) -> Pipeline:
    pre_cfg = cfg["preprocesamiento"]
    # Categóricas primero: el booster usa sus índices (0, 1, 2) como categóricas nativas.
    pre = ColumnTransformer(
        [
            ("cat", _rama_categorica(pre_cfg["categoricas"], pre_cfg.get("onehot_min_frecuencia")),
             CATEGORICAS),
            ("num", _rama_numerica(pre_cfg["numericas"]), numericas),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    m_cfg = cfg["modelo"]
    params = dict(m_cfg.get("parametros", {}))
    clase = ESTIMADORES[m_cfg["clase"]]
    if "random_state" in clase().get_params():
        params["random_state"] = cfg["semilla"]
    if m_cfg["clase"] == "HistGradientBoostingClassifier" and pre_cfg["categoricas"] == "ordinal":
        params["categorical_features"] = list(range(len(CATEGORICAS)))
    return Pipeline([("preprocesamiento", pre), ("modelo", clase(**params))])

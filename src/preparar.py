"""Auditoría de los CSV originales y creación de particiones sin fuga.

Uso:  python -m src.preparar

Entradas : data/raw/UNSW_NB15_{training,testing}-set.csv   (versión DVC "original")
Salidas  : data/prepared/{train,val,test}.csv + splits.json (versión DVC "preparada")
           evidencias/auditoria.json

Política (justificada en el informe):
* Las entradas idénticas (mismas 42 variables predictoras) forman un grupo. Un grupo
  completo va a entrenamiento o a validación, nunca a ambos (StratifiedGroupKFold).
* No se eliminan duplicados ni grupos contradictorios: reflejan la frecuencia real del
  tráfico y el ruido de etiqueta que también existe en test. Quitarlos inflaría las
  métricas de validación. Se documentan y quedan íntegros en una sola partición.
* El archivo de test solo se audita (esquema, clases, duplicados, cruces de entradas).
  Sus etiquetas no participan en ninguna decisión.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.config import (ARCHIVO_TEST, ARCHIVO_TRAIN, CATEGORICAS, DATA_PREP, EVID, EXCLUIDAS,
                        FECHA_DESCARGA, FUENTE_OFICIAL, MIRROR_BASE, N_FOLDS_SPLIT, OBJETIVO,
                        SEMILLA, SHA256_ESPERADO)
from src.utils import (columnas_predictoras, guardar_json, leer_csv, sha256_archivo,
                       sha256_texto)


def clave_entrada(X: pd.DataFrame) -> pd.Series:
    """Hash determinista de las variables predictoras (identifica entradas idénticas)."""
    return pd.util.hash_pandas_object(X, index=False).astype("uint64")


def auditar_archivo(ruta, df: pd.DataFrame, X_cols: list[str]) -> dict:
    sha = sha256_archivo(ruta)
    esperado = SHA256_ESPERADO[ruta.name]
    if sha != esperado:
        raise ValueError(f"{ruta.name}: SHA-256 {sha} no coincide con el anexo {esperado}")
    clave = clave_entrada(df[X_cols])
    por_grupo = df.groupby(clave)[OBJETIVO].agg(["nunique", "size"])
    contradic = por_grupo[por_grupo["nunique"] > 1]
    return {
        "archivo": ruta.name,
        "url_descarga": MIRROR_BASE + ruta.name,
        "sha256": sha,
        "sha256_anexo": esperado,
        "sha256_coincide": True,
        "bytes": ruta.stat().st_size,
        "filas": int(len(df)),
        "columnas": int(df.shape[1]),
        "tipos": {c: str(t) for c, t in df.dtypes.items()},
        "faltantes_total": int(df.isna().sum().sum()),
        "faltantes_por_columna": {c: int(v) for c, v in df.isna().sum().items() if v > 0},
        "clases_label": {str(k): int(v) for k, v in df[OBJETIVO].value_counts().sort_index().items()},
        "proporcion_ataque": round(float(df[OBJETIVO].mean()), 6),
        "attack_cat": {str(k): int(v) for k, v in df["attack_cat"].value_counts().items()},
        "id_unico": bool(df["id"].is_unique),
        "duplicados_por_entradas": {
            "filas_duplicadas": int(clave.duplicated().sum()),
            "entradas_unicas": int(clave.nunique()),
            "grupos_con_mas_de_una_fila": int((por_grupo["size"] > 1).sum()),
        },
        "contradicciones": {
            "grupos_con_etiquetas_distintas": int(len(contradic)),
            "filas_en_grupos_contradictorios": int(contradic["size"].sum()),
        },
        "duplicados_fila_completa_sin_id": int(df.drop(columns=["id"]).duplicated().sum()),
    }


def main() -> None:
    DATA_PREP.mkdir(parents=True, exist_ok=True)
    train_raw = leer_csv(ARCHIVO_TRAIN)
    test_raw = leer_csv(ARCHIVO_TEST)

    X_cols = columnas_predictoras(train_raw)
    assert list(columnas_predictoras(test_raw)) == X_cols, "Esquemas distintos entre archivos"
    assert len(X_cols) == 42 and not set(X_cols) & set(EXCLUIDAS)

    aud_train = auditar_archivo(ARCHIVO_TRAIN, train_raw, X_cols)
    aud_test = auditar_archivo(ARCHIVO_TEST, test_raw, X_cols)

    # Identificadores inequívocos archivo/fila.
    train = train_raw.copy()
    test = test_raw.copy()
    train.insert(0, "row_id", "train:" + train["id"].astype(str))
    test.insert(0, "row_id", "test:" + test["id"].astype(str))
    train.insert(1, "grupo", clave_entrada(train[X_cols]).astype(str))
    test.insert(1, "grupo", clave_entrada(test[X_cols]).astype(str))

    # Partición agrupada y estratificada: pliegue 0 = validación (20 %).
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS_SPLIT, shuffle=True, random_state=SEMILLA)
    idx_tr, idx_va = next(sgkf.split(train[X_cols], train[OBJETIVO], groups=train["grupo"]))
    train["split"] = "train"
    train.loc[train.index[idx_va], "split"] = "val"

    tr = train[train.split == "train"].drop(columns="split")
    va = train[train.split == "val"].drop(columns="split")
    grupos_tr, grupos_va = set(tr.grupo), set(va.grupo)
    comunes = grupos_tr & grupos_va
    assert not comunes, "Fuga: grupos de entradas idénticas en train y val"

    orden = ["row_id", "grupo"] + X_cols + ["attack_cat", OBJETIVO]
    tr = tr[orden]
    va = va[orden]
    te = test[orden]
    rutas = {"train": DATA_PREP / "train.csv", "val": DATA_PREP / "val.csv",
             "test": DATA_PREP / "test.csv"}
    tr.to_csv(rutas["train"], index=False)
    va.to_csv(rutas["val"], index=False)
    te.to_csv(rutas["test"], index=False)

    def resumen(df: pd.DataFrame, ruta) -> dict:
        return {
            "archivo": f"data/prepared/{ruta.name}",
            "filas": int(len(df)),
            "grupos": int(df.grupo.nunique()),
            "clases_label": {str(k): int(v) for k, v in df[OBJETIVO].value_counts().sort_index().items()},
            "proporcion_ataque": round(float(df[OBJETIVO].mean()), 6),
            "sha256_archivo": sha256_archivo(ruta),
            "sha256_lista_row_id": sha256_texto("\n".join(df.row_id)),
        }

    splits = {
        "metodo": "StratifiedGroupKFold(n_splits=5, shuffle=True) sobre el archivo de "
                  "entrenamiento; pliegue 0 = validación; grupos = hash de las 42 variables predictoras",
        "semilla": SEMILLA,
        "train": resumen(tr, rutas["train"]),
        "val": resumen(va, rutas["val"]),
        "test": resumen(te, rutas["test"]),
        "grupos_compartidos_train_val": len(comunes),
    }
    guardar_json(splits, DATA_PREP / "splits.json")

    # Cruces de entradas entre test y entrenamiento/validación (solo auditoría).
    g_te = te.grupo
    cruce = {
        "filas_test_con_entrada_en_train": int(g_te.isin(grupos_tr).sum()),
        "filas_test_con_entrada_en_val": int(g_te.isin(grupos_va).sum()),
        "filas_test_con_entrada_en_train_o_val": int(g_te.isin(grupos_tr | grupos_va).sum()),
    }
    cruce["proporcion_test_solapada"] = round(cruce["filas_test_con_entrada_en_train_o_val"] / len(te), 6)

    num_cols = [c for c in X_cols if c not in CATEGORICAS]
    categorias_nuevas_test = {
        c: sorted(set(te[c].astype(str)) - set(tr[c].astype(str))) for c in CATEGORICAS
    }

    auditoria = {
        "fuente_oficial": FUENTE_OFICIAL,
        "acceso": "Mirror de terceros fijado por commit (indicado en el enunciado). "
                  "Las huellas coinciden con el anexo; no autentican contra el archivo oficial.",
        "fecha_descarga_utc": FECHA_DESCARGA,
        "archivos": {"entrenamiento": aud_train, "test": aud_test},
        "variables": {
            "objetivo": OBJETIVO,
            "clase_positiva": "1 = ataque",
            "excluidas_del_modelo": EXCLUIDAS,
            "razon_exclusion": {
                "label": "es la respuesta",
                "attack_cat": "revela la respuesta (Normal <=> label 0)",
                "id": "identificador de fila; se conserva solo como row_id para auditoría",
            },
            "usadas": X_cols,
            "n_usadas": len(X_cols),
            "categoricas": CATEGORICAS,
            "numericas": num_cols,
        },
        "politica_limpieza": {
            "faltantes": "No hay faltantes en ningún archivo; el pipeline incluye imputación "
                         "(mediana / más frecuente) ajustada solo con entrenamiento como salvaguarda.",
            "duplicados": "Se conservan. Las filas con entradas idénticas forman un grupo que va "
                          "completo a entrenamiento o validación.",
            "contradicciones": "Se conservan y se documentan: son ruido de etiqueta irreducible "
                               "que también aparece en test. Cada grupo contradictorio queda "
                               "íntegro en una sola partición.",
            "categorias_desconocidas": "OneHotEncoder(handle_unknown='ignore') / "
                                       "OrdinalEncoder(unknown_value=-1).",
            "bom": "Los CSV inician con BOM UTF-8; se leen con encoding='utf-8-sig'.",
        },
        "particiones": splits,
        "cruces_test": cruce,
        "categorias_de_test_no_vistas_en_train": categorias_nuevas_test,
    }
    guardar_json(auditoria, EVID / "auditoria.json")
    print(f"train={len(tr)} val={len(va)} test={len(te)} | solape test={cruce}")


if __name__ == "__main__":
    np.random.seed(SEMILLA)
    main()

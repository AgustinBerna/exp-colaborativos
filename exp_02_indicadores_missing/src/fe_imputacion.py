"""src/fe_imputacion.py - Funciones compartidas para generar lags/deltas y guardar particiones."""
import json
import logging
from datetime import datetime
from pathlib import Path
import polars as pl


def get_feature_cols(df: pl.DataFrame, config: dict) -> list[str]:
    exclude = {
        config["Id_col"], config["time_col"], config["target_col"],
        "target_binario", "target_baja2", "partition_role"
    }
    # Los indicadores ya están calculados desde el valor original del mes.
    # Deben llegar intactos al modelo, sin pasar por un imputador ni generar
    # lags/deltas propios.
    suffix = config["missing_indicator_suffix"]
    return [c for c in df.columns if c not in exclude and not c.endswith(suffix)]


def build_lags_deltas_and_save(df_imputed: pl.DataFrame, feature_cols: list[str], work: Path, config: dict, method: str):
    """Calcula lag_1, lag_2, delta_1 y delta_2 por cliente y guarda train/valid/test."""
    id_col = config["Id_col"]
    time_col = config["time_col"]

    logging.info("[%s] Ordenando panel y generando lag_1, lag_2, delta_1 y delta_2 para %d variables...", method, len(feature_cols))
    # E06 concatena archivos mes a mes; sin este orden train quedaría
    # mes-cliente, mientras E0-E4 queda cliente-mes. Ordenar antes de shift
    # garantiza lags cronológicos y el mismo orden físico para LightGBM.
    df_imputed = df_imputed.sort([id_col, time_col])
    df_imputed = df_imputed.with_columns([pl.col(c).cast(pl.Float32) for c in feature_cols])

    # Expresiones vectorizadas en Polars para lag 1 y lag 2 dentro de cada cliente
    lag_exprs = []
    for c in feature_cols:
        lag1 = pl.col(c).shift(1).over(id_col)
        lag2 = pl.col(c).shift(2).over(id_col)
        lag_exprs.extend([
            lag1.alias(f"{c}_lag1"),
            lag2.alias(f"{c}_lag2"),
            (pl.col(c) - lag1).alias(f"{c}_delta1"),
            (pl.col(c) - lag2).alias(f"{c}_delta2")
        ])

    df_full = df_imputed.with_columns(lag_exprs)

    out_dir = work / method / "prepared"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Separar estrictamente en train (201903-201906), valid (201908) y test (201910)
    for role in ["train", "valid", "test"]:
        sub = df_full.filter(pl.col("partition_role") == role)
        sub_path = out_dir / f"{role}.parquet"
        sub.write_parquet(sub_path, compression="zstd")
        logging.info("[%s] Guardado %s: %d filas, %d columnas en %s", method, role, sub.height, sub.width, sub_path)

    done_path = out_dir / "DONE.json"
    done_path.write_text(json.dumps({
        "method": method,
        "timestamp": datetime.utcnow().isoformat(),
        "base_features": len(feature_cols),
        "total_columns": df_full.width
    }, indent=2), encoding="utf-8")
    logging.info("[%s] Checkpoint escrito en %s", method, done_path)

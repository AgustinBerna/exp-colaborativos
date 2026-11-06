"""E7: imputación MICE con LightGBM mediante la biblioteca miceforest.

1. Toma una muestra reproducible de train para ajustar el kernel (sin clase).
2. Guarda el kernel antes de transformar meses, para poder retomar una Spot.
3. Aplica los modelos aprendidos a cada mes, sin reentrenarlos con valid/test.
4. Guarda cada mes y después genera lags/deltas con la función compartida.

Se crea un solo dataset imputado, con predictive mean matching de 5 donantes.
La muestra limita únicamente el fit del imputador; el modelo final sigue usando
todas las filas de train. Las 30 semillas del modelo no se usan en el imputador.
"""
import argparse
import json
import logging
from pathlib import Path

import joblib
import miceforest as mf
import numpy as np
import pandas as pd
import polars as pl

from src.fe_imputacion import build_lags_deltas_and_save, get_feature_cols
from src.remote import upload, upload_files_then_marker


METHOD = "miceforest_6_single_pmm5"
SEED = 100003


def main():
    parser = argparse.ArgumentParser(description="E7: imputación MICE con miceforest")
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--fit_sample_size", type=int, default=20000)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--trees", type=int, default=50,
                        help="Árboles LightGBM por variable y ronda MICE")
    args = parser.parse_args()
    if min(args.fit_sample_size, args.iterations, args.trees) < 1:
        raise ValueError("fit_sample_size, iterations y trees deben ser positivos")

    work = args.work.resolve()
    log_path = work / "logs" / "03_imputar_E7.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
                        format="%(asctime)s %(message)s")

    source = work / "E7" / "source"
    marker = source / "METHOD.json"
    expected = {"method": METHOD, "fit_sample_size": args.fit_sample_size,
                "iterations": args.iterations, "trees": args.trees, "seed": SEED}
    # No mezclar los checkpoints del E7 anterior (IterativeImputer) con miceforest.
    if marker.exists():
        if json.loads(marker.read_text(encoding="utf-8")) != expected:
            raise RuntimeError("E7/source/METHOD.json difiere de esta configuración; use otro --work")
    elif (work / "E7").exists() and any((work / "E7").rglob("*")):
        raise RuntimeError("E7 contiene artefactos anteriores sin METHOD.json; use otro --work")
    else:
        source.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps(expected, indent=2), encoding="utf-8")
    upload(marker)

    if (work / "E7" / "prepared" / "DONE.json").exists():
        logging.info("[E7] Preparación miceforest ya terminada")
        return

    config = json.loads(args.config.read_text(encoding="utf-8"))
    df = pl.read_parquet(work / "partition" / "panel_2019.parquet")
    features = get_feature_cols(df, config)
    train = df.filter(pl.col("partition_role") == "train")
    if train.height > args.fit_sample_size:
        train = train.sample(n=args.fit_sample_size, seed=SEED)
    fit = train.select(features).to_pandas().astype(np.float32)

    # Una columna completamente vacía en train no ofrece donantes. Una columna
    # constante se puede completar sin entrenar un árbol para ella.
    all_empty = [c for c in features if fit[c].notna().sum() == 0]
    constants = {c: float(fit[c].dropna().iloc[0]) for c in features
                 if c not in all_empty and fit[c].dropna().nunique() == 1}
    model_features = [c for c in features if c not in all_empty]
    fit = fit[model_features].fillna(value=constants)
    impute_cols = [c for c in model_features if c not in constants
                   and df[c].null_count() > 0]
    kernel_path = source / "miceforest_kernel.joblib"

    if kernel_path.exists():
        bundle = joblib.load(kernel_path)
        if bundle["model_features"] != model_features or bundle["impute_cols"] != impute_cols:
            raise RuntimeError("Las columnas de E7 cambiaron; use otro --work")
        kernel = bundle["kernel"]
        logging.info("[E7] Kernel recuperado: %s", kernel_path)
    else:
        kernel = None
        if impute_cols:
            kernel = mf.ImputationKernel(
                fit, num_datasets=1, variable_schema=impute_cols,
                mean_match_candidates=5, save_all_iterations_data=True,
                random_state=SEED,
            )
            logging.info("[E7] Ajustando %d variables sobre %d filas de train",
                         len(impute_cols), len(fit))
            kernel.mice(iterations=args.iterations, num_iterations=args.trees,
                        num_threads=10, verbosity=-1)
        joblib.dump({"kernel": kernel, "model_features": model_features,
                     "impute_cols": impute_cols, "constants": constants}, kernel_path)
        logging.info("[E7] Kernel guardado: %s", kernel_path)
    upload(kernel_path)

    # Cada mes es recuperable por separado. Los modelos y donantes proceden
    # del fit de train; valid/test no participan en el ajuste del kernel.
    monthly = []
    for month in sorted(config["all_months"]):
        month_path = source / f"{month}.parquet"
        month_done = source / f"{month}.DONE.json"
        if month_done.exists() and month_path.exists():
            upload_files_then_marker([month_path], month_done)
            monthly.append(pl.read_parquet(month_path))
            logging.info("[E7] Mes %s recuperado", month)
            continue
        part = df.filter(pl.col(config["time_col"]) == month)
        original = part.select(features).to_pandas().astype(np.float32)
        updated = original.copy()
        updated.fillna(value=constants, inplace=True)
        if kernel is not None:
            new_data = updated[model_features]
            completed = kernel.impute_new_data(
                new_data, datasets=[0], iterations=args.iterations,
                save_all_iterations_data=True, random_state=SEED + int(month),
            ).complete_data(dataset=0)
            # miceforest conserva observados; este reemplazo explícito lo audita.
            for col in impute_cols:
                updated[col] = original[col].fillna(completed[col])
        part = part.with_columns([pl.Series(name=c, values=updated[c].to_numpy())
                                  for c in features])
        part.write_parquet(month_path, compression="zstd")
        month_done.write_text(json.dumps({"month": month, "rows": part.height,
                                          "method": METHOD}), encoding="utf-8")
        upload_files_then_marker([month_path], month_done)
        monthly.append(part)
        logging.info("[E7] Mes %s terminado: %d filas", month, part.height)

    panel = pl.concat(monthly, how="vertical_relaxed")
    build_lags_deltas_and_save(panel, features, work, config, method="E7")


if __name__ == "__main__":
    main()

"""E06: imputación MICE con LightGBM mediante la biblioteca miceforest.

1. Usa todas las filas de train para ajustar el kernel (sin clase).
2. Guarda el kernel antes de transformar meses, para poder retomar una Spot.
3. Aplica los modelos aprendidos a cada mes, sin reentrenarlos con valid/test.
4. Guarda cada mes y después genera lags/deltas con la función compartida.

Se crea un solo dataset imputado, con predictive mean matching de 5 donantes.
Una muestra menor sólo se usa si se solicita explícitamente. El kernel se
guarda después de cada ronda MICE para retomar una VM Spot interrumpida.
Las 30 semillas del modelo predictivo no se usan en el imputador.
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
    parser = argparse.ArgumentParser(description="E06: imputación MICE con miceforest")
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--method", choices=["E06"], default="E06")
    parser.add_argument("--fit_sample_size", type=int, default=0,
                        help="0 = todo train (por defecto); >0 limita sólo el fit del imputador")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--lgbm_num_iterations", type=int, default=100,
                        help="Iteraciones de boosting LightGBM por variable y ronda MICE")
    args = parser.parse_args()
    if args.fit_sample_size < 0 or min(args.iterations, args.lgbm_num_iterations) < 1:
        raise ValueError("fit_sample_size >= 0 e iterations/lgbm_num_iterations positivos")

    work = args.work.resolve()
    method = args.method
    log_path = work / "logs" / f"03_imputar_{method}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
                        format="%(asctime)s %(message)s")

    source = work / method / "source"
    marker = source / "METHOD.json"
    expected = {"method": METHOD, "output_method": method,
                "fit_sample_size": args.fit_sample_size,
                "iterations": args.iterations,
                "lgbm_num_iterations": args.lgbm_num_iterations, "seed": SEED,
                "data_subset": 0}
    # No mezclar checkpoints de otro imputador bajo el mismo ID de método.
    if marker.exists():
        if json.loads(marker.read_text(encoding="utf-8")) != expected:
            raise RuntimeError(f"{method}/source/METHOD.json difiere de esta configuración")
    elif (work / method).exists() and any((work / method).rglob("*")):
        raise RuntimeError(f"{method} contiene artefactos anteriores sin METHOD.json")
    else:
        source.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps(expected, indent=2), encoding="utf-8")
    upload(marker)

    if (work / method / "prepared" / "DONE.json").exists():
        logging.info("[%s] Preparación miceforest ya terminada", method)
        return

    config = json.loads(args.config.read_text(encoding="utf-8"))
    df = pl.read_parquet(work / "partition" / "panel_2019.parquet")
    features = get_feature_cols(df, config)
    train = df.filter(pl.col("partition_role") == "train")
    train = train.sort([config["Id_col"], config["time_col"]])
    if args.fit_sample_size and train.height > args.fit_sample_size:
        train = train.sample(n=args.fit_sample_size, seed=SEED)
    fit = train.select(features).to_pandas().astype(np.float32)
    logging.info("[%s] Filas para fit: %d de %d filas train; data_subset=0",
                 method, len(fit), df.filter(pl.col("partition_role") == "train").height)

    # Una columna completamente vacía en train no ofrece donantes. Una columna
    # constante se puede completar sin entrenar un árbol para ella.
    all_empty = [c for c in features if fit[c].notna().sum() == 0]
    constants = {c: float(fit[c].dropna().iloc[0]) for c in features
                 if c not in all_empty and fit[c].dropna().nunique() == 1}
    logging.info("[%s] Columnas sin donantes en train: %s", method, ", ".join(all_empty) or "(ninguna)")
    logging.info("[%s] Columnas constantes en train: %s", method, ", ".join(constants) or "(ninguna)")
    model_features = [c for c in features if c not in all_empty]
    fit = fit[model_features].fillna(value=constants)
    impute_cols = [c for c in model_features if c not in constants
                   and df[c].null_count() > 0]
    kernel_path = source / "miceforest_kernel.joblib"

    if kernel_path.exists():
        bundle = joblib.load(kernel_path)
        if (bundle["model_features"] != model_features or
                bundle["impute_cols"] != impute_cols or
                bundle["constants"] != constants or
                bundle["all_empty"] != all_empty):
            raise RuntimeError(f"Las columnas de {method} cambiaron; use otro --work")
        kernel = bundle["kernel"]
        constants = bundle["constants"]
        logging.info("[%s] Kernel recuperado: %s", method, kernel_path)
    else:
        kernel = None
        if impute_cols:
            kernel = mf.ImputationKernel(
                fit, num_datasets=1, variable_schema=impute_cols,
                mean_match_candidates=5, data_subset=0,
                save_all_iterations_data=True,
                random_state=SEED,
            )
    # variable_schema entrena también variables completas en train que tienen
    # faltantes en otro mes; miceforest 6 no requiere train_nonmissing.
    completed = int(kernel.iteration_count()) if kernel is not None else 0
    if completed > args.iterations:
        raise RuntimeError("El checkpoint tiene más rondas que la configuración")
    for iteration in range(completed, args.iterations if kernel is not None else 0):
        logging.info("[%s] Ronda MICE %d/%d: %d variables, %d filas train",
                     method, iteration + 1, args.iterations, len(impute_cols), len(fit))
        kernel.mice(iterations=1, num_iterations=args.lgbm_num_iterations,
                    num_threads=10, verbosity=-1, verbose=False)
        partial = kernel_path.with_suffix(".partial")
        joblib.dump({"kernel": kernel, "model_features": model_features,
                     "impute_cols": impute_cols, "constants": constants,
                     "all_empty": all_empty}, partial)
        partial.replace(kernel_path)
        upload(kernel_path)
        logging.info("[%s] Checkpoint de ronda %d guardado en %s", method, iteration + 1, kernel_path)
    if kernel is None and not kernel_path.exists():
        joblib.dump({"kernel": None, "model_features": model_features,
                     "impute_cols": impute_cols, "constants": constants,
                     "all_empty": all_empty}, kernel_path)
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
            logging.info("[%s] Mes %s recuperado", method, month)
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
                verbose=False,
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
        logging.info("[%s] Mes %s terminado: %d filas", method, month, part.height)

    panel = pl.concat(monthly, how="vertical_relaxed")
    build_lags_deltas_and_save(panel, features, work, config, method=method)


if __name__ == "__main__":
    main()

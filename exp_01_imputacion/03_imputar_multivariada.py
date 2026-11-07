"""E5: imputación iterativa multivariada con BayesianRidge.

1. Ajusta scaler e imputador en una muestra configurable de train.
2. Guarda el imputador y lo copia al bucket.
3. Transforma cada mes y publica parquet antes de su DONE.json.
4. Genera lags/deltas y particiones con la función compartida.

Produce una imputación única, no múltiples datasets imputados. La muestra de
fit no usa la clase y no altera el entrenamiento completo del modelo final.
"""
import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler

from src.fe_imputacion import get_feature_cols, build_lags_deltas_and_save
from src.remote import upload, upload_files_then_marker


def main():
    parser = argparse.ArgumentParser(description="E5: imputación iterativa Ridge")
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--method", choices=["E5"], default="E5")
    parser.add_argument("--algo", choices=["mice_ridge"], default="mice_ridge")
    parser.add_argument("--fit_sample_size", type=int, default=60000)
    parser.add_argument("--max_iter", type=int, default=10,
                        help="Máximo de rondas de IterativeImputer")
    parser.add_argument("--n_nearest_features", type=int, default=25,
                        help="Cantidad de variables predictoras candidatas por imputación")
    args = parser.parse_args()
    if min(args.fit_sample_size, args.max_iter, args.n_nearest_features) < 1:
        raise ValueError("fit_sample_size, max_iter y n_nearest_features deben ser positivos")

    work = args.work.resolve()
    log_path = work / "logs" / "03_imputar_E5.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
                        format="%(asctime)s %(message)s")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    df = pl.read_parquet(work / "partition" / "panel_2019.parquet")
    features = get_feature_cols(df, config)
    source = work / "E5" / "source"
    source.mkdir(parents=True, exist_ok=True)
    imputer_path = source / "ridge_imputer.joblib"
    params_path = source / "PARAMS.json"
    expected = {"method": "E5", "algo": args.algo,
                "fit_sample_size": args.fit_sample_size,
                "max_iter": args.max_iter,
                "n_nearest_features": args.n_nearest_features,
                "seed": 100003}
    if params_path.exists():
        if json.loads(params_path.read_text(encoding="utf-8")) != expected:
            raise RuntimeError("E5/PARAMS.json difiere: usar otro --work")
    elif imputer_path.exists() or any(source.glob("*.parquet")):
        raise RuntimeError("E5 tiene checkpoints sin PARAMS.json: usar otro --work")
    else:
        params_path.write_text(json.dumps(expected, indent=2), encoding="utf-8")
    upload(params_path)
    if (work / "E5" / "prepared" / "DONE.json").exists():
        logging.info("[E5] Preparación ya terminada con estos parámetros")
        return

    if imputer_path.exists():
        bundle = joblib.load(imputer_path)
        scaler, imputer = bundle["scaler"], bundle["imputer"]
        logging.info("[E5] Imputador recuperado: %s", imputer_path)
    else:
        train = df.filter(pl.col("partition_role") == "train")
        if train.height > args.fit_sample_size:
            train = train.sample(n=args.fit_sample_size, seed=100003)
        X_train = train.select(features).to_numpy().astype(np.float32)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        empty = np.isnan(X_train).all(axis=0)
        scaler.mean_[empty], scaler.scale_[empty], scaler.var_[empty] = 0.0, 1.0, 0.0
        imputer = IterativeImputer(
            estimator=BayesianRidge(), max_iter=args.max_iter, initial_strategy="median",
            n_nearest_features=args.n_nearest_features,
            keep_empty_features=True, random_state=100003,
        )
        logging.info("[E5] Ajustando con %d filas, max_iter=%d, n_nearest_features=%d",
                     len(X_train), args.max_iter, args.n_nearest_features)
        imputer.fit(X_scaled)
        joblib.dump({"scaler": scaler, "imputer": imputer}, imputer_path)
        logging.info("[E5] Imputador guardado: %s", imputer_path)
    upload(imputer_path)

    monthly = []
    for month in sorted(config["all_months"]):
        month_path = source / f"{month}.parquet"
        month_done = source / f"{month}.DONE.json"
        if month_done.exists() and month_path.exists():
            upload_files_then_marker([month_path], month_done)
            monthly.append(pl.read_parquet(month_path))
            logging.info("[E5] Mes %s recuperado", month)
            continue
        part = df.filter(pl.col(config["time_col"]) == month)
        X = part.select(features).to_numpy().astype(np.float32)
        predicted = scaler.inverse_transform(imputer.transform(scaler.transform(X)))
        result = np.where(np.isnan(X), predicted, X).astype(np.float32)
        part = part.with_columns([pl.Series(name=col, values=result[:, i])
                                  for i, col in enumerate(features)])
        part.write_parquet(month_path, compression="zstd")
        month_done.write_text(json.dumps({"month": month, "rows": part.height}), encoding="utf-8")
        upload_files_then_marker([month_path], month_done)
        monthly.append(part)
        logging.info("[E5] Mes %s terminado: %d filas", month, part.height)

    panel = pl.concat(monthly, how="vertical_relaxed")
    build_lags_deltas_and_save(panel, features, work, config, method="E5")


if __name__ == "__main__":
    main()

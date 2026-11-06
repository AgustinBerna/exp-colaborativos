"""E6: autoencoder de denoising para imputar faltantes sin usar la clase.

Se ajusta solo con filas de train. En cada época oculta valores observados al
azar y minimiza el error de reconstrucción únicamente en esos valores. Esto
evita entrenar contra valores imputados como si fueran la verdad. El modelo,
el scaler y cada mes transformado se publican antes de su marcador DONE.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from sklearn.preprocessing import MinMaxScaler

from src.fe_imputacion import get_feature_cols, build_lags_deltas_and_save
from src.remote import upload_files_then_marker


def build_model(n_features: int):
    import tensorflow as tf

    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n_features,)),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(n_features, activation="sigmoid"),
    ])


def fit_autoencoder(X_train: np.ndarray, epochs: int, batch_size: int,
                    hide_fraction: float, seed: int):
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    rng = np.random.default_rng(seed)
    observed = np.isfinite(X_train)
    empty = ~observed.any(axis=0)

    # El scaler no puede aprender extremos de columnas totalmente vacías.
    X_for_scaler = X_train.copy()
    X_for_scaler[:, empty] = 0.0
    scaler = MinMaxScaler()
    scaler.fit(X_for_scaler)
    scaled = np.clip(scaler.transform(X_train), 0.0, 1.0).astype(np.float32)
    medians = np.nanmedian(scaled, axis=0).astype(np.float32)
    medians[~np.isfinite(medians)] = 0.5

    model = build_model(X_train.shape[1])
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    for epoch in range(epochs):
        order = rng.permutation(len(scaled))
        losses = []
        for start in range(0, len(order), batch_size):
            batch = scaled[order[start:start + batch_size]]
            known = np.isfinite(batch)
            hidden = known & (rng.random(batch.shape) < hide_fraction)
            if not hidden.any():
                continue
            filled = np.where(known, batch, medians)
            corrupted = np.where(hidden, medians, filled).astype(np.float32)
            targets = np.where(known, batch, 0.0).astype(np.float32)
            mask = hidden.astype(np.float32)
            with tf.GradientTape() as tape:
                predictions = model(corrupted, training=True)
                loss = tf.reduce_sum(tf.square(predictions - targets) * mask) / tf.reduce_sum(mask)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            losses.append(float(loss))
        logging.info("[E6] Época %d/%d: MSE en observados ocultos %.6f",
                     epoch + 1, epochs, float(np.mean(losses)))
    return model, scaler, medians


def impute_month(model, scaler, medians: np.ndarray, X: np.ndarray,
                 batch_size: int) -> np.ndarray:
    """Predice faltantes y devuelve exactamente los valores observados."""
    missing = ~np.isfinite(X)
    scaled = np.clip(scaler.transform(X), 0.0, 1.0).astype(np.float32)
    filled = np.where(np.isfinite(scaled), scaled, medians).astype(np.float32)
    predicted = model.predict(filled, batch_size=batch_size, verbose=0)
    imputed = scaler.inverse_transform(predicted).astype(np.float32)
    return np.where(missing, imputed, X)


def main():
    parser = argparse.ArgumentParser(description="E6: autoencoder de denoising")
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--method", choices=["E6"], default="E6")
    parser.add_argument("--fit_sample_size", type=int, default=100000)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--hide_fraction", type=float, default=0.15)
    args = parser.parse_args()
    if args.fit_sample_size < 1 or args.epochs < 1 or args.batch_size < 1:
        raise ValueError("Muestra, épocas y batch_size deben ser positivos")
    if not 0 < args.hide_fraction < 1:
        raise ValueError("hide_fraction debe estar entre 0 y 1")

    # TensorFlow estable 2.21 ofrece ruedas para Python 3.13 en Linux/Windows.
    try:
        import tensorflow as tf
    except ImportError as error:
        raise RuntimeError("E6 requiere Python 3.13 y tensorflow==2.21.0") from error

    work = args.work.resolve()
    log_path = work / "logs" / "03_imputar_E6.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
                        format="%(asctime)s %(message)s")
    if (work / "E6" / "prepared" / "DONE.json").exists():
        logging.info("[E6] Preparación ya terminada")
        return

    config = json.loads(args.config.read_text(encoding="utf-8"))
    df = pl.read_parquet(work / "partition" / "panel_2019.parquet")
    features = get_feature_cols(df, config)
    source = work / "E6" / "source"
    source.mkdir(parents=True, exist_ok=True)
    model_path = source / "autoencoder.keras"
    scaler_path = source / "scaler.joblib"
    fit_done = source / "FIT.DONE.json"

    if fit_done.exists() and model_path.exists() and scaler_path.exists():
        model = tf.keras.models.load_model(model_path, compile=False)
        bundle = joblib.load(scaler_path)
        scaler, medians = bundle["scaler"], bundle["medians"]
        logging.info("[E6] Modelo recuperado de %s", model_path)
    else:
        train = df.filter(pl.col("partition_role") == "train")
        if train.height > args.fit_sample_size:
            train = train.sample(n=args.fit_sample_size, seed=100003)
        X_train = train.select(features).to_numpy().astype(np.float32)
        logging.info("[E6] Ajustando con %d filas de train", len(X_train))
        model, scaler, medians = fit_autoencoder(
            X_train, args.epochs, args.batch_size, args.hide_fraction, 100003)
        model.save(model_path)
        joblib.dump({"scaler": scaler, "medians": medians}, scaler_path)
        fit_done.write_text(json.dumps({"rows": len(X_train), "epochs": args.epochs,
                                        "hide_fraction": args.hide_fraction}), encoding="utf-8")
    upload_files_then_marker([model_path, scaler_path], fit_done)

    monthly = []
    for month in sorted(config["all_months"]):
        month_path = source / f"{month}.parquet"
        month_done = source / f"{month}.DONE.json"
        if month_done.exists() and month_path.exists():
            logging.info("[E6] Mes %s recuperado", month)
            upload_files_then_marker([month_path], month_done)
            monthly.append(pl.read_parquet(month_path))
            continue
        part = df.filter(pl.col(config["time_col"]) == month)
        X = part.select(features).to_numpy().astype(np.float32)
        result = impute_month(model, scaler, medians, X, args.batch_size)
        part = part.with_columns([pl.Series(name=col, values=result[:, i])
                                  for i, col in enumerate(features)])
        part.write_parquet(month_path, compression="zstd")
        month_done.write_text(json.dumps({"month": month, "rows": part.height}), encoding="utf-8")
        upload_files_then_marker([month_path], month_done)
        monthly.append(part)
        logging.info("[E6] Mes %s terminado: %d filas", month, part.height)

    panel = pl.concat(monthly, how="vertical_relaxed")
    build_lags_deltas_and_save(panel, features, work, config, method="E6")


if __name__ == "__main__":
    main()

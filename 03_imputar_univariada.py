"""03_imputar_univariada.py - Imputación Univariada E1 (Media), E2 (Mediana), E3 (Moda).

Uso:
  python 03_imputar_univariada.py --work ./work --method E1  # Media
  python 03_imputar_univariada.py --work ./work --method E2  # Mediana
  python 03_imputar_univariada.py --work ./work --method E3  # Moda (most_frequent)

Proceso paso a paso:
  1. Carga partition/panel_2019.parquet.
  2. Filtra exclusivamente los meses de entrenamiento (201903 a 201906) para hacer el .fit()
     del SimpleImputer de scikit-learn (evitando data leakage de validación y test).
  3. Transforma todos los meses con los estadísticos aprendidos en train.
  4. Calcula lag_1, lag_2, delta_1 y delta_2 y escribe <method>/prepared/DONE.json.
"""
import argparse, json, logging, joblib
from pathlib import Path
import numpy as np
import polars as pl
from sklearn.impute import SimpleImputer
from src.fe_imputacion import get_feature_cols, build_lags_deltas_and_save

STRATEGY_MAP = {
    "E1": "mean",
    "E2": "median",
    "E3": "most_frequent"
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--method", choices=["E1", "E2", "E3"], required=True)
    args = parser.parse_args()

    work = args.work.resolve()
    method = args.method
    strategy = STRATEGY_MAP[method]

    log_path = work / "logs" / f"03_imputar_{method}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(log_path), logging.StreamHandler()], format="%(asctime)s %(message)s")

    if (work / method / "prepared" / "DONE.json").exists():
        logging.info("%s ya cuenta con DONE.json. Omitiendo.", method)
        return

    config = json.loads(args.config.read_text(encoding="utf-8"))
    df = pl.read_parquet(work / "partition" / "panel_2019.parquet")
    fcols = get_feature_cols(df, config)

    # Extraer matriz de entrenamiento (201903-201906) para ajustar el imputador sin leakage
    train_mask = (df["partition_role"] == "train").to_numpy()
    X_all = df.select(fcols).to_numpy().astype(np.float32)

    logging.info("[%s] Ajustando SimpleImputer(strategy='%s') solo en train (%d filas)...", method, strategy, train_mask.sum())
    # Keep columns that are entirely missing in train so feature order and
    # dimensionality remain identical across methods and months.
    imputer = SimpleImputer(strategy=strategy, keep_empty_features=True)
    imputer.fit(X_all[train_mask])

    # Guardar los valores de imputación aprendidos para auditoría
    model_dir = work / method / "imputer"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(imputer, model_dir / "simple_imputer.joblib")

    logging.info("[%s] Transformando el panel completo...", method)
    X_imputed = imputer.transform(X_all)

    # Reemplazar columnas imputadas en Polars y construir lags/deltas
    df_imputed = df.with_columns([
        pl.Series(name=col, values=X_imputed[:, idx]) for idx, col in enumerate(fcols)
    ])
    build_lags_deltas_and_save(df_imputed, fcols, work, config, method=method)

if __name__ == "__main__":
    main()

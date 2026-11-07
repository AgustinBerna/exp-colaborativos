"""03_imputar_E0_baseline.py - Experimento E0 (Baseline sin imputación explícita).

Deja los NaN intactos (incluyendo los ceros masivos convertidos en 01_auditar.py)
para que LightGBM aprenda la dirección por defecto en cada split, y genera
lag_1, lag_2, delta_1 y delta_2.
Salida: E0/prepared/{train,valid,test}.parquet y E0/prepared/DONE.json.
"""
import argparse, json, logging
from pathlib import Path
import polars as pl
from src.fe_imputacion import get_feature_cols, build_lags_deltas_and_save

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    args = parser.parse_args()

    work = args.work.resolve()
    (work / "logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, filename=work / "logs" / "03_imputar_E0.log", format="%(asctime)s %(message)s")
    config = json.loads(args.config.read_text(encoding="utf-8"))

    if (work / "E0" / "prepared" / "DONE.json").exists():
        logging.info("E0 ya completado.")
        return

    df = pl.read_parquet(work / "partition" / "panel_2019.parquet")
    fcols = get_feature_cols(df, config)
    build_lags_deltas_and_save(df, fcols, work, config, method="E0")

if __name__ == "__main__":
    main()

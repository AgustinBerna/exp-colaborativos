"""03_imputar_E4_temporal.py - Experimento E4: Imputación Temporal Causal (Forward Fill).

Proceso paso a paso:
  1. Ordena la base por (numero_de_cliente, foto_mes).
  2. Aplica Forward Fill (.forward_fill().over('numero_de_cliente')) para que cada NaN
     tome el último valor válido conocido de ese mismo cliente en meses previos.
  3. NUNCA aplica Backward Fill para no filtrar información futura del cliente.
  4. Calcula lag_1, lag_2, delta_1 y delta_2 y guarda E4/prepared/DONE.json.
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
    log_path = work / "logs" / "03_imputar_E4.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(log_path), logging.StreamHandler()], format="%(asctime)s %(message)s")

    if (work / "E4" / "prepared" / "DONE.json").exists():
        logging.info("E4 ya completado.")
        return

    config = json.loads(args.config.read_text(encoding="utf-8"))
    id_col = config["Id_col"]
    time_col = config["time_col"]

    df = pl.read_parquet(work / "partition" / "panel_2019.parquet").sort([id_col, time_col])
    fcols = get_feature_cols(df, config)

    logging.info("[E4] Aplicando Forward Fill por cliente hacia adelante en el tiempo...")
    df_ffill = df.with_columns([
        pl.col(c).forward_fill().over(id_col).alias(c) for c in fcols
    ])

    build_lags_deltas_and_save(df_ffill, fcols, work, config, method="E4")

if __name__ == "__main__":
    main()
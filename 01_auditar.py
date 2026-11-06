"""01_auditar.py - conversión de 100% ceros a NaN y auditoría de % NaN.

Entradas:
  - Dataset crudo CSV/Parquet (--data) y config.json (--work).
Proceso paso a paso:
  1. Verifica si existe el checkpoint raw/DONE.json. Si existe, omite el proceso.
  2. Lee el dataset con Polars y conserva 201901 a 201912.
     Noviembre y diciembre se necesitan para resolver las clases de octubre;
     no se usarán como filas de train, validación ni test.
  3. Recorre cada mes (201901 a 201912):
     a. Identifica columnas numéricas donde el 100% de las filas son exactamente 0.
     b. Reemplaza esos ceros masivos por NaN/Null para que sean tratados por la imputación.
     c. Calcula el % de NaN por variable para ese mes.
     d. Guarda el mes limpio en formato columnar comprimido: raw/<mes>.parquet.
Salidas y Checkpoints:
  - Log detallado: logs/auditar.log
  - Reporte único de % NaN: raw/reporte_nan_por_mes.csv
  - Detalle de variables rotas (100% ceros): raw/variables_100pct_ceros.csv
  - Particiones mensuales limpias: raw/<foto_mes>.parquet
  - Checkpoint de cierre: raw/DONE.json
"""
import argparse
import json
import hashlib
import gzip
import logging
import shutil
from datetime import datetime
from pathlib import Path
import polars as pl
import numpy as np


def setup_logger(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()]
    )


def write_checkpoint(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload["timestamp"] = datetime.utcnow().isoformat()
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def ensure_numeric_features(df: pl.DataFrame, feature_cols: list[str]) -> pl.DataFrame:
    """Parse numeric-looking text columns, rejecting genuine categories."""
    text_features = [c for c in feature_cols if df.schema[c] == pl.String]
    if text_features:
        logging.info("Convirtiendo variables numéricas leídas como texto: %s", text_features)
        df = df.with_columns([pl.col(c).cast(pl.Float64, strict=True).alias(c) for c in text_features])
    non_numeric = [c for c in feature_cols if not df.schema[c].is_numeric()]
    if non_numeric:
        raise TypeError(f"Variables no numéricas sin codificación definida: {non_numeric}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Etapa 01: Auditoría y carga óptima")
    parser.add_argument("--work", type=Path, required=True, help="Directorio de trabajo (ej. ./work)")
    parser.add_argument("--config", type=Path, default=Path("config.json"), help="Ruta al config.json")
    parser.add_argument("--data", type=Path, required=True, help="Ruta al CSV/CSV.GZ de entrada")
    args = parser.parse_args()

    work = args.work.resolve()
    setup_logger(work / "logs" / "01_auditar.log")
    config = json.loads(args.config.read_text(encoding="utf-8"))

    done_file = work / "raw" / "DONE.json"
    if done_file.exists():
        logging.info("Checkpoint detectado en %s. Etapa 01 ya completada.", done_file)
        return

    csv_path = args.data.resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"No se encontró el archivo de entrada: {csv_path}")

    raw_dir = work / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    id_col = config["Id_col"]
    time_col = config["time_col"]
    target_col = config["target_col"]
    valid_months = config["all_months"]

    # Polars receives an uncompressed path. For CSV.GZ, decompress once to
    # local work/input; the cache is never uploaded to Cloud Storage.
    read_path = csv_path
    if csv_path.suffix.lower() == ".gz":
        input_dir = work / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        read_path = input_dir / "data.csv"
        partial = input_dir / "data.csv.partial"
        if not read_path.exists():
            with gzip.open(csv_path, "rb") as source, partial.open("wb") as target:
                shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
            partial.replace(read_path)
    logging.info("Iniciando lectura con Polars desde: %s", read_path)
    # El crudo mezcla IDs enteros y notación científica (por ejemplo 4.8e+07).
    # Se leen como Float64 para aceptar ambas formas, y enseguida se verifica
    # que cada valor pueda convertirse sin pérdida a un ID entero.
    df = pl.read_csv(read_path, infer_schema_length=50000, ignore_errors=False,
                     schema_overrides={id_col: pl.Float64, time_col: pl.Int64})
    if read_path != csv_path:
        read_path.unlink(missing_ok=True)
    ids = df[id_col].to_numpy()
    if (not np.isfinite(ids).all() or (np.abs(ids) > 2**53).any()
            or (np.floor(ids) != ids).any()):
        raise ValueError("numero_de_cliente contiene valores no enteros o no representables exactamente")
    df = df.with_columns(pl.col(id_col).cast(pl.Int64))

    # Conservamos todo 2019: noviembre y diciembre aportan presencia futura.
    df = df.filter(pl.col(time_col).is_in(valid_months))
    logging.info("Filas retenidas en meses válidos (%s): %d", valid_months, df.height)

    # Algunas variables numéricas del crudo llegan como texto porque comienzan
    # con muchos meses vacíos (por ejemplo tmobile_app y cmobile_app_trx).
    # Convertimos solo esas columnas a números; un valor no numérico real debe
    # fallar aquí en vez de convertirse silenciosamente a un faltante.
    meta_cols = {id_col, time_col, target_col}
    feature_cols = [c for c in df.columns if c not in meta_cols]
    df = ensure_numeric_features(df, feature_cols)

    nan_records = []
    zero_broken_records = []

    # Procesamos mes a mes para detectar 100% ceros, pasar a Null/NaN y guardar Parquet
    for mes in sorted(valid_months):
        df_mes = df.filter(pl.col(time_col) == mes)
        n_rows = df_mes.height
        if n_rows == 0:
            raise RuntimeError(f"El mes {mes} no contiene registros en el dataset.")

        # Verificar unicidad de clave primaria (numero_de_cliente, foto_mes)
        n_unique = df_mes.select(pl.col(id_col).n_unique()).item()
        if n_unique != n_rows:
            raise ValueError(f"Duplicados detectados en mes {mes}: {n_rows} filas vs {n_unique} clientes.")

        # 1. Detectar variables que tienen 0 en el 100% de las filas del mes
        cols_to_nullify = []
        for col in feature_cols:
            # Contamos cuántos valores son exactamente 0
            zero_count = df_mes.select((pl.col(col) == 0).sum()).item()
            if zero_count == n_rows:
                cols_to_nullify.append(col)
                zero_broken_records.append({
                    "foto_mes": mes,
                    "variable": col,
                    "filas_mes": n_rows,
                    "accion": "100%_ceros_convertido_a_NaN"
                })

        # 2. Reemplazar por Null (NaN) las columnas con 100% ceros en este mes
        if cols_to_nullify:
            logging.info("Mes %s: %d variables con 100%% ceros convertidas a NaN.", mes, len(cols_to_nullify))
            df_mes = df_mes.with_columns([
                pl.lit(None).cast(df_mes[c].dtype).alias(c) for c in cols_to_nullify
            ])

        # 3. Calcular % de NaN por variable (tras la conversión de ceros masivos)
        null_counts = df_mes.select([pl.col(c).null_count().alias(c) for c in feature_cols]).row(0, named=True)
        for col, n_nulls in null_counts.items():
            nan_records.append({
                "foto_mes": mes,
                "variable": col,
                "n_filas": n_rows,
                "n_nan": n_nulls,
                "pct_nan": round((n_nulls / n_rows) * 100.0, 4),
                "convertida_de_100pct_cero": col in cols_to_nullify
            })

        # 4. Guardar el mes auditado en Parquet
        out_parquet = raw_dir / f"{mes}.parquet"
        df_mes.write_parquet(out_parquet, compression="zstd")
        logging.info("Mes %s guardado en %s (%d filas).", mes, out_parquet, n_rows)

    # Exportar reportes únicos de auditoría
    nan_df = pl.DataFrame(nan_records)
    nan_csv_path = raw_dir / "reporte_nan_por_mes.csv"
    nan_df.write_csv(nan_csv_path)

    zeros_df = pl.DataFrame(zero_broken_records) if zero_broken_records else pl.DataFrame(
        schema={"foto_mes": pl.Int64, "variable": pl.Utf8, "filas_mes": pl.Int64, "accion": pl.Utf8}
    )
    zeros_csv_path = raw_dir / "variables_100pct_ceros.csv"
    zeros_df.write_csv(zeros_csv_path)

    # Escribir checkpoint final de la etapa 01
    write_checkpoint(done_file, {
        "stage": "01_auditar",
        "months_processed": valid_months,
        "total_features": len(feature_cols),
        "total_zero_conversions": len(zero_broken_records),
        "nan_report": str(nan_csv_path),
        "zeros_report": str(zeros_csv_path)
    })
    logging.info("Auditoría completada con éxito. Reporte NaN en: %s", nan_csv_path)


if __name__ == "__main__":
    main()

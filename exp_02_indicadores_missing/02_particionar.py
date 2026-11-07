"""02_particionar.py - Definición de particiones (Train/Valid/Test) y targets.

Entradas:
  - Archivos mensuales auditados: raw/<mes>.parquet y raw/DONE.json.
Proceso paso a paso:
  1. Verifica el checkpoint raw/DONE.json.
  2. Excluye las variables 100% faltantes en train y crea indicadores de
     ausencia para todas las variables originales restantes.
  3. Consolida el panel ordenado por (numero_de_cliente, foto_mes).
  4. Crea las columnas de target:
     - clase_ternaria se calcula a partir de la presencia futura del cliente:
       ausente en t+1 y t+2 -> BAJA+1;
       presente en t+1 y ausente en t+2 -> BAJA+2;
       presente en t+2 -> CONTINUA; horizonte insuficiente -> NULL.
     - target_binario: 1 si clase_ternaria in ('BAJA+1', 'BAJA+2'), 0 si 'CONTINUA'.
       (Usado en entrenamiento y optimización de AUC).
     - target_baja2: 1 si clase_ternaria == 'BAJA+2', 0 en caso contrario.
       (Usado en evaluación de ganancia económica en test).
  5. Etiqueta cada fila con su rol ('lag_only', 'train', 'valid', 'test', 'dashboard_only').
Salidas y Checkpoints:
  - Log: logs/02_particionar.log
  - Panel base particionado: partition/panel_2019.parquet
  - Resumen de clases por partición: partition/resumen_particiones.csv
  - Checkpoint: partition/DONE.json
"""
import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
import polars as pl


def setup_logger(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()]
    )


def main():
    parser = argparse.ArgumentParser(description="Etapa 02: Particionamiento Train / Valid / Test")
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    args = parser.parse_args()

    work = args.work.resolve()
    setup_logger(work / "logs" / "02_particionar.log")
    config = json.loads(args.config.read_text(encoding="utf-8"))

    if not (work / "raw" / "DONE.json").exists():
        raise RuntimeError("Falta raw/DONE.json. Ejecute primero 01_auditar.py")

    part_dir = work / "partition"
    done_file = part_dir / "DONE.json"
    if done_file.exists():
        logging.info("Checkpoint detectado en %s. Partición ya realizada.", done_file)
        return

    part_dir.mkdir(parents=True, exist_ok=True)
    id_col = config["Id_col"]
    time_col = config["time_col"]
    target_col = config["target_col"]

    train_months = config["train_months"]
    valid_month = config["valid_month"]
    test_month = config["test_month"]

    # Cargamos todo 2019: noviembre y diciembre permiten etiquetar octubre.
    parquet_files = [work / "raw" / f"{m}.parquet" for m in sorted(config["all_months"])]
    df = pl.concat([pl.read_parquet(f) for f in parquet_files], how="vertical_relaxed")

    # La exclusión se decide ÚNICAMENTE con marzo-junio: una variable sin
    # observaciones en train no puede aportar un efecto aprendido al modelo.
    # Se elimina del panel completo para que E0-E06 compartan las mismas columnas.
    meta = {id_col, time_col, target_col}
    candidates = [c for c in df.columns if c not in meta]
    train_df = df.filter(pl.col(time_col).is_in(train_months))
    observed_train = train_df.select([
        (pl.col(c).is_not_null() & ~pl.col(c).is_nan().fill_null(False))
        .sum().alias(c) for c in candidates
    ]).row(0, named=True)
    drop_features = [c for c in candidates if observed_train[c] == 0]
    exclusions = [{"feature": c, "observed_in_train": 0,
                   "reason": "100% faltante en meses de entrenamiento"}
                  for c in drop_features]
    if drop_features:
        df = df.drop(drop_features)
    (part_dir / "features_excluidas.json").write_text(
        json.dumps(exclusions, indent=2), encoding="utf-8")
    logging.info("Variables 100%% faltantes en train excluidas para todos los meses: %s",
                 drop_features)

    # Definimos el esquema de indicadores para TODAS las variables originales
    # restantes, sin consultar qué variables fallan en validación o test.
    # Cada indicador registra el faltante ORIGINAL del mes, antes de imputar.
    suffix = config["missing_indicator_suffix"]
    original_features = [c for c in df.columns if c not in meta]
    collisions = [c for c in original_features if c.endswith(suffix)
                  or f"{c}{suffix}" in df.columns]
    if collisions:
        raise ValueError(f"Nombres incompatibles con indicadores: {collisions}")
    missing_counts = train_df.select([
        (pl.col(c).is_null() | pl.col(c).is_nan().fill_null(False)).sum().alias(c)
        for c in original_features
    ]).row(0, named=True)
    indicators = [{"feature": col, "indicator": f"{col}{suffix}",
                   "missing_in_train": int(missing_counts[col])}
                  for col in original_features]
    if indicators:
        df = df.with_columns([
            (pl.col(row["feature"]).is_null() |
             pl.col(row["feature"]).is_nan().fill_null(False))
            .cast(pl.Int8).alias(row["indicator"])
            for row in indicators
        ])
    (part_dir / "indicadores_ausente.json").write_text(
        json.dumps(indicators, indent=2), encoding="utf-8")
    logging.info("Indicadores de ausencia añadidos (uno por variable original): %d de %d variables",
                 len(indicators), len(original_features))

    # Orden estricto por cliente y mes para garantizar lags/deltas exactos en la etapa 3
    df = df.sort([id_col, time_col])

    # Creación de targets y rol de partición
    # Convertimos YYYYMM a un índice de meses consecutivos.
    # Así diciembre + 1 apunta correctamente a enero del año siguiente.
    df = df.with_columns(
        ((pl.col(time_col) // 100) * 12 + pl.col(time_col) % 100).alias("_mes_idx")
    )

    # Una fila en t+1 o t+2 indica presencia futura del cliente.
    presencias = df.select([id_col, "_mes_idx"])
    en_t1 = presencias.with_columns(
        (pl.col("_mes_idx") - 1).alias("_mes_idx"),
        pl.lit(True).alias("_en_t1"),
    )
    en_t2 = presencias.with_columns(
        (pl.col("_mes_idx") - 2).alias("_mes_idx"),
        pl.lit(True).alias("_en_t2"),
    )

    df = (
        df.join(en_t1, on=[id_col, "_mes_idx"], how="left")
          .join(en_t2, on=[id_col, "_mes_idx"], how="left")
    )

    # Distinguimos ausencia del cliente de un mes futuro que no está disponible.
    meses_observados = {
        (mes // 100) * 12 + mes % 100
        for mes in config["all_months"]
    }
    con_t1_observable = [mes - 1 for mes in meses_observados]
    con_t2_observable = [mes - 2 for mes in meses_observados]

    p1 = pl.col("_en_t1").fill_null(False)
    p2 = pl.col("_en_t2").fill_null(False)
    t1_observable = pl.col("_mes_idx").is_in(con_t1_observable)
    t2_observable = pl.col("_mes_idx").is_in(con_t2_observable)

    # Ausente en t+1 y t+2 -> BAJA+1.
    # Presente en t+1 y ausente en t+2 -> BAJA+2.
    # Presente en t+2 -> CONTINUA.
    df = df.with_columns(
        pl.when(t2_observable & p2)
        .then(pl.lit("CONTINUA"))

        .when(t1_observable & t2_observable & p1 & ~p2)
        .then(pl.lit("BAJA+2"))

        .when(t1_observable & t2_observable & ~p1 & ~p2)
        .then(pl.lit("BAJA+1"))

        .otherwise(pl.lit(None, dtype=pl.Utf8))
        .alias(target_col)
    )

    # Una clase todavía no resuelta conserva targets nulos.
    df = df.with_columns(
        pl.when(pl.col(target_col).is_null()).then(None)
        .when(pl.col(target_col).is_in(["BAJA+1", "BAJA+2"])).then(1)
        .otherwise(0).cast(pl.Int8).alias("target_binario"),

        pl.when(pl.col(target_col).is_null()).then(None)
        .when(pl.col(target_col) == "BAJA+2").then(1)
        .otherwise(0).cast(pl.Int8).alias("target_baja2"),

        pl.when(pl.col(time_col).is_in(train_months)).then(pl.lit("train"))
        .when(pl.col(time_col) == valid_month).then(pl.lit("valid"))
        .when(pl.col(time_col) == test_month).then(pl.lit("test"))
        .when(pl.col(time_col).is_in(config["excluded_months"]))
        .then(pl.lit("dashboard_only"))
        .otherwise(pl.lit("lag_only"))
        .alias("partition_role")
    ).drop(["_mes_idx", "_en_t1", "_en_t2"])

    # Ninguna fila usada para modelar puede quedar sin clase.
    sin_clase = df.filter(
        pl.col(time_col).is_in(train_months + [valid_month, test_month])
        & pl.col(target_col).is_null()
    )
    if sin_clase.height:
        raise ValueError(f"{sin_clase.height} filas de modelado quedaron sin clase")

    # Resumen de control de clases por mes y partición
    resumen = (
        df.group_by([time_col, "partition_role", target_col])
        .len()
        .sort([time_col, target_col])
    )
    resumen_path = part_dir / "resumen_particiones.csv"
    resumen.write_csv(resumen_path)

    panel_path = part_dir / "panel_2019.parquet"
    df.write_parquet(panel_path, compression="zstd")

    done_file.write_text(json.dumps({
        "stage": "02_particionar",
        "timestamp": datetime.utcnow().isoformat(),
        "train_months": train_months,
        "valid_month": valid_month,
        "test_month": test_month,
        "panel_path": str(panel_path)
    }, indent=2), encoding="utf-8")

    logging.info("Partición completada. Panel guardado en %s", panel_path)


if __name__ == "__main__":
    main()

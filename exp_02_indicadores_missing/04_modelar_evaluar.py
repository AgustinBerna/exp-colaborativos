"""04_modelar_evaluar.py - Optuna solo en E0, semillero común y test 201910.

Los 50 trials usan marzo-junio para entrenar y AUC de agosto para elegir
hiperparámetros. El modelo ganador E0 elige N por ganancia en agosto. Todos
los métodos usan esos mismos hiperparámetros y N con 30 semillas comunes.
Octubre no elige N ni se usa para reoptimizar.

Salidas y Checkpoints:
  - Estudio Optuna de E0: E0/optimization/study.sqlite3 y E0/optimization/seleccion.json
  - Checkpoints por semilla en Test: <method>/test/seed_<id>/DONE.json
  - Tablas finales para R: resultados/metricas_por_semilla_test.csv,
    resultados/resumen_ensamble_test.csv, resultados/curvas_ensamble_test.csv
"""
import argparse
import gc
import json
import logging
from pathlib import Path
import numpy as np
import polars as pl
import lightgbm as lgb
import optuna
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import wilcoxon
from src.remote import snapshot_sqlite, upload, upload_files_then_marker

META_COLS = {"numero_de_cliente", "foto_mes", "clase_ternaria", "target_binario", "target_baja2", "partition_role"}


def smooth_gain_curve(cumulative_gain: np.ndarray, radius: int) -> np.ndarray:
    """Promedia la ganancia acumulada en N±radio para todos los N observables."""
    if radius < 0 or len(cumulative_gain) == 0:
        raise ValueError("Curva vacía o radio de suavizado negativo")
    positions = np.arange(len(cumulative_gain))
    left = np.maximum(0, positions - radius)
    right = np.minimum(len(cumulative_gain) - 1, positions + radius)
    prefix = np.r_[0.0, np.cumsum(cumulative_gain, dtype=np.float64)]
    smooth = (prefix[right + 1] - prefix[left]) / (right - left + 1)
    return smooth


def select_n_from_plateau(cumulative_gain: np.ndarray, radius: int) -> tuple[int, float]:
    """Maximiza la curva suavizada; en empate np.argmax elige el menor N."""
    smooth = smooth_gain_curve(cumulative_gain, radius)
    selected = int(np.argmax(smooth))
    # No elegir una campaña con ganancia real negativa aunque su entorno
    # suavizado contenga mejores puntos cercanos.
    if cumulative_gain[selected] <= 0:
        selected = 0
    return selected, float(smooth[selected])


def load_xy(parquet_path: Path):
    df = pl.read_parquet(parquet_path)
    fcols = [c for c in df.columns if c not in META_COLS]
    X = df.select(fcols).to_numpy().astype(np.float32)
    y_bin = df["target_binario"].to_numpy()
    y_baja2 = df["target_baja2"].to_numpy()
    ids = df["numero_de_cliente"].to_numpy()
    del df
    gc.collect()
    return X, y_bin, y_baja2, ids, fcols


def compute_curves_and_metrics(y_bin: np.ndarray, y_baja2: np.ndarray, ids: np.ndarray, scores: np.ndarray, tp_val: float, fp_val: float, cut_n: int):
    """Evalúa octubre al N elegido en E0/201908; no optimiza N sobre test."""
    if not 0 <= cut_n <= len(scores):
        raise ValueError(f"N congelado fuera de la población de test: {cut_n}")
    roc_auc = float(roc_auc_score(y_bin, scores))
    pr_auc = float(average_precision_score(y_bin, scores))
    auc_baja2 = float(roc_auc_score(y_baja2, scores))
    ap_baja2 = float(average_precision_score(y_baja2, scores))

    # Un score empatado se resuelve siempre por menor ID de cliente.
    order = np.lexsort((ids, -scores))
    y_b2_sorted = y_baja2[order]

    # Ganancia económica sobre BAJA+2
    gains = np.where(y_b2_sorted == 1, tp_val, fp_val)
    cum_gain = np.cumsum(gains)
    gain_at_cut = float(cum_gain[cut_n - 1]) if cut_n else 0.0

    # Lift y recall se refieren a BAJA+2, la clase que remunera la campaña.
    base_rate = float(np.mean(y_baja2))
    captured = int(np.sum(y_b2_sorted[:cut_n]))
    precision_at_cut = captured / cut_n if cut_n else 0.0
    recall_at_cut = captured / int(np.sum(y_baja2)) if np.sum(y_baja2) else 0.0
    lift_at_cut = precision_at_cut / base_rate if base_rate > 0 else 0.0

    # Puntos para el dashboard: hasta el mismo N, incluyendo siempre N exacto.
    grid = np.unique(np.r_[np.arange(250, cut_n + 1, 250), cut_n]) if cut_n else np.array([0], dtype=int)
    cum_pos = np.cumsum(y_b2_sorted)
    total_pos = max(1, int(np.sum(y_baja2)))

    curve_rows = []
    for k in grid:
        if k == 0:
            curve_rows.append({"envios": 0, "ganancia": 0.0, "precision": 0.0,
                               "recall": 0.0, "lift": 0.0})
            continue
        idx = k - 1
        prec_k = float(cum_pos[idx] / k)
        rec_k = float(cum_pos[idx] / total_pos)
        lift_k = prec_k / base_rate if base_rate > 0 else 0.0
        curve_rows.append({
            "envios": int(k),
            "ganancia": float(cum_gain[idx]),
            "precision": prec_k,
            "recall": rec_k,
            "lift": lift_k
        })

    metrics = {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "auc_baja2": auc_baja2,
        "ap_baja2": ap_baja2,
        "n_fijo_evaluado": int(cut_n),
        "gain_at_n_fijo": gain_at_cut,
        "lift_at_n_fijo": lift_at_cut,
        "recall_baja2_at_n_fijo": recall_at_cut,
        "precision_baja2_at_n_fijo": precision_at_cut,
        "baja2_capturadas_at_n_fijo": captured,
    }
    return metrics, curve_rows


def optimize_lgbm(X_tr, y_tr, X_va, y_va, yb2_va, ids_va, opt_dir: Path, config: dict, method: str, n_jobs: int) -> dict:
    """Ejecuta Optuna en 201903-201906 validando AUC en 201908 y guarda seleccion.json."""
    opt_dir.mkdir(parents=True, exist_ok=True)
    opt_done = opt_dir / "DONE.json"
    sel_file = opt_dir / "seleccion.json"
    curve_file = opt_dir / "curva_ganancia_valid_e0.csv"

    if opt_done.exists() and sel_file.exists() and curve_file.exists():
        logging.info("[%s] Optimización ya completada en %s", method, sel_file)
        return json.loads(sel_file.read_text(encoding="utf-8"))
    if opt_done.exists():
        raise RuntimeError("Checkpoint E0 incompleto: falta selección o curva de validación")

    opt_cfg = config["optimization"]
    sp = opt_cfg["search_space"]

    storage_url = f"sqlite:///{opt_dir / 'study.sqlite3'}"
    study = optuna.create_study(study_name=f"lgbm_{method}", storage=storage_url, direction="maximize", load_if_exists=True)
    completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    remaining = max(0, opt_cfg["trials"] - completed_trials)
    logging.info("[%s] Optuna BO: %d trials listos, faltan %d", method, completed_trials, remaining)

    ds_params = {"feature_pre_filter": False}
    dtrain = lgb.Dataset(X_tr, label=y_tr, params=ds_params, free_raw_data=False)
    dvalid = lgb.Dataset(X_va, label=y_va, reference=dtrain, params=ds_params, free_raw_data=False)
    dtrain.construct()

    def objective(trial: optuna.Trial):
        params = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "first_metric_only": True,
            "boost_from_average": True,
            "feature_pre_filter": False,
            "verbosity": -1,
            "n_jobs": n_jobs,
            "seed": config["seeds"][0],
            "bagging_freq": int(opt_cfg["bagging_freq"]),  # Fijo >= 1 según consigna
            "num_leaves": trial.suggest_int("num_leaves", sp["num_leaves"][0], sp["num_leaves"][1]),
            "learning_rate": trial.suggest_float("learning_rate", sp["learning_rate"][0], sp["learning_rate"][1], log=True),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", sp["min_data_in_leaf"][0], sp["min_data_in_leaf"][1]),
            "feature_fraction": trial.suggest_float("feature_fraction", sp["feature_fraction"][0], sp["feature_fraction"][1]),
            "bagging_fraction": trial.suggest_float("bagging_fraction", sp["bagging_fraction"][0], sp["bagging_fraction"][1]),
        }
        gbm = lgb.train(
            params,
            dtrain,
            num_boost_round=opt_cfg["num_boost_round"],
            valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(opt_cfg["early_stopping_rounds"], verbose=False)]
        )
        preds_va = gbm.predict(X_va, num_iteration=gbm.best_iteration)
        auc_val = float(roc_auc_score(y_va, preds_va))
        trial.set_user_attr("best_iteration", int(gbm.best_iteration))
        return auc_val

    if remaining > 0:
        # Optuna commits the trial before invoking this callback. A consistent
        # SQLite snapshot on GCS lets a new Spot VM continue at the next trial.
        study.optimize(objective, n_trials=remaining,
                       callbacks=[lambda _study, _trial: snapshot_sqlite(opt_dir / "study.sqlite3")])

    # Elegir N solo con E0 en validación, antes de mirar octubre.
    if method != "E0":
        raise ValueError("Optuna y la selección de N pertenecen exclusivamente a E0")
    best_params = study.best_params
    best_iter = int(study.best_trial.user_attrs["best_iteration"])
    final_params = {
        "objective": "binary", "metric": "auc", "boosting_type": "gbdt",
        "verbosity": -1, "n_jobs": n_jobs, "seed": config["seeds"][0],
        "bagging_freq": int(opt_cfg["bagging_freq"]), **best_params
    }
    gbm_best = lgb.train(final_params, dtrain, num_boost_round=best_iter)
    preds_va_best = gbm_best.predict(X_va)
    order_va = np.lexsort((ids_va, -preds_va_best))
    cum_gain_va = np.r_[0, np.cumsum(np.where(yb2_va[order_va] == 1,
                                config["gain"]["tp_value"], config["gain"]["fp_value"]))]
    # La meseta se evalúa en agosto con E0: promedio de la ganancia acumulada
    # entre N-500 y N+500 envíos (radio configurable). Empates: menor N.
    radius = int(config["gain"]["smoothing_radius_n"])
    n_selected_valid, smooth_gain = select_n_from_plateau(cum_gain_va, radius)
    # Guardar los 131k+ puntos de agosto, no sólo el N ganador. El dashboard
    # puede mostrar la meseta y la caída posterior sin consultar octubre.
    smooth_all = smooth_gain_curve(cum_gain_va, radius)
    pl.DataFrame({"envios": np.arange(len(cum_gain_va), dtype=np.int32),
                  "ganancia": cum_gain_va, "ganancia_suavizada": smooth_all}).write_csv(curve_file)

    best_info = {
        "source_method": method,
        "best_auc_valid": float(study.best_value),
        "best_iteration": best_iter,
        "n_selected_valid": n_selected_valid,
        "gain_valid_at_n": float(cum_gain_va[n_selected_valid]),
        "gain_valid_smoothed_at_n": smooth_gain,
        "smoothing_radius_n": radius,
        "n_raw_max_valid": int(np.argmax(cum_gain_va)),
        "best_params": best_params
    }
    sel_file.write_text(json.dumps(best_info, indent=2), encoding="utf-8")
    opt_done.write_text(json.dumps({"status": "done"}), encoding="utf-8")
    upload_files_then_marker([sel_file, curve_file], opt_done)

    del dtrain, dvalid
    gc.collect()
    return best_info


def run_method(work: Path, config: dict, method: str, baseline_selection: dict | None, n_jobs: int):
    prep_done = work / method / "prepared" / "DONE.json"
    if not prep_done.exists():
        raise RuntimeError(f"Falta preparar {method} ({prep_done})")

    logging.info("=== Procesando método %s con hiperparámetros de E0 ===", method)
    X_tr, y_tr, _, _, _ = load_xy(work / method / "prepared" / "train.parquet")

    opt_dir = work / method / "optimization"
    opt_cfg = config["optimization"]

    # Solo E0 ajusta Optuna y elige N con agosto; los demás heredan ambos.
    if method == "E0":
        X_va, y_va, yb2_va, ids_va, _ = load_xy(work / method / "prepared" / "valid.parquet")
        best_info = optimize_lgbm(X_tr, y_tr, X_va, y_va, yb2_va, ids_va, opt_dir, config, method, n_jobs)
        del X_va, y_va, yb2_va, ids_va
        gc.collect()
    else:
        if baseline_selection is None or baseline_selection.get("source_method") != "E0":
            raise RuntimeError("E0 debe optimizarse primero y fijar N en validación")
        opt_dir.mkdir(parents=True, exist_ok=True)
        best_info = baseline_selection
        (opt_dir / "seleccion.json").write_text(json.dumps(best_info, indent=2), encoding="utf-8")
        (opt_dir / "DONE.json").write_text(json.dumps({"status": "inherited_from_E0"}), encoding="utf-8")
        logging.info("[%s] Usando hiperparámetros congelados de E0 (iter=%d, N=%d)",
                     method, best_info["best_iteration"], best_info["n_selected_valid"])

    # Write the same E0 decision before any October scoring. A restart checks
    # that neither best hyperparameters nor N changed mid-run.
    frozen = work / "SELECCION_FROZEN.json"
    if frozen.exists():
        if json.loads(frozen.read_text(encoding="utf-8")) != best_info:
            raise RuntimeError("La selección E0 cambió después de abrir octubre")
    else:
        temporary = frozen.with_suffix(".partial")
        temporary.write_text(json.dumps(best_info, indent=2), encoding="utf-8")
        temporary.replace(frozen)
    upload(frozen)

    # October is read only after E0's hyperparameters and N are frozen.
    X_te, y_te, yb2_te, ids_te, _ = load_xy(work / method / "prepared" / "test.parquet")

    # Entrenamiento del Semillero y Evaluación en Test (201910)
    seeds = config["seeds"]
    tp_val = config["gain"]["tp_value"]
    fp_val = config["gain"]["fp_value"]
    cut_n = int(best_info["n_selected_valid"])

    dtrain = lgb.Dataset(X_tr, label=y_tr, params={"feature_pre_filter": False}, free_raw_data=False)
    dtrain.construct()

    seed_metrics_rows = []
    seed_curves_rows = []
    ensemble_scores_test = np.zeros(len(y_te), dtype=np.float64)

    for seed in seeds:
        seed_dir = work / method / "test" / f"seed_{seed}"
        seed_done = seed_dir / "DONE.json"
        pred_file = seed_dir / "pred_test.npy"

        if seed_done.exists() and pred_file.exists():
            scores_te = np.load(pred_file)
            m_te = json.loads((seed_dir / "metrics.json").read_text(encoding="utf-8"))
            c_te = json.loads((seed_dir / "curves.json").read_text(encoding="utf-8"))
        else:
            seed_dir.mkdir(parents=True, exist_ok=True)
            params = {
                "objective": "binary",
                "metric": "auc",
                "boosting_type": "gbdt",
                "feature_pre_filter": False,
                "verbosity": -1,
                "n_jobs": n_jobs,
                "seed": int(seed),
                "feature_fraction_seed": int(seed) + 1,
                "bagging_seed": int(seed) + 2,
                "bagging_freq": int(opt_cfg["bagging_freq"]),
                **best_info["best_params"]
            }
            gbm = lgb.train(params, dtrain, num_boost_round=best_info["best_iteration"])
            scores_te = gbm.predict(X_te)
            np.save(pred_file, scores_te)

            m_te, c_te = compute_curves_and_metrics(y_te, yb2_te, ids_te, scores_te, tp_val, fp_val, cut_n)
            (seed_dir / "metrics.json").write_text(json.dumps(m_te, indent=2), encoding="utf-8")
            (seed_dir / "curves.json").write_text(json.dumps(c_te, indent=2), encoding="utf-8")
            seed_done.write_text(json.dumps({"seed": seed, "status": "ok"}), encoding="utf-8")
            logging.info("[%s] Semilla %d evaluada en Test | AUC=%.5f | Ganancia@N=%0.f | N=%d",
                         method, seed, m_te["roc_auc"], m_te["gain_at_n_fijo"], cut_n)

        # Publish the three finished files before the seed marker. Repeating
        # this on resume also repairs an interrupted upload of a local DONE.
        upload_files_then_marker([pred_file, seed_dir / "metrics.json", seed_dir / "curves.json"], seed_done)

        ensemble_scores_test += scores_te / len(seeds)
        seed_metrics_rows.append({"method": method, "seed": seed, **m_te})
        for row in c_te:
            seed_curves_rows.append({"method": method, "seed": seed, **row})

    ens_metrics, ens_curves = compute_curves_and_metrics(y_te, yb2_te, ids_te, ensemble_scores_test, tp_val, fp_val, cut_n)

    del X_tr, X_te, dtrain
    gc.collect()

    return best_info, seed_metrics_rows, seed_curves_rows, {"method": method, **ens_metrics}, [
        {"method": method, **r} for r in ens_curves
    ]


def main():
    parser = argparse.ArgumentParser(description="Etapa 04: Optimización en E0, Semillero y Test en 201910")
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--bo_mode", choices=["baseline"], default="baseline",
                        help="Optuna solo en E0; todos los demás métodos usan sus hiperparámetros y N")
    parser.add_argument("--n_jobs", type=int, default=7, help="Hilos de CPU para LightGBM")
    args = parser.parse_args()

    work = args.work.resolve()
    log_path = work / "logs" / "04_modelar_evaluar.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(log_path), logging.StreamHandler()], format="%(asctime)s %(message)s")

    config = json.loads(args.config.read_text(encoding="utf-8"))
    methods = args.methods if args.methods else config["methods"]
    if not methods or methods[0] != "E0" or len(methods) != len(set(methods)):
        raise ValueError("La comparación debe empezar con E0 y no repetir métodos")
    if len(config["seeds"]) != 30 or len(set(config["seeds"])) != 30:
        raise ValueError("El diseño exige 30 semillas distintas y comunes")
    # No abrir test si algún tratamiento todavía carece de features preparadas.
    for method in methods:
        if not (work / method / "prepared" / "DONE.json").exists():
            raise RuntimeError(f"Falta preparar {method} antes de evaluar octubre")

    # La selección de E0 contiene los hiperparámetros y N elegidos en agosto.
    baseline_sel = None
    e0_sel_path = work / "E0" / "optimization" / "seleccion.json"
    if e0_sel_path.exists() and (work / "E0" / "optimization" / "DONE.json").exists():
        baseline_sel = json.loads(e0_sel_path.read_text(encoding="utf-8"))

    all_seed_metrics, all_seed_curves, all_ens_metrics, all_ens_curves = [], [], [], []
    for m in methods:
        best_info, sm, sc, em, ec = run_method(work, config, m, baseline_sel, args.n_jobs)
        if m == "E0" and baseline_sel is None:
            baseline_sel = best_info
        all_seed_metrics.extend(sm)
        all_seed_curves.extend(sc)
        all_ens_metrics.append(em)
        all_ens_curves.extend(ec)

    res_dir = work / "resultados"
    res_dir.mkdir(parents=True, exist_ok=True)

    df_seed_m = pl.DataFrame(all_seed_metrics)
    df_seed_c = pl.DataFrame(all_seed_curves)
    df_ens_m = pl.DataFrame(all_ens_metrics)
    df_ens_c = pl.DataFrame(all_ens_curves)

    # Test pareado al mismo N. Holm ajusta cada familia de contrastes entre variantes.
    if "E0" in methods and len(methods) > 1:
        e0_auc = df_seed_m.filter(pl.col("method") == "E0").sort("seed")["roc_auc"].to_numpy()
        e0_gain = df_seed_m.filter(pl.col("method") == "E0").sort("seed")["gain_at_n_fijo"].to_numpy()
        wilcox_rows = []
        for m in methods:
            if m == "E0":
                wilcox_rows.append({"method": m, "p_val_auc_vs_E0": 1.0, "p_val_gain_vs_E0": 1.0})
                continue
            m_auc = df_seed_m.filter(pl.col("method") == m).sort("seed")["roc_auc"].to_numpy()
            m_gain = df_seed_m.filter(pl.col("method") == m).sort("seed")["gain_at_n_fijo"].to_numpy()
            _, p_auc = wilcoxon(m_auc, e0_auc, zero_method="zsplit")
            _, p_gain = wilcoxon(m_gain, e0_gain, zero_method="zsplit")
            wilcox_rows.append({"method": m, "p_val_auc_vs_E0": float(p_auc), "p_val_gain_vs_E0": float(p_gain)})
        candidates = [row for row in wilcox_rows if row["method"] != "E0"]
        for raw_name, adjusted_name in (
            ("p_val_auc_vs_E0", "p_holm_auc_vs_E0"),
            ("p_val_gain_vs_E0", "p_holm_gain_vs_E0"),
        ):
            ranked = sorted(candidates, key=lambda row: row[raw_name])
            previous = 0.0
            for index, row in enumerate(ranked):
                previous = min(1.0, max(previous, row[raw_name] * (len(ranked) - index)))
                row[adjusted_name] = previous
            wilcox_rows[0][adjusted_name] = 1.0
        df_ens_m = df_ens_m.join(pl.DataFrame(wilcox_rows), on="method", how="left")

    df_seed_m.write_csv(res_dir / "metricas_por_semilla_test.csv")
    df_seed_c.write_csv(res_dir / "curvas_por_semilla_test.csv")
    df_ens_m.write_csv(res_dir / "resumen_ensamble_test.csv")
    df_ens_c.write_csv(res_dir / "curvas_ensamble_test.csv")

    final_marker = res_dir / "FINALIZADO.json"
    final_marker.write_text(json.dumps({"methods": methods, "bo_mode": args.bo_mode, "status": "ok"}, indent=2), encoding="utf-8")
    upload_files_then_marker([res_dir / name for name in (
        "metricas_por_semilla_test.csv", "curvas_por_semilla_test.csv",
        "resumen_ensamble_test.csv", "curvas_ensamble_test.csv")], final_marker)
    logging.info("Modelado y evaluación finalizados. Resultados listos en %s", res_dir)


if __name__ == "__main__":
    main()

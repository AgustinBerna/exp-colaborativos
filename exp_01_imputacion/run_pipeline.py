"""Run the complete experiment locally or on a Spot VM.

Only paths differ between environments. A completed stage can be repeated;
its DONE markers are reused. On GCP, --remote restores/publishes checkpoints.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

from src.remote import restore, sync_stage

ROOT = Path(__file__).resolve().parent
STEPS = ("auditar", "particionar", "preparar", "modelar", "dashboard")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(config_path: Path, data_path: Path) -> dict:
    sources = sorted([*ROOT.glob("*.py"), *ROOT.glob("*.R"), *ROOT.glob("src/*.py"),
                      *ROOT.glob("requirements*.txt")])
    return {
        "config_sha256": sha256(config_path),
        "data_sha256": sha256(data_path),
        "code_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in sources},
        "python": sys.version.split()[0],
    }


def validate_config(config: dict):
    if config["train_months"] != [201903, 201904, 201905, 201906]:
        raise ValueError("Meses de entrenamiento fuera del protocolo")
    if (config["valid_month"], config["test_month"]) != (201908, 201910):
        raise ValueError("Validación/test fuera del protocolo")
    if not {201911, 201912} <= set(config["all_months"]):
        raise ValueError("Faltan noviembre/diciembre para etiquetar octubre")
    methods = config["methods"]
    if methods != ["E0", "E1", "E2", "E3", "E4", "E5", "E06"]:
        raise ValueError("imputacion_v3 exige E0-E5 y E06 en ese orden")
    if config.get("drop_features_without_history") != ["tmobile_app", "cmobile_app_trx"]:
        raise ValueError("La exclusión de variables debe estar fijada en config.json")
    if config["miceforest"]["data_subset"] != 0:
        raise ValueError("E06 debe usar todos los donantes observados (data_subset=0)")
    if len(config["seeds"]) != 30 or len(set(config["seeds"])) != 30:
        raise ValueError("El semillero debe contener 30 semillas comunes distintas")
    if config["optimization"]["trials"] != 50:
        raise ValueError("La búsqueda de E0 debe contener 50 trials")
    if int(config["optimization"]["bagging_freq"]) < 1:
        raise ValueError("bagging_freq debe ser al menos 1")
    if int(config["gain"]["smoothing_radius_n"]) < 0:
        raise ValueError("smoothing_radius_n no puede ser negativo")


@contextmanager
def single_writer(work: Path):
    """One local pipeline writer at a time; fail quickly if already running."""
    path = work / ".pipeline.lock"
    handle = path.open("a+b")
    acquired = False
    try:
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        acquired = True
        yield
    finally:
        if acquired and os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        elif acquired:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()


def command(step: str, work: Path, config_path: Path, data_path: Path, config: dict):
    common = ["--work", str(work), "--config", str(config_path)]
    if step == "auditar":
        return [[sys.executable, str(ROOT / "01_auditar.py"), *common, "--data", str(data_path)]]
    if step == "particionar":
        return [[sys.executable, str(ROOT / "02_particionar.py"), *common]]
    if step == "preparar":
        commands = []
        for method in config["methods"]:
            if method == "E0":
                script, extra = "03_imputar_baseline.py", []
            elif method in ("E1", "E2", "E3"):
                script, extra = "03_imputar_univariada.py", ["--method", method]
            elif method == "E4":
                script, extra = "03_imputar_temporal.py", []
            elif method == "E5":
                p = config["ridge"]
                script, extra = "03_imputar_multivariada.py", [
                    "--fit_sample_size", str(p["fit_sample_size"]),
                    "--max_iter", str(p["max_iter"]),
                    "--n_nearest_features", str(p["n_nearest_features"])]
            elif method == "E06":
                p = config["miceforest"]
                script, extra = "03_imputar_miceforest.py", [
                    "--method", "E06", "--fit_sample_size", str(p["fit_sample_size"]),
                    "--iterations", str(p["iterations"]),
                    "--lgbm_num_iterations", str(p["lgbm_num_iterations"])]
            else:
                raise ValueError(f"Método sin script de preparación: {method}")
            commands.append([sys.executable, str(ROOT / script), *common, *extra])
        return commands
    if step == "modelar":
        return [[sys.executable, str(ROOT / "04_modelar_evaluar.py"), *common,
                 "--methods", *config["methods"]]]
    if step == "dashboard":
        if not shutil.which("Rscript"):
            raise RuntimeError("Rscript no está instalado; consultar README.md")
        return [["Rscript", str(ROOT / "dashboard.R"), str(work)]]
    raise ValueError(step)


def main() -> int:
    parser = argparse.ArgumentParser(description="Imputaciones 2019: local/GCP con las mismas etapas")
    parser.add_argument("--data", type=Path, required=True, help="CSV o CSV.GZ crudo")
    parser.add_argument("--work", type=Path, required=True, help="Salida activa en disco local/ext4")
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--step", choices=("all", *STEPS, "estado"), default="all")
    parser.add_argument("--remote", help="gs://bucket/exp/prefijo exclusivo; omitir en local")
    args = parser.parse_args()

    work, data_path, config_path = args.work.resolve(), args.data.resolve(), args.config.resolve()
    if not data_path.is_file() or not config_path.is_file():
        parser.error("No existe --data o --config")
    work.mkdir(parents=True, exist_ok=True)
    (work / "logs").mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(work / "logs" / "main.log", encoding="utf-8"),
                                  logging.StreamHandler(sys.stdout)], force=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    expected = identity(config_path, data_path)
    with single_writer(work):
        manifest = work / "run_identity.json"
        if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8")) != expected:
            raise RuntimeError("Cambió código, configuración o CSV: usar otro --work y --remote")
        # A fresh Spot VM restores from Cloud Storage. An existing local run
        # keeps its newer files and republishes any interrupted uploads.
        if args.remote and not manifest.exists():
            restore(work, args.remote)
        if manifest.exists():
            if json.loads(manifest.read_text(encoding="utf-8")) != expected:
                raise RuntimeError("El prefijo remoto pertenece a otra identidad de corrida")
        else:
            manifest.write_text(json.dumps(expected, indent=2), encoding="utf-8")
        if args.remote:
            os.environ["PIPELINE_REMOTE_URI"] = args.remote.rstrip("/")
            os.environ["PIPELINE_WORK_DIR"] = str(work)
            sync_stage(work)  # upload identity before any stage

        if args.step == "estado":
            subprocess.run([sys.executable, str(ROOT / "05_estado.py"), "--work", str(work),
                            "--config", str(config_path)], check=True)
            return 0
        stages = STEPS if args.step == "all" else (args.step,)
        for stage in stages:
            for cmd in command(stage, work, config_path, data_path, config):
                logging.info("INICIO %s: %s", stage, Path(cmd[1]).name)
                subprocess.run(cmd, cwd=ROOT, check=True)
                sync_stage(work)
                logging.info("FIN %s: %s", stage, Path(cmd[1]).name)
        logging.info("Corrida completada. Resultados: %s", work / "resultados")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

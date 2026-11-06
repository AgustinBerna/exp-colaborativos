"""05_estado.py - Monitorea el avance de todos los checkpoints sin bloquear la ejecución.

Muestra en consola y guarda en logs/05_estado.log el estado de:
  - 01_auditar (raw/DONE.json)
  - 02_particionar (partition/DONE.json)
  - Cada experimento (E0..E7): preparación, selección heredada de E0 y semillas de test.
"""
import argparse, json, logging
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Estado general del pipeline")
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    args = parser.parse_args()

    work = args.work.resolve()
    log_path = work / "logs" / "05_estado.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(log_path), logging.StreamHandler()], format="%(asctime)s %(message)s")

    config = json.loads(args.config.read_text(encoding="utf-8"))
    seeds = config["seeds"]
    methods = config["methods"]

    logging.info("Etapa 01 (Auditar):     %s", "OK" if (work / "raw" / "DONE.json").exists() else "PENDIENTE")
    logging.info("Etapa 02 (Particionar): %s", "OK" if (work / "partition" / "DONE.json").exists() else "PENDIENTE")
    logging.info("-" * 65)

    for m in methods:
        prep_ok = (work / m / "prepared" / "DONE.json").exists()
        opt_ok = (work / m / "optimization" / "DONE.json").exists()
        tested_seeds = sum((work / m / "test" / f"seed_{s}" / "DONE.json").exists() for s in seeds)
        logging.info(
            "Experimento %-4s | Imputado+Lags: %-5s | Selección E0: %-5s | Test Semillas: %d/%d",
            m, str(prep_ok), str(opt_ok), tested_seeds, len(seeds)
        )

    logging.info("-" * 65)
    logging.info("Tablas para R listas:   %s", "OK" if (work / "resultados" / "FINALIZADO.json").exists() else "PENDIENTE")
    logging.info("Dashboard HTML en R:    %s", "OK" if (work / "resultados" / "dashboard_comparativo.html").exists() else "PENDIENTE")

if __name__ == "__main__":
    main()

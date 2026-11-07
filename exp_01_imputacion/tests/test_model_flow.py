"""Small end-to-end model/report contract with invented customers only."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import polars as pl


ROOT = Path(__file__).resolve().parents[1]


class ModelFlowTests(unittest.TestCase):
    def test_baseline_selection_is_shared_and_resume_skips_completed_fits(self):
        rng = np.random.default_rng(17)
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary) / "run"
            (work / "logs").mkdir(parents=True)
            config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
            config["methods"] = ["E0", "E5"]
            config["gain"]["smoothing_radius_n"] = 5
            config["optimization"].update({"trials": 2, "num_boost_round": 25,
                                           "early_stopping_rounds": 5})
            config["optimization"]["search_space"] = {
                "num_leaves": [4, 8], "learning_rate": [0.05, 0.1],
                "min_data_in_leaf": [5, 10], "feature_fraction": [0.8, 1.0],
                "bagging_fraction": [0.8, 1.0],
            }
            config_path = work / "smoke_config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")

            for method in config["methods"]:
                prepared = work / method / "prepared"
                prepared.mkdir(parents=True)
                for role, month, size in (("train", 201906, 160),
                                          ("valid", 201908, 80),
                                          ("test", 201910, 80)):
                    y_baja2 = (np.arange(size) % 10 == 0).astype(np.int8)
                    y_bin = (np.arange(size) % 4 == 0).astype(np.int8)
                    y_bin = np.maximum(y_bin, y_baja2)
                    feature = y_bin.astype(float) + rng.normal(0, 0.5, size)
                    feature[::13] = np.nan if method == "E0" else 0.0
                    frame = pl.DataFrame({
                        "numero_de_cliente": np.arange(size) + 100000,
                        "foto_mes": [month] * size,
                        "clase_ternaria": ["BAJA+2" if b else "BAJA+1" if y else "CONTINUA"
                                            for b, y in zip(y_baja2, y_bin)],
                        "target_binario": y_bin,
                        "target_baja2": y_baja2,
                        "partition_role": [role] * size,
                        "x": feature,
                    })
                    frame.write_parquet(prepared / f"{role}.parquet")
                (prepared / "DONE.json").write_text("{}", encoding="utf-8")

            command = [sys.executable, str(ROOT / "04_modelar_evaluar.py"),
                       "--work", str(work), "--config", str(config_path),
                       "--n_jobs", "2"]
            for _ in range(2):
                subprocess.run(command, cwd=ROOT, check=True,
                               capture_output=True, text=True)

            selection = json.loads((work / "E0" / "optimization" / "seleccion.json").read_text())
            inherited = json.loads((work / "E5" / "optimization" / "seleccion.json").read_text())
            self.assertEqual(selection, inherited)
            self.assertEqual(selection["smoothing_radius_n"], 5)
            curve = pl.read_csv(work / "E0" / "optimization" / "curva_ganancia_valid_e0.csv")
            self.assertEqual(curve.height, 81)  # N=0 y los 80 clientes de agosto.
            self.assertEqual(curve["envios"].min(), 0)
            self.assertEqual(curve["envios"].max(), 80)
            self.assertAlmostEqual(curve["ganancia_suavizada"][selection["n_selected_valid"]],
                                   selection["gain_valid_smoothed_at_n"])
            metrics = pl.read_csv(work / "resultados" / "metricas_por_semilla_test.csv")
            self.assertEqual(metrics.height, 60)
            self.assertEqual(metrics["n_fijo_evaluado"].unique().to_list(),
                             [selection["n_selected_valid"]])
            self.assertTrue((work / "resultados" / "FINALIZADO.json").exists())
            self.assertTrue((work / "E5" / "test" /
                             f"seed_{config['seeds'][-1]}" / "DONE.json").exists())


if __name__ == "__main__":
    unittest.main()

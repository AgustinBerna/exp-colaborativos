"""Prueba sintética de E0-E06: mismas filas y orden físico en cada partición."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import polars as pl


ROOT = Path(__file__).resolve().parents[1]


class ImputadoresTests(unittest.TestCase):
    def test_all_methods_share_rows_and_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            (work / "partition").mkdir()
            rows = []
            for customer in range(10):
                for month in range(201901, 201913):
                    rows.append({
                        "numero_de_cliente": customer, "foto_mes": month,
                        "x": None if month == 201904 and customer % 3 == 0
                             else float(customer + month % 100),
                        "y": float(customer * 2 + month % 100),
                        "z": None if month == 201908 and customer % 3 == 0
                             else float(customer * 3 + month % 100),
                        "clase_ternaria": "CONTINUA", "target_binario": 0,
                        "target_baja2": 0,
                        "partition_role": "train" if month in (201903, 201904, 201905, 201906)
                                          else "valid" if month == 201908
                                          else "test" if month == 201910 else "lag_only",
                    })
            pl.DataFrame(rows).write_parquet(work / "partition/panel_2019.parquet")
            common = ["--work", str(work), "--config", str(ROOT / "config.json")]
            commands = [
                ("03_imputar_baseline.py", []),
                ("03_imputar_univariada.py", ["--method", "E1"]),
                ("03_imputar_univariada.py", ["--method", "E2"]),
                ("03_imputar_univariada.py", ["--method", "E3"]),
                ("03_imputar_temporal.py", []),
                ("03_imputar_multivariada.py", ["--fit_sample_size", "30",
                                                  "--max_iter", "2", "--n_nearest_features", "2"]),
                ("03_imputar_miceforest.py", ["--method", "E06", "--fit_sample_size", "30",
                                               "--iterations", "1", "--lgbm_num_iterations", "5"]),
            ]
            for script, extra in commands:
                result = subprocess.run([sys.executable, str(ROOT / script), *common, *extra],
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, f"{script}: {result.stderr}")
            for role in ("train", "valid", "test"):
                base = pl.read_parquet(work / "E0/prepared" / f"{role}.parquet")
                for method in ("E1", "E2", "E3", "E4", "E5", "E06"):
                    other = pl.read_parquet(work / method / "prepared" / f"{role}.parquet")
                    self.assertEqual(base.select(["numero_de_cliente", "foto_mes"]).rows(),
                                     other.select(["numero_de_cliente", "foto_mes"]).rows())
            e06_valid = pl.read_parquet(work / "E06/prepared/valid.parquet")
            self.assertEqual(e06_valid["z"].null_count(), 0)


if __name__ == "__main__":
    unittest.main()

"""Contrato sintético: la exclusión ocurre antes de cualquier imputación."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import polars as pl


ROOT = Path(__file__).resolve().parents[1]


class ExclusionTests(unittest.TestCase):
    def test_two_unavailable_features_are_removed_from_panel_and_lags(self):
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            raw = work / "raw"
            raw.mkdir()
            (raw / "DONE.json").write_text("{}", encoding="utf-8")
            for month in range(201901, 201913):
                pl.DataFrame({
                    "numero_de_cliente": [1, 2, 3, 4],
                    "foto_mes": [month] * 4,
                    "x": [float(i + month % 100) for i in range(4)],
                    "tmobile_app": [None] * 4 if month <= 201906 else [1.0] * 4,
                    "cmobile_app_trx": [None] * 4 if month <= 201906 else [2.0] * 4,
                }).write_parquet(raw / f"{month}.parquet")
            subprocess.run([sys.executable, str(ROOT / "02_particionar.py"),
                            "--work", str(work), "--config", str(ROOT / "config.json")],
                           check=True, capture_output=True, text=True)
            panel = pl.read_parquet(work / "partition/panel_2019.parquet")
            self.assertNotIn("tmobile_app", panel.columns)
            self.assertNotIn("cmobile_app_trx", panel.columns)
            excluded = json.loads((work / "partition/features_excluidas.json").read_text())
            self.assertEqual({row["feature"] for row in excluded},
                             {"tmobile_app", "cmobile_app_trx"})
            subprocess.run([sys.executable, str(ROOT / "03_imputar_baseline.py"),
                            "--work", str(work), "--config", str(ROOT / "config.json")],
                           check=True, capture_output=True, text=True)
            train = pl.read_parquet(work / "E0/prepared/train.parquet")
            self.assertFalse(any("tmobile_app" in c or "cmobile_app_trx" in c
                                 for c in train.columns))


if __name__ == "__main__":
    unittest.main()

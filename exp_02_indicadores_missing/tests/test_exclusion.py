"""Contrato sintético: exclusión con train e indicadores para todas las variables."""
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
                    "x": [None if month == 201904 and i == 0
                          else float(i + month % 100) for i in range(4)],
                    # Completa en train, faltante recién en octubre: debe
                    # tener indicador aunque train contenga sólo ceros en él.
                    "z": [None if month == 201910 and i == 0
                          else float(i + 1) for i in range(4)],
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
            indicator_plan = json.loads((work / "partition/indicadores_ausente.json").read_text())
            self.assertEqual({row["feature"] for row in indicator_plan}, {"x", "z"})
            self.assertEqual({row["indicator"] for row in indicator_plan},
                             {"x__ausente", "z__ausente"})
            self.assertEqual(next(row["missing_in_train"] for row in indicator_plan
                                  if row["feature"] == "z"), 0)
            april = panel.filter((pl.col("foto_mes") == 201904) &
                                 (pl.col("numero_de_cliente") == 1))
            self.assertEqual(april["x__ausente"][0], 1)
            october = panel.filter((pl.col("foto_mes") == 201910) &
                                   (pl.col("numero_de_cliente") == 1))
            self.assertEqual(october["z__ausente"][0], 1)
            subprocess.run([sys.executable, str(ROOT / "03_imputar_baseline.py"),
                            "--work", str(work), "--config", str(ROOT / "config.json")],
                           check=True, capture_output=True, text=True)
            train = pl.read_parquet(work / "E0/prepared/train.parquet")
            self.assertFalse(any("tmobile_app" in c or "cmobile_app_trx" in c
                                 for c in train.columns))
            subprocess.run([sys.executable, str(ROOT / "03_imputar_univariada.py"),
                            "--work", str(work), "--config", str(ROOT / "config.json"),
                            "--method", "E2"],
                           check=True, capture_output=True, text=True)
            median_train = pl.read_parquet(work / "E2/prepared/train.parquet")
            median_april = median_train.filter((pl.col("foto_mes") == 201904) &
                                               (pl.col("numero_de_cliente") == 1))
            self.assertEqual(median_april["x__ausente"][0], 1)
            self.assertFalse(median_april["x"].is_null()[0])
            median_test = pl.read_parquet(work / "E2/prepared/test.parquet")
            median_october = median_test.filter(pl.col("numero_de_cliente") == 1)
            self.assertEqual(median_october["z__ausente"][0], 1)
            self.assertFalse(median_october["z"].is_null()[0])
            self.assertFalse(any(c.endswith("__ausente_lag1") for c in median_train.columns))


if __name__ == "__main__":
    unittest.main()

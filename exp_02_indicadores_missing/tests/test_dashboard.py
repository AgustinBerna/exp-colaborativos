"""Smoke test del dashboard con datos inventados; requiere Rscript instalado."""
import csv
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("E0", "E1", "E2", "E3", "E4", "E5", "E06")


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class DashboardTests(unittest.TestCase):
    def test_dashboard_renders_validation_curve_and_significance(self):
        rscript = os.environ.get("RSCRIPT") or shutil.which("Rscript")
        if not rscript:
            self.skipTest("Rscript no disponible")
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            result = work / "resultados"
            result.mkdir()
            write_csv(result / "metricas_por_semilla_test.csv", [
                {"method": method, "seed": 100000 + seed, "roc_auc": .93 + j*.001 + seed*.00001,
                 "pr_auc": .4 + j*.001, "gain_at_n_fijo": 200000000 + j*1000000 + seed*10000,
                 "lift_at_n_fijo": 12 + j*.1, "recall_baja2_at_n_fijo": .5 + j*.01}
                for j, method in enumerate(METHODS) for seed in range(30)
            ])
            write_csv(result / "resumen_ensamble_test.csv", [
                {"method": method, "roc_auc": .93+j*.001,
                 "gain_at_n_fijo": 200000000+j*1000000, "n_fijo_evaluado": 50,
                 "p_holm_auc_vs_E0": p, "p_holm_gain_vs_E0": p}
                for j, (method, p) in enumerate(zip(METHODS, (1, .03, .2, .001, .04, .02, .07)))
            ])
            write_csv(result / "curvas_ensamble_test.csv", [
                {"method": method, "envios": n, "ganancia": n*1000000+j*10000,
                 "precision": .1+j*.01, "recall": n/100, "lift": 8+j*.1}
                for j, method in enumerate(METHODS) for n in (0, 10, 20, 30, 40, 50)
            ])
            opt = work / "E0" / "optimization"
            write_csv(opt / "curva_ganancia_valid_e0.csv", [
                {"envios": n, "ganancia": n*1000000 if n <= 50 else (100-n)*1000000,
                 "ganancia_suavizada": n*900000 if n <= 50 else (100-n)*900000}
                for n in range(101)
            ])
            (opt / "seleccion.json").write_text(json.dumps({"n_selected_valid": 50,
                    "n_raw_max_valid": 50, "smoothing_radius_n": 5}), encoding="utf-8")
            partition = work / "partition"
            partition.mkdir()
            (partition / "indicadores_ausente.json").write_text(
                json.dumps([{"feature": "x", "indicator": "x__ausente"}]), encoding="utf-8")
            write_csv(work / "raw" / "reporte_nan_por_mes.csv", [
                {"foto_mes": month, "pct_nan": 12.0,
                 "convertida_de_100pct_cero": 1} for month in (201903, 201908, 201910)
            ])
            environment = os.environ.copy()
            if os.name == "nt":
                environment["LC_ALL"] = "English_United States.utf8"
            run = subprocess.run([str(rscript), str(ROOT / "dashboard.R"), str(work)],
                                 capture_output=True, text=True, env=environment)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            html = result / "dashboard_comparativo.html"
            self.assertTrue(html.exists())
            contents = html.read_text(encoding="utf-8")
            self.assertIn("Cómo se eligió N", contents)
            self.assertIn("Wilcoxon", contents)
            self.assertIn("* ↑", contents)
            self.assertIn("ns ↑", contents)
            self.assertTrue((result / "lib").exists())


if __name__ == "__main__":
    unittest.main()

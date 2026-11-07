# Exp 1 · Imputación de valores faltantes

Compara las imputaciones con los NaN nativos de LightGBM, sin indicadores adicionales de ausencia.

## Resultados de esta corrida

- [Dashboard comparativo autocontenido](resultados/dashboard_comparativo.html): descargar y abrir en el navegador; funciona sin `lib/` ni conexión.
- [Selección E0 de esta corrida](seleccion.json): AUC de validación 0.937603422, **280 iteraciones** y **N=4323**.
- [Resumen numérico del dashboard](resultados/resumen_dashboard.csv): exportado de la tabla HTML, sin volver a entrenar ni recalcular resultados.
- [Manifiesto de publicación](resultados/manifiesto_publicacion.json): hashes del código, selección, HTML original, recursos embebidos y HTML publicado.

El dashboard conserva las métricas y los datos de todos los gráficos del HTML suministrado. Se integraron JavaScript, CSS e imágenes de las dependencias locales; se muestra la tabla completa de los siete métodos sin paginación.

## Protocolo

Train completo marzo–junio de 2019; validación agosto; test octubre. Target binario BAJA+1/BAJA+2; ganancia sólo para BAJA+2 (+$1.072.500 por acierto, −$27.500 por los demás contactos). Imputación de variables base antes de lags/deltas 1 y 2. Auditoría común de variables mes a mes con 100% ceros a null y exclusión de variables sin información en entrenamiento.

E0: NaN nativo; E1: media; E2: mediana; E3: moda; E4: forward fill causal por cliente; E5: IterativeImputer/BayesianRidge (60.000 filas, 10 rondas, 25 vecinas); E06: miceforest/LightGBM (5 rondas, 100 iteraciones por variable, 5 donantes). Ver configuración exacta en [config.json](config.json).

La configuración del código contempla 50 trials de Optuna sólo para E0, elegidos por AUC de agosto. Cada método hereda sus parámetros, iteración y N. La evaluación publicada contiene 30 semillas comunes por método. Los p-valores del dashboard se conservan como parte del artefacto recibido; las semillas comparten clientes y mes, por lo que no representan 30 muestras independientes de población ni 30 tests temporales.

## Reproducir una ejecución nueva

Desde esta carpeta, con Python 3.13.9 y las dependencias de `requirements.txt`, más R y los paquetes de `install_r_packages.R` para generar el dashboard:

```bash
python -m pip install -r requirements.txt
Rscript install_r_packages.R
python run_pipeline.py --data /ruta/competencia_02_crudo.csv --work /ruta/nueva/corrida_exp_01_imputacion --step all
```

La ejecución nueva guarda su propia optimización y sus checkpoints en `--work`; el `seleccion.json` de esta carpeta es el registro de la corrida publicada y no sustituye automáticamente esa optimización. Para autocontener el nuevo HTML de R: `python ../herramientas/autocontener_dashboard.py /ruta/nueva/corrida/resultados/dashboard_comparativo.html /ruta/dashboard_autocontenido.html`.

El código se copió de `imputacion_v3`. Sin el `run_identity.json` original no se certifica por hash que sea exactamente el código ejecutado al generar estos resultados. No se incluyen el dataset, modelos, predicciones individuales ni entornos. Los originales locales permanecen intactos.

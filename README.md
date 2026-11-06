# Imputación v3 · corrida completa desde el CSV crudo

Este directorio es el **único código** necesario para reproducir esta versión
del experimento. No utiliza resultados ni checkpoints de `imputaciones_v2`.
Las variables `tmobile_app` y `cmobile_app_trx` se excluyen **para todos los
métodos** en `02_particionar.py`, antes de imputar y crear lags/deltas. El
script comprueba que ambas tengan cero valores observados de enero a junio;
si los datos cambian, falla en vez de excluir información disponible.

## Protocolo fijo

| ID | Método |
|---|---|
| E0 | NaN nativo de LightGBM |
| E1 | Media de train |
| E2 | Mediana de train |
| E3 | Moda de train |
| E4 | Forward fill por cliente, sólo desde meses anteriores |
| E5 | `IterativeImputer` + BayesianRidge; fit en 60.000 filas, máximo 10 rondas, 25 variables vecinas (configuración que completó v2) |
| E06 | `miceforest` + LightGBM; todo train, `data_subset=0`, 5 rondas MICE, 100 iteraciones LightGBM por variable/ronda, 5 donantes PMM |

La auditoría transforma un **mes-variable** 100% cero en faltante y guarda el
detalle en `raw/variables_100pct_ceros.csv`. No agrega indicadores de missing.
Las dos variables excluidas estaban vacías también en enero y febrero, por lo
que no podían aportar lags al entrenamiento.

- Entrenamiento sin undersampling: 201903–201906; BAJA+1 y BAJA+2 son positivas.
- Optuna: 50 trials **sólo en E0**, ordenados por AUC binaria de 201908.
- E0 elige un N único por ganancia acumulada suavizada en agosto, con radio
  ±500 envíos. La curva completa de agosto queda guardada.
- Test: 201910, con 30 semillas comunes. Todos usan parámetros, iteración y N
  elegidos por E0; la ganancia remunera sólo BAJA+2.
- Wilcoxon pareado por semilla frente a E0, con ajuste Holm. Las semillas
  comparten un mismo mes test; no son 30 tests temporales independientes.

## Archivos

`run_pipeline.py` llama secuencialmente a `01_auditar.py`, `02_particionar.py`,
los scripts `03_imputar_*.py`, `04_modelar_evaluar.py` y `dashboard.R`.
`05_estado.py` muestra checkpoints. `src/` contiene las funciones compartidas
de features y publicación en Cloud Storage. Cada etapa deja logs y marcadores
`DONE.json`; E06 guarda un checkpoint por ronda MICE y por mes. E5 guarda el
imputador al terminar el fit, por lo que una interrupción durante ese fit lo
repite. Los métodos se preparan en el orden E0–E5, E06.
E5 usa su algoritmo anterior sin acotar predicciones al rango observado.

## Instalar y probar

Requiere Python 3.13.9 y R para generar el dashboard. Desde la raíz de este repositorio:

```bash
uv python install 3.13.9
uv venv --python 3.13.9 .venv
uv pip install --python .venv/bin/python -r requirements.txt
Rscript install_r_packages.R
.venv/bin/python -m unittest discover -s tests -v
```

Las pruebas usan clientes inventados; **no entrenan con el CSV real**.

Para correr localmente, indicar rutas propias para el CSV y la salida. No hace
falta `--remote`:

```bash
.venv/bin/python run_pipeline.py \
  --data /ruta/competencia_02_crudo.csv.gz \
  --work /ruta/salidas/imputacion_v3 \
  --step all
```

## Corrida completa en GCP Spot

Clonar este repositorio en la VM y poner el CSV en una ruta accesible. El
trabajo activo va a disco local y los checkpoints a un prefijo nuevo del bucket:

```bash
tmux new -s imputacion_v3
cd /home/ds/exp-colaborativos
.venv/bin/python run_pipeline.py \
  --data /ruta/competencia_02_crudo.csv.gz \
  --work /home/ds/backup/imputacion_v3 \
  --remote gs://TU_BUCKET/exp/imputacion_v3 \
  --step all
```

Salir de tmux con `Ctrl+B`, `D`. Para retomar tras una Spot, restaurar el mismo
código y entorno y repetir el comando. `run_identity.json` impide mezclar
datos, configuración o código diferentes bajo el mismo `--work`/`--remote`.
Para ejecutar sólo una etapa: `--step auditar`, `particionar`, `preparar`,
`modelar`, `dashboard` o `estado` (manteniendo los demás argumentos).

El resultado final está en `resultados/dashboard_comparativo_descargar.zip`:
descomprimir el HTML junto a `lib/` para ver los gráficos. También quedan CSV
de métricas por semilla, ensambles, curvas y `FINALIZADO.json`.

Los resultados de corridas reales se guardan fuera de este directorio de código,
en la ruta indicada con `--work` y, si corresponde, en `--remote`. No se incluyen
datos, modelos, logs ni checkpoints en el repositorio. No mezclar resultados de
esta versión con v2 ni con `imputacion_v3_indicadores`.

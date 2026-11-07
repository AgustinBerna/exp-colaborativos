# Experimentos colaborativos

Dos experimentos de imputación de valores faltantes en churn bancario, con código, configuración y resultados de cada corrida.

| Experimento | Descripción | Iteraciones E0 | N elegido en agosto | Resultados |
|---|---|---:|---:|---|
| [Exp 1 · Imputación](exp_01_imputacion/) | NaN nativo y seis imputaciones | 280 | 4323 | [Dashboard autocontenido](exp_01_imputacion/resultados/dashboard_comparativo.html) · [seleccion.json](exp_01_imputacion/seleccion.json) |
| [Exp 2 · Indicadores de missing](exp_02_indicadores_missing/) | Los mismos métodos con 150 indicadores de ausencia | 232 | 4000 | [Dashboard autocontenido](exp_02_indicadores_missing/resultados/dashboard_comparativo.html) · [seleccion.json](exp_02_indicadores_missing/seleccion.json) |

Para ver un dashboard, descargar el archivo HTML y abrirlo localmente: todas sus dependencias están embebidas. Cada carpeta incluye también un CSV con la tabla del dashboard y un manifiesto de publicación con hashes. Se publican los resultados suministrados, sin nuevos entrenamientos.

Ambas corridas entrenan con marzo–junio de 2019, validan en agosto y evalúan octubre con 30 semillas por método. **Cada experimento tiene su propia selección E0 y su propio N**; la comparación entre Exp 1 y Exp 2 es descriptiva y no aísla causalmente el efecto de los indicadores. Los detalles y límites de procedencia están en los README de cada experimento.


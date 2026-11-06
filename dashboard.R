#!/usr/bin/env Rscript
# ==============================================================================
# dashboard.R
# Genera un Dashboard HTML interactivo comparando E0 (Baseline) vs E1..E7
# Métricas clave: Ganancia, ROC-AUC, Average PR (PR-AUC), Lift y estabilidad
# ==============================================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(plotly)
  library(DT)
  library(htmltools)
  library(htmlwidgets)
})

args <- commandArgs(trailingOnly = TRUE)
work_dir <- ifelse(length(args) >= 1, args[1], "./work")
res_dir  <- file.path(work_dir, "resultados")
raw_dir  <- file.path(work_dir, "raw")

# 1. Carga de salidas de los experimentos
seed_metrics <- read_csv(file.path(res_dir, "metricas_por_semilla_test.csv"), show_col_types = FALSE)
ens_metrics  <- read_csv(file.path(res_dir, "resumen_ensamble_test.csv"), show_col_types = FALSE)
ens_curves   <- read_csv(file.path(res_dir, "curvas_ensamble_test.csv"), show_col_types = FALSE)
nan_report   <- read_csv(file.path(raw_dir, "reporte_nan_por_mes.csv"), show_col_types = FALSE)

# Etiquetas descriptivas de los experimentos
method_labels <- c(
  "E0" = "E0: Baseline (NaN nativo)",
  "E1" = "E1: Media (Train)",
  "E2" = "E2: Mediana (Train)",
  "E3" = "E3: Moda (Train)",
  "E4" = "E4: Temporal (Forward Fill)",
  "E5" = "E5: Ridge (120k, 15 iter., 50 vecinas)",
  "E06" = "E06: miceforest (MICE/LightGBM)",
  "E6" = "E6: Autoencoder",
  "E7" = "E7: miceforest (MICE/LightGBM)"
)

seed_metrics <- seed_metrics %>%
  mutate(Experimento = recode(method, !!!method_labels, .default = method))
ens_metrics <- ens_metrics %>%
  mutate(Experimento = recode(method, !!!method_labels, .default = method))
ens_curves <- ens_curves %>%
  mutate(Experimento = recode(method, !!!method_labels, .default = method))

# 2. Tabla Resumen Ejecutiva (Ensamble + Media/SD de Semillas + Wilcoxon)
summary_table <- seed_metrics %>%
  group_by(method, Experimento) %>%
  summarise(
    ROC_AUC_Media = mean(roc_auc),
    ROC_AUC_SD    = sd(roc_auc),
    PR_AUC_Media  = mean(pr_auc),
    Lift_Media    = mean(lift_at_n_fijo),
    Ganancia_Media = mean(gain_at_n_fijo),
    Ganancia_SD    = sd(gain_at_n_fijo),
    .groups = "drop"
  ) %>%
  left_join(
    ens_metrics %>% select(method, Ensamble_ROC_AUC = roc_auc, Ensamble_PR_AUC = pr_auc,
                           Ensamble_Ganancia_N_Fijo = gain_at_n_fijo, N_Fijo_Validacion = n_fijo_evaluado,
                           any_of(c("p_val_auc_vs_E0", "p_holm_auc_vs_E0", "p_val_gain_vs_E0", "p_holm_gain_vs_E0"))),
    by = "method"
  ) %>%
  arrange(desc(Ensamble_Ganancia_N_Fijo))

dt_widget <- datatable(
  summary_table,
  rownames = FALSE,
  options = list(pageLength = 10, dom = "t", scrollX = TRUE),
  caption = "Comparación en Test (201910): Métricas por Semillero y Ensamble de Semillas"
) %>%
  formatRound(columns = c("ROC_AUC_Media", "ROC_AUC_SD", "PR_AUC_Media", "Ensamble_ROC_AUC", "Ensamble_PR_AUC"), digits = 5) %>%
  formatRound(columns = c("Lift_Media"), digits = 2) %>%
  formatCurrency(columns = c("Ganancia_Media", "Ganancia_SD", "Ensamble_Ganancia_N_Fijo"), currency = "$", digits = 0)

# 3. Curvas de ganancia hasta el N elegido con E0 en agosto.
n_fijo <- unique(ens_metrics$n_fijo_evaluado)
stopifnot(length(n_fijo) == 1)
ens_curves <- ens_curves %>%
  filter(envios <= n_fijo) %>%
  group_by(method) %>%
  arrange(envios, .by_group = TRUE) %>%
  mutate(ganancia_suave = vapply(seq_along(ganancia), function(i) {
           mean(ganancia[max(1, i - 2):min(length(ganancia), i + 2)])
         }, numeric(1))) %>%
  ungroup()
g_ganancia <- ggplot(ens_curves, aes(x = envios, color = Experimento)) +
  geom_line(aes(y = ganancia / 1e6), linewidth = 0.45, alpha = 0.25) +
  geom_line(aes(y = ganancia_suave / 1e6), linewidth = 1) +
  geom_vline(xintercept = n_fijo, linetype = "dashed", color = "gray40") +
  labs(
    title = "Ganancia en Test hasta N fijado en Validación (línea gruesa suavizada)",
    x = "Cantidad de Envíos Estímulo (N)",
    y = "Ganancia Acumulada (Millones de $)"
  ) +
  theme_minimal()
p_ganancia <- ggplotly(g_ganancia)

# 4. Gráfico 2: Curva de Lift Acumulado por Cantidad de Envíos
g_lift <- ggplot(ens_curves, aes(x = envios, y = lift, color = Experimento)) +
  geom_line(linewidth = 1) +
  geom_hline(yintercept = 1, linetype = "dashed", color = "gray40") +
  labs(
    title = "Curva de Lift Acumulado en Test (201910)",
    x = "Cantidad de Envíos (N)",
    y = "Lift sobre Tasa Base"
  ) +
  theme_minimal()
p_lift <- ggplotly(g_lift)

# 5. Puntos de precisión/recall BAJA+2 hasta N (no es la AP binaria completa).
g_pr <- ggplot(ens_curves, aes(x = recall, y = precision, color = Experimento)) +
  geom_line(linewidth = 1) +
  labs(
    title = "Precisión y recall BAJA+2 hasta N en Test (puntos de la curva)",
    x = "Recall (Sensibilidad)",
    y = "Precision (Valor Predictivo Positivo)"
  ) +
  theme_minimal()
p_pr <- ggplotly(g_pr)

# 6. Gráfico 4: Distribución de Ganancia y ROC-AUC por Semilla (Semillero)
g_box_gain <- ggplot(seed_metrics, aes(x = reorder(Experimento, gain_at_n_fijo, FUN = median), y = gain_at_n_fijo / 1e6, fill = Experimento)) +
  geom_boxplot(alpha = 0.7) +
  geom_jitter(width = 0.15, alpha = 0.8, size = 2) +
  coord_flip() +
  labs(
    title = "Estabilidad por Semilla: Ganancia al N fijo en Test (201910)",
    x = "",
    y = "Ganancia al N fijo (Millones de $)"
  ) +
  theme_minimal() +
  theme(legend.position = "none")
p_box_gain <- ggplotly(g_box_gain)

g_box_auc <- ggplot(seed_metrics, aes(x = reorder(Experimento, roc_auc, FUN = median), y = roc_auc, fill = Experimento)) +
  geom_boxplot(alpha = 0.7) +
  geom_jitter(width = 0.15, alpha = 0.8, size = 2) +
  coord_flip() +
  labs(
    title = "Estabilidad por Semilla: ROC-AUC en Test (201910)",
    x = "",
    y = "ROC-AUC (201910)"
  ) +
  theme_minimal() +
  theme(legend.position = "none")
p_box_auc <- ggplotly(g_box_auc)

# 7. Gráfico 5: Auditoría de % NaN y Variables con 100% Ceros por Mes (Etapa 01)
nan_mes_resumen <- nan_report %>%
  group_by(foto_mes) %>%
  summarise(
    Pct_NaN_Promedio = mean(pct_nan),
    Vars_Con_Faltantes = sum(n_nan > 0),
    Vars_Rotas_100pct_Cero = sum(convertida_de_100pct_cero),
    .groups = "drop"
  ) %>%
  mutate(foto_mes = as.factor(foto_mes))

g_nan <- ggplot(nan_mes_resumen, aes(x = foto_mes, y = Vars_Rotas_100pct_Cero, fill = Pct_NaN_Promedio)) +
  geom_col() +
  labs(
    title = "Auditoría Etapa 01: Variables con 100% Ceros Convertidas a NaN por Mes",
    x = "Mes (foto_mes)",
    y = "Cant. Variables 100% Cero -> NaN"
  ) +
  theme_minimal()
p_nan <- ggplotly(g_nan)

# 8. Ensamblado del Dashboard en un único archivo HTML autónomo
dashboard_ui <- tagList(
  tags$head(
    tags$title("Dashboard Comparativo de Imputación - 2019"),
    tags$style(HTML("
      body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 24px; background: #f8f9fa; color: #212529; }
      .card { background: white; border-radius: 8px; padding: 20px; margin-bottom: 24px; box-shadow: 0 2px 6px rgba(0,0,0,0.06); }
      .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
      h1, h2 { color: #1a365d; margin-top: 0; }
    "))
  ),
  tags$h1("Dashboard Experimental: Estrategias de Imputación (2019)"),
  tags$p("Entrenamiento: 201903-201906 (BAJA+1 + BAJA+2) | Validación BO: 201908 (AUC) | Evaluación Test: 201910 (Ganancia, ROC-AUC, PR-AUC, Lift)"),
  div(class = "card", tags$h2("1. Resumen Ejecutivo de Métricas y Test de Wilcoxon vs E0"), dt_widget),
  div(class = "grid-2",
    div(class = "card", p_ganancia),
    div(class = "card", p_lift)
  ),
  div(class = "grid-2",
    div(class = "card", p_box_gain),
    div(class = "card", p_box_auc)
  ),
  div(class = "grid-2",
    div(class = "card", p_pr),
    div(class = "card", p_nan)
  )
)

out_html <- file.path(res_dir, "dashboard_comparativo.html")
save_html(dashboard_ui, file = out_html)
cat(sprintf("Dashboard generado con éxito en: %s\n", out_html))

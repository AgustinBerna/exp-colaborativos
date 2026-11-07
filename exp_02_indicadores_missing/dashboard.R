#!/usr/bin/env Rscript
# Dashboard v3: N se elige con E0 en agosto; octubre sólo compara métodos.
suppressPackageStartupMessages({
  library(tidyverse)
  library(plotly)
  library(DT)
  library(htmltools)
  library(htmlwidgets)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1) stop("Uso: Rscript dashboard.R <work>")
work <- normalizePath(args[1], mustWork = TRUE)
res <- file.path(work, "resultados")
seed <- read_csv(file.path(res, "metricas_por_semilla_test.csv"), show_col_types = FALSE)
ens <- read_csv(file.path(res, "resumen_ensamble_test.csv"), show_col_types = FALSE)
curves <- read_csv(file.path(res, "curvas_ensamble_test.csv"), show_col_types = FALSE)
valid <- read_csv(file.path(work, "E0", "optimization", "curva_ganancia_valid_e0.csv"), show_col_types = FALSE)
nan_report <- read_csv(file.path(work, "raw", "reporte_nan_por_mes.csv"), show_col_types = FALSE)
selected <- jsonlite::fromJSON(file.path(work, "E0", "optimization", "seleccion.json"))
indicator_plan <- jsonlite::fromJSON(file.path(work, "partition", "indicadores_ausente.json"))
n_fijo <- unique(ens$n_fijo_evaluado)
stopifnot(length(n_fijo) == 1, n_fijo == selected$n_selected_valid,
          identical(sort(unique(seed$method)), sort(c("E0", "E1", "E2", "E3", "E4", "E5", "E06"))))

labels <- c(E0 = "E0 · NaN nativo", E1 = "E1 · Media", E2 = "E2 · Mediana",
            E3 = "E3 · Moda", E4 = "E4 · Forward fill",
            E5 = "E5 · Ridge 60k/10/25", E06 = "E06 · miceforest")
decorate <- function(df) {
  df %>% mutate(Experimento = factor(unname(labels[method]), levels = unname(labels)))
}
seed <- decorate(seed)
ens <- decorate(ens)
curves <- decorate(curves)

# Los p-valores vienen del Wilcoxon pareado por semilla, con ajuste Holm.
stats <- seed %>% group_by(method, Experimento) %>%
  summarise(auc_media = mean(roc_auc), auc_sd = sd(roc_auc),
            ap_media = mean(pr_auc), ganancia_media = mean(gain_at_n_fijo),
            ganancia_sd = sd(gain_at_n_fijo), lift_medio = mean(lift_at_n_fijo),
            recall_medio = mean(recall_baja2_at_n_fijo), .groups = "drop") %>%
  left_join(ens %>% select(method, auc_ensamble = roc_auc,
                           ganancia_ensamble = gain_at_n_fijo,
                           p_holm_auc_vs_E0, p_holm_gain_vs_E0), by = "method")
base_auc <- stats$auc_media[stats$method == "E0"]
base_gain <- stats$ganancia_media[stats$method == "E0"]
stars <- function(p) case_when(is.na(p) ~ "?", p < .001 ~ "***", p < .01 ~ "**",
                               p < .05 ~ "*", TRUE ~ "ns")
stats <- stats %>% mutate(
  auc_marca = if_else(method == "E0", "referencia",
                      paste0(stars(p_holm_auc_vs_E0), if_else(auc_media > base_auc, " ↑", " ↓"))),
  ganancia_marca = if_else(method == "E0", "referencia",
                           paste0(stars(p_holm_gain_vs_E0),
                                  if_else(ganancia_media > base_gain, " ↑", " ↓")))
)
tabla <- datatable(
  stats %>% select(Experimento, auc_media, auc_sd, ap_media, ganancia_media,
                   ganancia_sd, lift_medio, recall_medio, auc_ensamble,
                   ganancia_ensamble, p_holm_auc_vs_E0, p_holm_gain_vs_E0,
                   auc_marca, ganancia_marca),
  rownames = FALSE, options = list(pageLength = 5, dom = "t", scrollX = TRUE),
  caption = "Octubre 2019 · 30 semillas · mismo N fijado en agosto"
) %>%
  formatRound(columns = c("auc_media", "auc_sd", "ap_media", "auc_ensamble"), digits = 5) %>%
  formatRound(columns = c("lift_medio", "recall_medio", "p_holm_auc_vs_E0",
                            "p_holm_gain_vs_E0"), digits = 4) %>%
  formatCurrency(columns = c("ganancia_media", "ganancia_sd", "ganancia_ensamble"),
                 currency = "$", digits = 0)

# CSV de agosto conserva TODOS los N; se submuestrea sólo la visualización.
limit_valid <- min(max(15000L, n_fijo + 500L), max(valid$envios))
valid_plot <- valid %>% filter(envios <= limit_valid,
                                envios %% 25L == 0L | envios == n_fijo |
                                  envios == selected$n_raw_max_valid)
g_valid <- ggplot(valid_plot, aes(x = envios)) +
  geom_line(aes(y = ganancia / 1e6, color = "Ganancia acumulada"),
            linewidth = .6, alpha = .6) +
  geom_line(aes(y = ganancia_suavizada / 1e6, color = "Meseta suavizada"),
            linewidth = 1.3) +
  geom_vline(xintercept = n_fijo, linetype = "dashed", color = "#13834b") +
  annotate("text", x = n_fijo, y = max(valid_plot$ganancia / 1e6),
           label = paste("N elegido:", n_fijo), hjust = 0, vjust = -0.6,
           color = "#13834b", size = 3.5) +
  scale_color_manual(values = c("Ganancia acumulada" = "#8998ac",
                                "Meseta suavizada" = "#1768a8"), name = NULL) +
  labs(title = "E0 · Ganancia en validación (agosto 2019)",
       subtitle = paste0("N = ", n_fijo, " por máximo de la media móvil ±",
                         selected$smoothing_radius_n, "; curva hasta ", limit_valid, " envíos"),
       x = "Cantidad de envíos", y = "Ganancia acumulada (millones de $)") + theme_minimal()

# Octubre se grafica únicamente hasta el N decidido sin ver test.
curves <- curves %>% filter(envios <= n_fijo) %>% group_by(method) %>%
  arrange(envios, .by_group = TRUE) %>%
  mutate(ganancia_suave = vapply(seq_along(ganancia), function(i) {
           mean(ganancia[max(1, i - 2):min(length(ganancia), i + 2)])
         }, numeric(1))) %>% ungroup()
g_gain <- ggplot(curves, aes(envios, ganancia_suave / 1e6, color = Experimento)) +
  geom_line(linewidth = 1) + geom_vline(xintercept = n_fijo, linetype = "dashed") +
  labs(title = "Ganancia en octubre hasta N", x = "Envíos", y = "Millones de $") + theme_minimal()
g_lift <- ggplot(curves, aes(envios, lift, color = Experimento)) +
  geom_line(linewidth = 1) +
  labs(title = "Lift BAJA+2 en octubre", x = "Envíos", y = "Lift") + theme_minimal()
g_pr <- ggplot(curves, aes(recall, precision, color = Experimento)) +
  geom_line(linewidth = 1) +
  labs(title = "Precisión y recall BAJA+2 hasta N", x = "Recall", y = "Precisión") + theme_minimal()

# Las marcas muestran significancia Y dirección frente a E0.
auc_y <- max(seed$roc_auc) + max(.0005, diff(range(seed$roc_auc)) * .12)
gain_y <- max(seed$gain_at_n_fijo / 1e6) +
  max(1, diff(range(seed$gain_at_n_fijo / 1e6)) * .12)
g_box_auc <- ggplot(seed, aes(Experimento, roc_auc, fill = Experimento)) +
  geom_boxplot(alpha = .65, outlier.shape = NA) +
  geom_jitter(width = .12, size = 1.3, alpha = .65) +
  geom_text(data = stats, aes(x = Experimento, y = auc_y, label = auc_marca),
            inherit.aes = FALSE, size = 3.3) +
  scale_y_continuous(expand = expansion(mult = c(.05, .18))) + coord_flip(clip = "off") +
  labs(title = "AUC por semilla · Wilcoxon vs E0", x = NULL, y = "AUC binaria") +
  theme_minimal() + theme(legend.position = "none")
g_box_gain <- ggplot(seed, aes(Experimento, gain_at_n_fijo / 1e6, fill = Experimento)) +
  geom_boxplot(alpha = .65, outlier.shape = NA) +
  geom_jitter(width = .12, size = 1.3, alpha = .65) +
  geom_text(data = stats, aes(x = Experimento, y = gain_y, label = ganancia_marca),
            inherit.aes = FALSE, size = 3.3) +
  scale_y_continuous(expand = expansion(mult = c(.05, .18))) + coord_flip(clip = "off") +
  labs(title = "Ganancia por semilla · Wilcoxon vs E0", x = NULL,
       y = "Ganancia al N fijo (millones de $)") +
  theme_minimal() + theme(legend.position = "none")

nan_summary <- nan_report %>% group_by(foto_mes) %>%
  summarise(pct_nan_medio = mean(pct_nan),
            variables_100pct_cero = sum(convertida_de_100pct_cero), .groups = "drop")
g_nan <- ggplot(nan_summary, aes(factor(foto_mes), variables_100pct_cero,
                                 fill = pct_nan_medio)) + geom_col() +
  labs(title = "Variables 100% cero por mes", x = "Mes", y = "Variables convertidas a NaN") +
  theme_minimal()

card <- function(title, ...) div(class = "card", tags$h2(title), ...)
dashboard <- tagList(
  tags$head(tags$title("Imputaciones v3 con indicadores · resultados 2019"),
            tags$style(HTML("body{font-family:Arial,sans-serif;margin:24px;background:#f5f7fa;color:#172b4d}.card{background:white;padding:20px;border-radius:10px;margin:20px 0}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(480px,1fr));gap:20px}.grid .card{margin:0}h1,h2{color:#12345c}"))),
  tags$h1("Experimento de imputaciones · v3 con indicadores de ausencia"),
  tags$p("Train: marzo-junio 2019 · Validación: agosto · Test: octubre · 30 semillas · ",
         nrow(indicator_plan), " indicadores originales"),
  card("1. Cómo se eligió N en validación", ggplotly(g_valid),
       tags$p("Optuna ordenó modelos por AUC binaria. Con el mejor E0 se calculó la ganancia para todos los N de agosto y se eligió la meseta suavizada. Octubre no participó.")),
  card("2. Comparación en octubre al N fijo", tabla,
       tags$p("* p<0,05 · ** p<0,01 · *** p<0,001 · ns: no significativo. Wilcoxon pareado por semilla, con ajuste Holm dentro de cada métrica. ↑/↓ indica dirección del promedio frente a E0.")),
  div(class = "grid", card("AUC por semilla", ggplotly(g_box_auc)),
      card("Ganancia por semilla", ggplotly(g_box_gain))),
  div(class = "grid", card("Curva de ganancia en test", ggplotly(g_gain)),
      card("Lift en test", ggplotly(g_lift))),
  div(class = "grid", card("Precisión y recall", ggplotly(g_pr)),
      card("Auditoría de faltantes", ggplotly(g_nan))),
  tags$p("Las semillas comparten el mismo test: el Wilcoxon mide estabilidad frente a la aleatoriedad del modelo, no generalización temporal independiente.")
)

out <- file.path(res, "dashboard_comparativo.html")
save_html(dashboard, file = out, libdir = "lib")
cat("Dashboard generado:", out, "\n")
# HTML + lib en un ZIP para descargar desde JupyterLab sin perder gráficos.
if (nzchar(Sys.which("zip"))) {
  previous <- getwd()
  setwd(res)
  utils::zip("dashboard_comparativo_descargar.zip",
             files = c("dashboard_comparativo.html", "lib"), flags = "-r9X")
  setwd(previous)
  cat("ZIP descargable:", file.path(res, "dashboard_comparativo_descargar.zip"), "\n")
}

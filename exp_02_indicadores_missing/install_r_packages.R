# Run once in a new local or GCP environment: Rscript install_r_packages.R
packages <- c("tidyverse", "plotly", "DT", "htmltools", "htmlwidgets", "jsonlite")
missing <- packages[!vapply(packages, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing)) install.packages(missing, repos = "https://cloud.r-project.org")
cat("Paquetes R disponibles:", paste(packages, collapse = ", "), "\n")

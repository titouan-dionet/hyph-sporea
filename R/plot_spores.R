library(data.table)
library(ggplot2)

# Définir le dossier contenant les fichiers CSV
dossier <- here::here("outputs/model/results/object_count")

# Lister tous les fichiers CSV qui contiennent "detections_summary.csv"
fichiers <- list.files(path = dossier, pattern = "detections_summary\\.csv$", full.names = TRUE)

# Fonction pour extraire les informations du nom de fichier
extraire_info <- function(fichier) {
  # Extraire le nom du fichier seul (sans le chemin)
  file_name <- basename(fichier)
  
  # Séparer les parties du nom de fichier
  split_parts <- unlist(strsplit(file_name, "_"))
  
  # Récupérer la date et l'heure
  date <- as.Date(split_parts[1], format = c("%Y%m%d"))
  heure <- split_parts[2]
  
  # Reconstruire le sample en excluant "detections_summary.csv"
  sample_name <- paste(split_parts[3:(length(split_parts)-2)], collapse = "_")
  
  # Vérifier si le dernier élément du sample est un nombre (réplicat)
  split_parts_sample <- unlist(strsplit(sample_name, "_"))
  dernier_element <- tail(split_parts_sample, 1)
  
  if (grepl("^[0-9]+$", dernier_element)) {
    replicate <- as.integer(dernier_element)  # Extraire le réplicat
    sample <- paste(split_parts_sample[-length(split_parts_sample)], collapse = "_")  # Nom sans réplicat
  } else {
    replicate <- NA  # Pas de réplicat
    sample <- sample_name
  }
  
  return(list(date = date, heure = heure, sample_name = sample_name, sample = sample, replicate = replicate))
}

# Charger et traiter tous les fichiers
dt_list <- lapply(fichiers, function(fichier) {
  # Lire le fichier CSV en data.table
  dt <- fread(fichier)
  
  # Extraire les infos du fichier
  info <- extraire_info(fichier)
  
  # Ajouter les colonnes à la table
  dt[, `:=`(
    date = info$date,
    heure = info$heure,
    sample_name = info$sample_name,
    sample = info$sample,
    replicate = info$replicate
  )]
  
  return(dt)
})

# Combiner tous les fichiers en un seul data.table
dt_total <- rbindlist(dt_list, fill = TRUE)

# Résumer les données par "date", "heure", "sample" et "replicate"
dt_summary <- dt_total[, .(
  spore_count = sum(spore_count, na.rm = TRUE),
  debris_count = sum(debris_count, na.rm = TRUE),
  mycelium_count = sum(mycelium_count, na.rm = TRUE)
), by = .(date, heure, sample_name, sample, replicate)]

# Afficher le résumé
print(dt_summary)

# Graph
ggplot(data = dt_summary) +
  aes(x = sample, y = spore_count) +
  geom_boxplot() +
  labs(
    title = "Distribution du nombre de spores par échantillon",
    x = "Échantillon",
    y = "Nombre de spores",
    color = "Type"
  ) +
  theme_bw()

# Sauvegarder le résumé dans un fichier CSV
# fwrite(dt_summary, file = file.path(dossier, "summary_counts.csv"))

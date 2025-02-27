library(data.table)
library(ggplot2)

# Définir le dossier contenant les fichiers CSV
dossier <- here::here("outputs/model_CLAQ/results/object_count")

# Lister tous les fichiers CSV qui contiennent "detections_summary.csv"
fichiers <- list.files(path = dossier, pattern = "detections_summary\\.csv$", full.names = TRUE)

# Fonction pour extraire la date depuis le nom du fichier
extraire_date_heure <- function(fichier) {
  file_name <- basename(fichier)
  # Séparer les parties du nom de fichier
  split_parts <- unlist(strsplit(file_name, "_"))
  
  # Récupérer la date et l'heure
  date <- as.Date(split_parts[1], format = c("%Y%m%d"))
  heure <- split_parts[2]
  
  return(list(date = date, heure = heure))
}

# Fonction pour extraire les informations depuis le nom de l'image
extraire_info_image <- function(image_name) {
  # Retirer l'extension
  image_name <- tools::file_path_sans_ext(image_name)
  
  # Séparer le nom de l'image en parties avec "_"
  split_parts <- unlist(strsplit(image_name, "_"))
  
  # Récupérer les infos
  item_name <- paste(split_parts[1:4], collapse = "_")  # Ex: "T5_CLAQ_T_1"
  sample_name <- paste(split_parts[1:3], collapse = "_") # Ex: "T5_CLAQ_T"
  replicate <- as.integer(split_parts[4])  # Ex: "1"
  
  return(list(item_name = item_name, sample_name = sample_name, replicate = replicate))
}

# Charger et traiter tous les fichiers
dt_list <- lapply(fichiers, function(fichier) {
  # Lire le fichier CSV en data.table
  dt <- fread(fichier)
  
  # Extraire la date et l'heure du fichier
  info_fichier <- extraire_date_heure(fichier)
  
  # Appliquer la fonction à chaque image
  info_list <- lapply(dt$image, extraire_info_image)
  
  # Ajouter les colonnes à la table
  dt[, `:=`(
    date = info_fichier$date,
    heure = info_fichier$heure,
    item_name = sapply(info_list, `[[`, "item_name"),
    sample_name = sapply(info_list, `[[`, "sample_name"),
    replicate = sapply(info_list, `[[`, "replicate")
  )]
  
  return(dt)
})

# Combiner tous les fichiers en un seul data.table
dt_total <- rbindlist(dt_list, fill = TRUE)

# Résumer les données par "date", "sample_name" et "replicate"
dt_summary <- dt_total[, .(
  CLAQ_count = sum(CLAQ_count, na.rm = TRUE)
), by = .(date, heure, item_name, sample_name, replicate)]

# Afficher le résumé
print(dt_summary)

# Graphique avec ggplot2
ggplot(data = dt_summary[replicate %in% c(1,2)]) +
  aes(x = sample_name, y = CLAQ_count) +
  geom_point() +
  labs(
    title = "Distribution du nombre de spores par échantillon",
    x = "Échantillon",
    y = "Nombre de spores"
  ) +
  theme_bw()

ggplot(data = dt_summary[replicate %in% c(3, 4)]) +
  aes(x = sample_name, y = CLAQ_count) +
  geom_point() +
  labs(
    title = "Distribution du nombre de spores par échantillon",
    x = "Échantillon",
    y = "Nombre de spores"
  ) +
  theme_bw()

ggplot(data = dt_summary) +
  aes(x = sample_name, y = CLAQ_count) +
  geom_point() +
  labs(
    title = "Distribution du nombre de spores par échantillon",
    x = "Échantillon",
    y = "Nombre de spores"
  ) +
  theme_bw()

# Sauvegarde du résumé dans un fichier CSV
# fwrite(dt_summary, file = file.path(dossier, "summary_counts.csv"))

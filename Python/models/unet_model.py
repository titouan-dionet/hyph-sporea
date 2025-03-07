"""
Module pour l'implémentation du modèle U-Net dans le projet HYPH-SPOREA.

Ce module contient des fonctions pour créer, entraîner et utiliser
le modèle U-Net pour la segmentation des spores d'hyphomycètes.
"""

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate
from tensorflow.keras.optimizers import Adam
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from pathlib import Path
import matplotlib.pyplot as plt


def create_unet_model(input_size=(256, 256, 3), dropout_rate=0.1):
    """
    Crée un modèle U-Net pour la segmentation d'images.
    
    Args:
        input_size (tuple, optional): Taille d'entrée (hauteur, largeur, canaux). Par défaut (256, 256, 3).
        dropout_rate (float, optional): Taux de dropout pour la régularisation. Par défaut 0.1.
    
    Returns:
        tensorflow.keras.Model: Modèle U-Net compilé
    
    Example:
        >>> model = create_unet_model()
        >>> model.summary()
    """
    inputs = Input(input_size)
    
    # Encoder (partie descendante)
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    d1 = Dropout(dropout_rate)(p1)
    
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(d1)
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
    d2 = Dropout(dropout_rate)(p2)
    
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(d2)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
    d3 = Dropout(dropout_rate)(p3)
    
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(d3)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D((2, 2))(c4)
    d4 = Dropout(dropout_rate)(p4)
    
    # Pont
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(d4)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    d5 = Dropout(dropout_rate)(c5)
    
    # Decoder (partie montante)
    u6 = UpSampling2D((2, 2))(d5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
    d6 = Dropout(dropout_rate)(c6)
    
    u7 = UpSampling2D((2, 2))(d6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
    d7 = Dropout(dropout_rate)(c7)
    
    u8 = UpSampling2D((2, 2))(d7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
    d8 = Dropout(dropout_rate)(c8)
    
    u9 = UpSampling2D((2, 2))(d8)
    u9 = concatenate([u9, c1])
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
    
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model


def prepare_training_data(image_paths, mask_paths, img_size=(256, 256)):
    """
    Prépare les données d'entraînement pour U-Net.
    
    Args:
        image_paths (list): Liste des chemins d'images
        mask_paths (list): Liste des chemins de masques correspondants
        img_size (tuple, optional): Taille de redimensionnement. Par défaut (256, 256).
    
    Returns:
        tuple: (images X, masques Y) sous forme de tableaux NumPy normalisés
    
    Example:
        >>> image_paths = [f"data/proc_data/preprocessed/image_{i}.jpeg" for i in range(10)]
        >>> mask_paths = [f"data/proc_data/masks/image_{i}.png" for i in range(10)]
        >>> X, Y = prepare_training_data(image_paths, mask_paths)
        >>> print(f"X shape: {X.shape}, Y shape: {Y.shape}")
    """
    X = []
    Y = []
    
    for img_path, mask_path in zip(image_paths, mask_paths):
        try:
            # Chargement et redimensionnement de l'image
            img = cv2.imread(img_path)
            if img is None:
                print(f"Impossible de lire l'image: {img_path}")
                continue
                
            img = cv2.resize(img, img_size)
            img = img / 255.0  # Normalisation
            
            # Chargement et redimensionnement du masque
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"Impossible de lire le masque: {mask_path}")
                continue
                
            mask = cv2.resize(mask, img_size)
            mask = mask / 255.0  # Normalisation
            mask = np.expand_dims(mask, axis=-1)  # Ajout d'une dimension pour le canal
            
            X.append(img)
            Y.append(mask)
            
        except Exception as e:
            print(f"Erreur lors de la préparation de {img_path}: {str(e)}")
    
    if not X or not Y:
        raise ValueError("Aucune donnée valide à préparer")
        
    return np.array(X), np.array(Y)


def train_unet(image_dir, mask_dir, output_model_path, epochs=50, batch_size=16, validation_split=0.2):
    """
    Entraîne un modèle U-Net avec les images et masques fournis.
    
    Args:
        image_dir (str): Répertoire contenant les images d'entraînement
        mask_dir (str): Répertoire contenant les masques de segmentation
        output_model_path (str): Chemin pour sauvegarder le modèle entraîné
        epochs (int, optional): Nombre d'époques d'entraînement. Par défaut 50.
        batch_size (int, optional): Taille des batchs. Par défaut 16.
        validation_split (float, optional): Fraction des données pour la validation. Par défaut 0.2.
    
    Returns:
        tensorflow.keras.Model: Modèle entraîné
    
    Example:
        >>> model = train_unet(
        ...     "data/proc_data/preprocessed",
        ...     "data/proc_data/masks",
        ...     "outputs/models/unet_model.h5"
        ... )
    """
    # Listage des fichiers d'images et de masques
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)
    
    # Trouver des paires d'images et de masques
    image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.jpeg', '.jpg', '.png'))])
    
    image_paths = []
    mask_paths = []
    
    for img_file in image_files:
        img_path = image_dir / img_file
        # Tenter de trouver le masque correspondant
        mask_file = Path(img_file).stem + '.png'  # Masques généralement en PNG
        mask_path = mask_dir / mask_file
        
        if mask_path.exists():
            image_paths.append(str(img_path))
            mask_paths.append(str(mask_path))
    
    if not image_paths or not mask_paths:
        raise ValueError(f"Aucune paire image-masque trouvée dans {image_dir} et {mask_dir}")
    
    print(f"Trouvé {len(image_paths)} paires image-masque")
    
    # Division en ensembles d'entraînement et de validation
    train_img_paths, val_img_paths, train_mask_paths, val_mask_paths = train_test_split(
        image_paths, mask_paths, test_size=validation_split, random_state=42
    )
    
    print(f"Entraînement sur {len(train_img_paths)} images, validation sur {len(val_img_paths)} images")
    
    # Préparation des données
    X_train, Y_train = prepare_training_data(train_img_paths, train_mask_paths)
    X_val, Y_val = prepare_training_data(val_img_paths, val_mask_paths)
    
    # Création du modèle
    model = create_unet_model()
    
    # Création du répertoire de sortie si nécessaire
    output_model_path = Path(output_model_path)
    output_model_path.parent.mkdir(exist_ok=True, parents=True)
    
    # Callbacks pour l'entraînement
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(output_model_path), 
            save_best_only=True, 
            monitor='val_loss'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.2, 
            patience=5, 
            min_lr=1e-6
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=str(output_model_path.parent / 'logs'),
            histogram_freq=1
        )
    ]
    
    # Entraînement avec augmentation de données en temps réel
    data_gen_args = dict(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.05,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )
    
    # Création d'un générateur d'augmentation
    image_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**data_gen_args)
    mask_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**data_gen_args)
    
    # Initialisation des générateurs
    seed = 42
    image_generator = image_datagen.flow(X_train, seed=seed, batch_size=batch_size)
    mask_generator = mask_datagen.flow(Y_train, seed=seed, batch_size=batch_size)
    
    # Combinaison des générateurs
    train_generator = zip(image_generator, mask_generator)
    
    # Entraînement du modèle
    history = model.fit(
        train_generator,
        steps_per_epoch=len(X_train) // batch_size,
        validation_data=(X_val, Y_val),
        epochs=epochs,
        callbacks=callbacks
    )
    
    # Visualisation des résultats d'entraînement
    plot_training_history(history, output_model_path.parent / 'training_history.png')
    
    print(f"Modèle U-Net entraîné et sauvegardé dans {output_model_path}")
    
    return model


def predict_with_unet(model, image_path, output_dir=None):
    """
    Prédit la segmentation d'une image avec U-Net.
    
    Args:
        model: Modèle U-Net chargé ou chemin vers le modèle
        image_path (str): Chemin de l'image à segmenter
        output_dir (str, optional): Répertoire de sortie pour les résultats
    
    Returns:
        tuple: (image avec contours, masque prédit, contours détectés)
    
    Example:
        >>> model = load_model("outputs/models/unet_model.h5")
        >>> result_img, mask, contours = predict_with_unet(
        ...     model,
        ...     "data/proc_data/jpeg_images/T1_ALAC_C6_1/T1_ALAC_C6_1_000156.jpeg",
        ...     "outputs/predictions/unet"
        ... )
    """
    # Chargement du modèle si c'est un chemin
    if isinstance(model, (str, Path)):
        model = load_model(model)
    
    # Chargement et prétraitement de l'image
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Impossible de lire l'image: {image_path}")
        
    original_shape = img.shape[:2]
    img_resized = cv2.resize(img, (256, 256))
    img_normalized = img_resized / 255.0
    
    # Prédiction
    pred = model.predict(np.expand_dims(img_normalized, axis=0))[0]
    
    # Post-traitement
    pred_binary = (pred > 0.5).astype(np.uint8) * 255
    pred_resized = cv2.resize(pred_binary, (original_shape[1], original_shape[0]))
    
    # Extraction des contours
    contours, _ = cv2.findContours(pred_resized.squeeze(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Dessin des contours sur l'image
    result_img = img.copy()
    cv2.drawContours(result_img, contours, -1, (0, 255, 0), 2)
    
    # Sauvegarde des résultats
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        base_name = Path(image_path).stem
        
        # Sauvegarde de l'image avec contours
        result_path = output_dir / f"{base_name}_contours.jpg"
        cv2.imwrite(str(result_path), result_img)
        
        # Sauvegarde du masque
        mask_path = output_dir / f"{base_name}_mask.png"
        cv2.imwrite(str(mask_path), pred_resized)
    
    return result_img, pred_resized, contours


def batch_predict_with_unet(model, image_dir, output_dir=None, pattern="*.jpeg"):
    """
    Effectue des prédictions sur un lot d'images avec un modèle U-Net.
    
    Args:
        model: Modèle U-Net chargé ou chemin vers le modèle
        image_dir (str): Répertoire contenant les images à segmenter
        output_dir (str, optional): Répertoire de sortie pour les résultats
        pattern (str, optional): Motif pour filtrer les fichiers. Par défaut "*.jpeg".
    
    Returns:
        dict: Dictionnaire contenant les détections par image
    
    Example:
        >>> detections = batch_predict_with_unet(
        ...     "outputs/models/unet_model.h5",
        ...     "data/proc_data/jpeg_images/T1_ALAC_C6_1",
        ...     "outputs/predictions/unet/T1_ALAC_C6_1"
        ... )
    """
    # Chargement du modèle si c'est un chemin
    if isinstance(model, (str, Path)):
        model = load_model(model)
    
    # Configuration du répertoire de sortie
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
    
    # Lister les images
    image_dir = Path(image_dir)
    image_files = list(image_dir.glob(pattern))
    
    if not image_files:
        print(f"Aucune image correspondant au motif '{pattern}' trouvée dans {image_dir}")
        return {}
    
    print(f"Traitement de {len(image_files)} images...")
    
    # Dictionnaire pour stocker les résultats
    detections = {}
    
    # Traiter chaque image
    for i, image_file in enumerate(image_files):
        try:
            # Prédiction
            _, _, contours = predict_with_unet(
                model,
                str(image_file),
                output_dir=output_dir / image_file.stem if output_dir else None
            )
            
            # Extraire les détections
            image_detections = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                area = cv2.contourArea(contour)
                
                # Ne conserver que les contours de taille raisonnable
                if area > 30 and area < 5000:  # Filtrage par taille
                    image_detections.append({
                        'bbox': [float(x), float(y), float(x+w), float(y+h)],
                        'width': float(w),
                        'height': float(h),
                        'area': float(area)
                    })
            
            detections[image_file.name] = image_detections
            
            # Afficher la progression
            if (i + 1) % 10 == 0 or i == len(image_files) - 1:
                print(f"Progression: {i + 1}/{len(image_files)} images traitées")
                
        except Exception as e:
            print(f"Erreur lors du traitement de {image_file}: {str(e)}")
    
    return detections


def plot_training_history(history, output_path=None):
    """
    Visualise l'historique d'entraînement d'un modèle U-Net.
    
    Args:
        history: Historique d'entraînement retourné par model.fit()
        output_path (str, optional): Chemin pour sauvegarder la figure
    
    Returns:
        matplotlib.figure.Figure: Figure créée
    
    Example:
        >>> history = model.fit(...)
        >>> fig = plot_training_history(
        ...     history,
        ...     "outputs/visualizations/unet_training_history.png"
        ... )
    """
    # Extraire les métriques
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs = range(1, len(acc) + 1)
    
    # Créer la figure
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    
    # Précision
    axs[0].plot(epochs, acc, 'bo-', label='Précision d\'entraînement')
    axs[0].plot(epochs, val_acc, 'ro-', label='Précision de validation')
    axs[0].set_title('Précision d\'entraînement et de validation')
    axs[0].set_xlabel('Époques')
    axs[0].set_ylabel('Précision')
    axs[0].legend()
    axs[0].grid(alpha=0.3)
    
    # Perte
    axs[1].plot(epochs, loss, 'bo-', label='Perte d\'entraînement')
    axs[1].plot(epochs, val_loss, 'ro-', label='Perte de validation')
    axs[1].set_title('Perte d\'entraînement et de validation')
    axs[1].set_xlabel('Époques')
    axs[1].set_ylabel('Perte')
    axs[1].legend()
    axs[1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Sauvegarder la figure
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig


def evaluate_unet_model(model, image_paths, mask_paths, output_dir=None):
    """
    Évalue un modèle U-Net sur un ensemble de données.
    
    Args:
        model: Modèle U-Net chargé ou chemin vers le modèle
        image_paths (list): Liste des chemins d'images
        mask_paths (list): Liste des chemins de masques correspondants
        output_dir (str, optional): Répertoire de sortie pour les résultats
    
    Returns:
        dict: Métriques d'évaluation
    
    Example:
        >>> image_paths = [f"data/proc_data/test/images/img_{i}.jpeg" for i in range(10)]
        >>> mask_paths = [f"data/proc_data/test/masks/img_{i}.png" for i in range(10)]
        >>> metrics = evaluate_unet_model(
        ...     "outputs/models/unet_model.h5",
        ...     image_paths,
        ...     mask_paths,
        ...     "outputs/evaluation/unet"
        ... )
        >>> print(f"IoU: {metrics['iou']:.4f}, Dice: {metrics['dice']:.4f}")
    """
    # Chargement du modèle si c'est un chemin
    if isinstance(model, (str, Path)):
        model = load_model(model)
    
    # Configuration du répertoire de sortie
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
    
    if len(image_paths) != len(mask_paths):
        raise ValueError("Le nombre d'images et de masques doit être identique")
    
    # Métriques
    ious = []
    dices = []
    
    # Traiter chaque paire d'image-masque
    for i, (image_path, mask_path) in enumerate(zip(image_paths, mask_paths)):
        try:
            # Charger l'image et le masque de référence
            img = cv2.imread(str(image_path))
            if img is None:
                print(f"Impossible de lire l'image: {image_path}")
                continue
                
            gt_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if gt_mask is None:
                print(f"Impossible de lire le masque: {mask_path}")
                continue
                
            # Binariser le masque de référence
            gt_mask = (gt_mask > 127).astype(np.uint8)
            
            # Effectuer la prédiction
            _, pred_mask, _ = predict_with_unet(model, str(image_path))
            
            # Binariser le masque prédit
            pred_mask = (pred_mask > 127).astype(np.uint8).squeeze()
            
            # Redimensionner si nécessaire
            if gt_mask.shape != pred_mask.shape:
                gt_mask = cv2.resize(gt_mask, (pred_mask.shape[1], pred_mask.shape[0]))
            
            # Calculer les métriques
            intersection = np.logical_and(gt_mask, pred_mask).sum()
            union = np.logical_or(gt_mask, pred_mask).sum()
            
            iou = intersection / (union + 1e-10)
            dice = 2 * intersection / (gt_mask.sum() + pred_mask.sum() + 1e-10)
            
            ious.append(iou)
            dices.append(dice)
            
            # Sauvegarder la comparaison si demandé
            if output_dir:
                base_name = Path(image_path).stem
                
                # Créer une visualisation côte à côte
                h, w = gt_mask.shape
                comparison = np.zeros((h, w * 3, 3), dtype=np.uint8)
                
                # Image originale
                comparison[:, :w] = cv2.cvtColor(cv2.resize(img, (w, h)), cv2.COLOR_BGR2RGB)
                
                # Masque de référence (rouge)
                ref_vis = np.zeros((h, w, 3), dtype=np.uint8)
                ref_vis[gt_mask == 1, 0] = 255  # Rouge pour le masque de référence
                comparison[:, w:2*w] = ref_vis
                
                # Masque prédit (vert)
                pred_vis = np.zeros((h, w, 3), dtype=np.uint8)
                pred_vis[pred_mask == 1, 1] = 255  # Vert pour le masque prédit
                comparison[:, 2*w:] = pred_vis
                
                # Sauvegarder la comparaison
                cv2.imwrite(str(output_dir / f"{base_name}_comparison.png"), comparison)
                
                # Sauvegarder également les métriques dans un fichier texte
                with open(output_dir / f"{base_name}_metrics.txt", 'w') as f:
                    f.write(f"IoU: {iou:.4f}\nDice: {dice:.4f}")
            
        except Exception as e:
            print(f"Erreur lors de l'évaluation de {image_path}: {str(e)}")
    
    # Calculer les métriques moyennes
    mean_iou = np.mean(ious) if ious else 0
    mean_dice = np.mean(dices) if dices else 0
    
    metrics = {
        'iou': mean_iou,
        'dice': mean_dice,
        'individual_ious': ious,
        'individual_dices': dices
    }
    
    # Sauvegarder les métriques globales
    if output_dir:
        with open(output_dir / "global_metrics.txt", 'w') as f:
            f.write(f"Mean IoU: {mean_iou:.4f}\nMean Dice: {mean_dice:.4f}\n")
            f.write(f"Number of images evaluated: {len(ious)}")
    
    return metrics

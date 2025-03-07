# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 20:42:29 2025

@author: Titouan Dionet
"""

#%% Packages

import torch
import torchvision.transforms as transforms
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from torch.utils.data import Dataset, DataLoader
import cv2
import os
import numpy as np
import pyprojroot
import matplotlib.pyplot as plt

#%% Racine du projet
PROJECT_ROOT = pyprojroot.here()

#%% 📌 Chemins du dataset
img_dir = PROJECT_ROOT / "data/proc_data/hypho_train_data_model/model_CLAQ/train/images"
mask_dir = PROJECT_ROOT / "data/proc_data/hypho_train_data_model/model_CLAQ/train/masks"
output_img_dir = PROJECT_ROOT / "outputs/SAM_tune/segmentation_images"  # Dossier pour enregistrer les images combinées
os.makedirs(output_img_dir, exist_ok=True)

#%% 📌 Charger SAM pré-entraîné
sam_checkpoint = PROJECT_ROOT / "data/blank_models" / "sam_vit_h_4b8939.pth"  # Modèle de base
model_type = "vit_h"  # Architecture de SAM
device = "cuda" if torch.cuda.is_available() else "cpu"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device)

#%% 📌 Définition du dataset
class SporeDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.img_files = sorted(os.listdir(img_dir))
        self.mask_files = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convertir en RGB
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Charger le masque

        # Normalisation
        img = img / 255.0
        mask = mask / 255.0

        # Transformation en Tensor
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        return img, mask

def calculate_iou(pred_mask, true_mask):
    intersection = np.sum((pred_mask == 1) & (true_mask == 1))
    union = np.sum((pred_mask == 1) | (true_mask == 1))
    return intersection / union if union != 0 else 0

#%% 📌 Chargement du dataset
dataset = SporeDataset(img_dir, mask_dir)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

#%% 📌 Optimiseur et fonction de perte
optimizer = torch.optim.Adam(sam.parameters(), lr=1e-5)
criterion = torch.nn.BCEWithLogitsLoss()  # Perte adaptée aux masques

#%% 📌 Boucle d'entraînement
num_epochs = 1
sam_predictor = SamPredictor(sam)
losses = []  # Pour stocker la perte à chaque époque
ious = []    # Pour stocker l'IoU à chaque époque


for epoch in range(num_epochs):
    epoch_loss = 0
    epoch_iou = 0
    for idx, (img_batch, mask_batch) in enumerate(dataloader): # Utiliser 'enumerate' pour obtenir 'idx'
        img_batch, mask_batch = img_batch.to(device), mask_batch.to(device)

        optimizer.zero_grad()
        
        # Sélectionner la première image et son masque dans le batch
        img, mask = img_batch[0], mask_batch[0]  # (C, H, W) et (H, W)
        
        # Récupérer le nom du fichier de l'image originale
        original_image_name = dataset.img_files[idx].split('.')[0]  # Extraire le nom sans l'extension

        # Convertir l'image au bon format pour SAM
        img_np = img.permute(1, 2, 0).cpu().numpy()  # Convertir en format numpy (H, W, C)

        # Passer l'image au prédicteur SAM
        sam_predictor.set_image(img_np)
        
        # # Détecter les objets avec les contours
        # gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)  # Convertir l'image RGB en niveaux de gris
        # gray = np.uint8(gray * 255)  # Convertir en uint8
        # _, mask_thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        # contours, _ = cv2.findContours(mask_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # # Créer une image vide pour dessiner les contours (identique à l'image d'entrée)
        # contour_image = np.zeros((img_np.shape[0], img_np.shape[1], 3), dtype=np.uint8)  # Créer une image noire de la même taille que l'image d'entrée
        
        # # Dessiner les contours sur l'image
        # cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)  # Dessiner en vert avec une épaisseur de 2 pixels
        
        # # Convertir l'image en RGB pour l'affichage
        # contour_image_rgb = cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB)
        
        # # Sauvegarder l'image avec les contours
        # output_img_path = os.path.join(output_img_dir, f"epoch_{epoch+1}_contours_{original_image_name}.jpeg")
        # cv2.imwrite(output_img_path, contour_image_rgb)

        # # Parcourir les objets détectés
        # for contour in contours:
        #     if cv2.contourArea(contour) > 100:  # Ignorer les petits objets
        #         # Calculer la bounding box de l'objet
        #         x, y, w, h = cv2.boundingRect(contour)
                
        #         # Définir un point d'interaction au centre de la bounding box
        #         input_point = np.array([[x + w // 2, y + h // 2]])  # (x, y)
        #         input_label = np.array([1])  # 1 = objet
                
        #         # Obtenir la segmentation pour cet objet
        #         masks, _, _ = sam_predictor.predict(point_coords=input_point, point_labels=input_label, multimask_output=False)

        #         # Convertir le masque prédictif en tenseur PyTorch pour la perte
        #         mask_pred = torch.tensor(masks, dtype=torch.float32, device=device, requires_grad=True)  # (N, H, W)
                
        #         # Convertir le masque SAM en format numpy pour l'affichage
        #         mask_pred_np = mask_pred[0].detach().cpu().numpy()
        #         mask_pred_np = (mask_pred_np > 0.1).astype(np.uint8)  # Seuil à 0.1
                
        #         # Convertir l'image en RGB pour l'affichage
        #         img_rgb = img.cpu().numpy().transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
        #         img_rgb = (img_rgb * 255).astype(np.uint8)
                
        #         # Répéter le masque sur 3 canaux pour l'affichage
        #         mask_pred_rgb = (cv2.merge([mask_pred_np] * 3)) * 255  # (H, W) -> (H, W, 3)
                
        #         # Superposer l'image et le masque
        #         combined = cv2.addWeighted(img_rgb, 0.7, mask_pred_rgb, 0.3, 0)
        #         combined = cv2.resize(combined, (600, 400))
                
        #         # Enregistrer l'image combinée dans le dossier de sortie
        #         output_img_path = os.path.join(output_img_dir, f"epoch_{epoch+1}_{original_image_name}_object_{x}_{y}.jpeg")
        #         cv2.imwrite(output_img_path, combined)
                
        #         # Calcul de l'IoU pour cet objet
        #         iou = calculate_iou(mask_pred_np, mask.squeeze(0).cpu().numpy())
        #         epoch_iou += iou  # Additionner l'IoU pour cette époque

        # Définir un point d'interaction (par exemple, le centre de l'image)
        input_point = np.array([[img.shape[2] // 2, img.shape[1] // 2]])  # (x, y)
        input_label = np.array([1])  # 1 = objet

        # Obtenir la segmentation
        masks, _, _ = sam_predictor.predict(point_coords=input_point, point_labels=input_label, multimask_output=False)

        # Convertir le masque SAM en tenseur PyTorch pour la perte
        mask_pred = mask_pred = torch.tensor(masks, dtype=torch.float32, device=device, requires_grad=True)  # (N, H, W)
        
        # Afficher l'image avec le masque prédit
        mask_pred_np = mask_pred[0].detach().cpu().numpy()
        mask_pred_np = (mask_pred_np > 0.1).astype(np.uint8)  # Seuil à 0.1
        
        # Convertir l'image en RGB pour l'affichage
        img_rgb = img.cpu().numpy().transpose(1, 2, 0) # C, H, W -> H, W, C
        img_rgb = (img_rgb * 255).astype(np.uint8)
        
        # Répéter le masque sur 3 canaux pour l'affichage
        mask_pred_rgb = (cv2.merge([mask_pred_np] * 3)) * 255  # (H, W) -> (H, W, 3)
        
        # Superposer l'image et le masque
        combined = cv2.addWeighted(img_rgb, 0.7, mask_pred_rgb, 0.3, 0)
        combined = cv2.resize(combined, (600, 400))
        
        # Enregistrer l'image combinée dans le dossier de sortie
        output_img_path = os.path.join(output_img_dir, f"epoch_{epoch+1}_{original_image_name}.jpeg")
        cv2.imwrite(output_img_path, combined)
        
        # cv2.imshow("Segmentation Prediction", combined)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        # Calcul de l'IoU pour cette image
        iou = calculate_iou(mask_pred_np, mask.squeeze(0).cpu().numpy())
        epoch_iou += iou  # Additionner l'IoU pour cette époque
        
        # Calcul de la perte (prendre le premier masque si plusieurs)
        loss = criterion(mask_pred[0].squeeze(0), mask.squeeze(0))
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    avg_epoch_loss = epoch_loss / len(dataloader)
    avg_epoch_iou = epoch_iou / len(dataloader)
    losses.append(avg_epoch_loss)
    ious.append(avg_epoch_iou)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_epoch_loss:.4f}, IoU: {avg_epoch_iou:.4f}")

#%% Afficher la courbe de perte et l'IoU
fig, ax1 = plt.subplots()

ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss', color='tab:blue')
ax1.plot(range(1, num_epochs + 1), losses, color='tab:blue', label='Loss')
ax1.tick_params(axis='y', labelcolor='tab:blue')

ax2 = ax1.twinx()  # Créer un second axe y pour l'IoU
ax2.set_ylabel('IoU', color='tab:orange')
ax2.plot(range(1, num_epochs + 1), ious, color='tab:orange', label='IoU')
ax2.tick_params(axis='y', labelcolor='tab:orange')

fig.tight_layout()
plt.title('Loss and IoU vs Epochs')
plt.show()

# # Afficher la courbe de perte
# plt.plot(range(1, num_epochs + 1), losses)
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.title('Loss vs Epochs')
# plt.show()

#%% 📌 Sauvegarde du modèle fine-tuné
output_path = PROJECT_ROOT / "outputs/SAM_tune"
os.makedirs(output_path, exist_ok=True)

torch.save(sam.state_dict(), os.path.join(output_path, "sam_finetuned_spores.pth"))
torch.save(sam.state_dict(), os.path.join(output_path, "sam_finetuned_spores.pt"))

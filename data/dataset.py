import os
import cv2
import torch
import pandas as pd
from torch.utils.data import Dataset
import torch.nn.functional as F
from utils.metrics import texto_a_tensor, FACTOR_REDUCCION_ANCHO


class IAMParagraphDataset(Dataset):
    def __init__(self, csv_file, img_dir, alto_fijo, max_lines, is_train=False):
        self.df = pd.read_csv(csv_file).dropna(subset=['text']) 
        self.img_dir = img_dir
        self.alto_fijo = alto_fijo
        self.max_lines = max_lines
        self.is_train = is_train

    def __len__(self): 
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.imread(os.path.join(self.img_dir, row['image_id']), cv2.IMREAD_GRAYSCALE)
        texto = row['text']
        
        if img is None:
            return torch.zeros((1, self.alto_fijo, self.alto_fijo), dtype=torch.float32), texto_a_tensor("")
            
        alto_orig, ancho_orig = max(img.shape[0], 1), img.shape[1]
        limite_maximo_ancho = 1536
        
        factor_escala = min(self.alto_fijo / alto_orig, limite_maximo_ancho / ancho_orig)
        nuevo_alto = max(1, int(alto_orig * factor_escala))
        nuevo_ancho = max(1, int(ancho_orig * factor_escala))
        
        img_resized = cv2.resize(img, (nuevo_ancho, nuevo_alto))
        
        if nuevo_alto < self.alto_fijo:
            pad_abajo = self.alto_fijo - nuevo_alto
            img_resized = cv2.copyMakeBorder(img_resized, 0, pad_abajo, 0, 0, cv2.BORDER_CONSTANT, value=255)
        
        timesteps_esperados = (nuevo_ancho // FACTOR_REDUCCION_ANCHO) * self.max_lines
        timesteps_necesarios = len(texto) + 10 
        
        if timesteps_esperados < timesteps_necesarios:
            pad_der = ((timesteps_necesarios // self.max_lines) + 1) * FACTOR_REDUCCION_ANCHO - nuevo_ancho
            img_resized = cv2.copyMakeBorder(img_resized, 0, 0, 0, max(0, pad_der), cv2.BORDER_CONSTANT, value=255)
        
        tensor_img = torch.from_numpy((img_resized / 255.0).astype(np.float32)).unsqueeze(0)
        return tensor_img, texto_a_tensor(texto)

def collate_fn_paragraphs(batch, alto_absoluto, max_lines):
    imagenes, textos = [item[0] for item in batch], [item[1] for item in batch]
    max_ancho = max([img.shape[2] for img in imagenes]) if imagenes else 1
    
    imagenes_padded = [F.pad(img, (0, max_ancho - img.shape[2], 0, alto_absoluto - img.shape[1]), value=0) for img in imagenes]
    
    imagenes_batch = torch.stack(imagenes_padded)
    etiquetas_concatenadas = torch.cat(textos) if textos else torch.tensor([], dtype=torch.long)
    longitudes_etiquetas = torch.tensor([len(t) for t in textos], dtype=torch.long)
    longitudes_pred = torch.clamp(torch.tensor([(max_ancho // FACTOR_REDUCCION_ANCHO) * max_lines for _ in imagenes_padded], dtype=torch.long), min=1)
    
    return imagenes_batch, etiquetas_concatenadas, longitudes_pred, longitudes_etiquetas
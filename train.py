import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from itertools import zip_longest

# Importaciones Modulares
from data.dataset import IAMParagraphDataset, collate_fn_paragraphs
from models.htr_network import FullParagraphHTR
from utils.metrics import NUM_CLASES, decodificar_prediccion, calcular_metricas_jiwer, idx_a_char

# PARCHE DE MEMORIA
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ==========================================
# 0. CONFIGURACIÓN
# ==========================================
parser = argparse.ArgumentParser(description="Entrenamiento HTR")
parser.add_argument('--phase', type=str, choices=['lines', 'semi', 'full'], required=True)
args = parser.parse_args()

CONFIGURACION = {
    'lines': {'input_csv': 'iam/labels/train_lines.csv', 'val_csv': 'iam/labels/val_lines.csv', 'img_dir': 'iam/lines', 'max_lines': 1, 'alto_fijo': 128, 'batch_size': 260, 'modelo_in': None, 'modelo_out': 'htr_lines_mejor_modelo_v4.pth', 'pt_out': 'htr_produccion_lines_v4.pt', 'accumulation_steps': 1},
    'semi':  {'input_csv': 'iam/labels/train_semi.csv', 'val_csv': 'iam/labels/val_semi.csv', 'img_dir': 'iam/semi_paragraphs', 'max_lines': 4, 'alto_fijo': 512, 'batch_size': 88, 'modelo_in': 'htr_lines_mejor_modelo_v4.pth', 'modelo_out': 'htr_semi_mejor_modelo_v4.pth', 'pt_out': 'htr_produccion_semi_v4.pt', 'accumulation_steps': 1},
    'full':  {'input_csv': 'iam/labels/train_full.csv', 'val_csv': 'iam/labels/val_full.csv', 'img_dir': 'iam/paragraphs', 'max_lines': 13, 'alto_fijo': 1024, 'batch_size': 50, 'modelo_in': 'htr_semi_mejor_modelo_v4.pth', 'modelo_out': 'htr_full_mejor_modelo_v4.pth', 'pt_out': 'htr_produccion_full_v4.pt', 'accumulation_steps': 1}
}

cfg = CONFIGURACION[args.phase]

# Redirigir pesos a la carpeta 'weights/'
checkpoint_dir = 'weights'
os.makedirs(checkpoint_dir, exist_ok=True)
ruta_mejor_modelo_final = os.path.join(checkpoint_dir, cfg['modelo_out'])
ruta_modelo_preentrenado = os.path.join(checkpoint_dir, cfg['modelo_in']) if cfg['modelo_in'] else None

# ==========================================
# 1. DATALOADERS
# ==========================================
collate_wrapper = lambda batch: collate_fn_paragraphs(batch, cfg['alto_fijo'], cfg['max_lines'])

dataloader_train = DataLoader(IAMParagraphDataset(cfg['input_csv'], cfg['img_dir'], cfg['alto_fijo'], cfg['max_lines'], is_train=True), 
                              batch_size=cfg['batch_size'], shuffle=True, collate_fn=collate_wrapper, num_workers=3, pin_memory=True)
dataloader_val = DataLoader(IAMParagraphDataset(cfg['val_csv'], cfg['img_dir'], cfg['alto_fijo'], cfg['max_lines'], is_train=False), 
                            batch_size=cfg['batch_size'], shuffle=False, collate_fn=collate_wrapper, num_workers=3, pin_memory=True)

# ==========================================
# 2. INICIALIZACIÓN DEL MODELO
# ==========================================
dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modelo_final = FullParagraphHTR(num_clases=NUM_CLASES, hidden_size=256, max_lines=cfg['max_lines'])

if ruta_modelo_preentrenado and os.path.exists(ruta_modelo_preentrenado):
    print(f"Cargando pesos: {ruta_modelo_preentrenado}...")
    modelo_final.load_state_dict(torch.load(ruta_modelo_preentrenado), strict=True)
modelo_final = modelo_final.to(dispositivo)

# ==========================================
# 3. OPTIMIZADOR Y SCHEDULER
# ==========================================
if args.phase == 'lines':
    optimizador = optim.AdamW(modelo_final.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizador, mode='min', factor=0.5, patience=20)
else:
    parametros_protegidos = list(modelo_final.cnn.parameters()) + list(modelo_final.decoder.parameters()) + list(modelo_final.fc.parameters())
    parametros_atencion = list(modelo_final.spatial_context.parameters()) + list(modelo_final.iterative_collapse.parameters())
    optimizador = optim.AdamW([
        {'params': parametros_protegidos, 'lr': 1e-5},
        {'params': parametros_atencion, 'lr': 5e-4}
    ], weight_decay=1e-4)
    patience_val = 20 if args.phase == 'semi' else 15
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizador, mode='min', factor=0.5, patience=patience_val)

criterio_ctc = nn.CTCLoss(blank=0, zero_infinity=True)
scaler = GradScaler()
mejor_cer_val = float('inf')

# ==========================================
# 4. BUCLE DE ENTRENAMIENTO
# ==========================================
for epoca in range(200):
    modelo_final.train()
    perdida_train_acumulada = 0.0
    optimizador.zero_grad() 
    loop_train = tqdm(dataloader_train, desc=f"Época [{epoca}] Train", leave=False)

    for i, (imagenes, etiquetas, long_pred, long_etiq) in enumerate(loop_train):
        imagenes, etiquetas = imagenes.to(dispositivo), etiquetas.to(dispositivo)
        long_pred, long_etiq = long_pred.to(dispositivo), long_etiq.to(dispositivo)
        
        with autocast("cuda"):
            predicciones = modelo_final(imagenes)
            T, B, _ = predicciones.size()
            long_pred_dinamico = torch.full(size=(B,), fill_value=T, dtype=torch.long, device=dispositivo)
            loss = criterio_ctc(F.log_softmax(predicciones, dim=2), etiquetas, long_pred_dinamico, long_etiq) / cfg['accumulation_steps']
            
        scaler.scale(loss).backward()
        
        if (i + 1) % cfg['accumulation_steps'] == 0 or (i + 1) == len(loop_train):
            torch.nn.utils.clip_grad_norm_(modelo_final.parameters(), max_norm=2.0)
            scaler.step(optimizador)
            scaler.update()
            optimizador.zero_grad()
            
        perdida_train_acumulada += (loss.item() * cfg['accumulation_steps'])
        loop_train.set_postfix(loss=(loss.item() * cfg['accumulation_steps']))
        
    # --- VALIDACIÓN ---
    modelo_final.eval() 
    cer_acumulado, wer_acumulado = 0.0, 0.0
    
    with torch.no_grad(): 
        for imagenes, etiquetas, long_pred, long_etiq in dataloader_val:
            imagenes, etiquetas = imagenes.to(dispositivo), etiquetas.to(dispositivo)
            with autocast("cuda"): 
                predicciones = modelo_final(imagenes)
            
            texto_pred = decodificar_prediccion(predicciones)
            long_real = long_etiq[0].item()
            texto_real = "".join([idx_a_char.get(idx, "") for idx in etiquetas[:long_real].cpu().numpy()])
            
            cer_batch, wer_batch = calcular_metricas_jiwer(texto_pred, texto_real)
            cer_acumulado += cer_batch
            wer_acumulado += wer_batch
            
    cer_val_medio = cer_acumulado / len(dataloader_val)
    wer_val_medio = wer_acumulado / len(dataloader_val)
    
    scheduler.step(cer_val_medio) 
    torch.cuda.empty_cache()
    
    print(f"\nÉpoca {epoca} | Train Loss: {perdida_train_acumulada/len(dataloader_train):.4f} | Val CER: {cer_val_medio * 100:.2f}% | Val WER: {wer_val_medio * 100:.2f}%")
    
    width = 75
    print(f"📝 {'Predicción Final':<{width}} | Texto Real")
    print("-" * (width + 60))
    for line1, line2 in zip_longest(texto_pred.split('\n'), texto_real.split('\n'), fillvalue=''):
        print(f"{line1:<{width}} | {line2}")
    
    if cer_val_medio < mejor_cer_val:
        mejor_cer_val = cer_val_medio
        torch.save(modelo_final.state_dict(), ruta_mejor_modelo_final)
        print(f" -> ¡Nuevo mejor modelo ({args.phase}) guardado en {ruta_mejor_modelo_final}!")

# ==========================================
# 5. EXPORTACIÓN PARA PRODUCCIÓN
# ==========================================
modelo_final.eval()
modelo_produccion = torch.jit.script(modelo_final)
ruta_pt = os.path.join(checkpoint_dir, cfg['pt_out'])
modelo_produccion.save(ruta_pt)
print(f"\n¡Modelo compilado y guardado como {ruta_pt}!")
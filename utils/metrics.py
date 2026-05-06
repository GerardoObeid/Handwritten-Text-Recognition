import torch
import jiwer

# ==========================================
# CONSTANTES Y VOCABULARIO
# ==========================================
CARACTERES = " !\"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz\n"
char_a_idx = {char: idx + 1 for idx, char in enumerate(CARACTERES)} 
idx_a_char = {idx + 1: char for idx, char in enumerate(CARACTERES)}
idx_a_char[0] = "" # CTC Blank token

NUM_CLASES = len(CARACTERES) + 1 
FACTOR_REDUCCION_ANCHO = 4 


# ==========================================
# FUNCIONES DE UTILIDAD
# ==========================================
def texto_a_tensor(texto):
    indices = [char_a_idx.get(c, 0) for c in str(texto)] 
    return torch.tensor([i for i in indices if i != 0], dtype=torch.long)


def decodificar_prediccion(tensor_prediccion):
    pred_indices = torch.argmax(tensor_prediccion[:, 0, :], dim=1).detach().cpu().numpy()
    texto_decodificado = []
    idx_anterior = -1
    for idx in pred_indices:
        if idx != 0 and idx != idx_anterior:
            texto_decodificado.append(idx_a_char.get(idx, ""))
        idx_anterior = idx
    return "".join(texto_decodificado)


def calcular_metricas_jiwer(prediccion, real):
    real_limpio = real.strip() if len(real.strip()) > 0 else "<VACIO>"
    pred_limpia = prediccion.strip() if len(prediccion.strip()) > 0 else "<VACIO>"
    cer = jiwer.cer(real_limpio, pred_limpia)
    wer = jiwer.wer(real_limpio, pred_limpia)
    return cer, wer
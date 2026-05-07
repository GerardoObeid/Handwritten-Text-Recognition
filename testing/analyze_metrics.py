
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import re
df = pd.read_csv('../results/analisis_detallado_test.csv')

df.describe()


# 1. Suponiendo que tu DF se llama 'df_resultados' y tiene una columna 'cer'
# Convertimos el CER a porcentaje para que sea más legible (0.12 -> 12%)
df['cer_pct'] = df['cer'] * 100

# 2. Definir rangos (bins) para ver la distribución
bins = [0, 5, 7.5, 10, 20, 30,100]
labels = ['0-5%', '5-7.5%', '7.5-10%', '10-20%', '20-30%','>30%']
df['rango_cer'] = pd.cut(df['cer_pct'], bins=bins, labels=labels)

# Ver cantidad por rango (Opcional, para tu control)
print(df['rango_cer'].value_counts().sort_index())

# Crear la figura
fig, ax = plt.subplots(figsize=(10, 6))

# Datos para el boxplot (los valores crudos de CER en porcentaje)
data = df['cer_pct']

# Crear el boxplot
# patch_artist=True permite llenar la caja de color
# notch=True muestra el intervalo de confianza de la mediana
bp = ax.boxplot(data, vert=False, patch_artist=True, notch=False,
                showmeans=True, # Muestra la media además de la mediana
                flierprops=dict(marker='o', markerfacecolor='red', markersize=10, alpha=0.5))

# --- Estética y Personalización ---
# Color de la caja
for patch in bp['boxes']:
    patch.set_facecolor('lightblue')

# Color de la línea de la mediana
for median in bp['medians']:
    median.set(color='darkblue', linewidth=4)

# Color de la media (el triángulo)
for mean in bp['means']:
    mean.set(markerfacecolor='green', markeredgecolor='green')

# Etiquetas y Títulos
#ax.set_title('Character Error Rate (CER) - N=154', fontsize=14, pad=15)
ax.set_xlabel('CER (%)', fontsize=12)
ax.set_yticklabels(['Test Set']) # Nombre de la variable en el eje Y

# Añadir rejilla solo para el eje X
ax.grid(axis='x', linestyle='--', alpha=0.7)

# Mostrar el gráfico
plt.tight_layout()
plt.savefig('../results/boxplot_cer.png', dpi=300)
plt.show()


df = pd.read_csv('../results/metricas_entrenamiento.csv')
dflines = df[df['Fase']=='lines']
dfsemi = df[df['Fase']=='semi']
dffull = df[df['Fase']=='full']

print(dflines.shape)
print(dfsemi.shape)
print(dffull.shape)


# Configure the plot: 1 row, 3 columns for subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
#fig.suptitle('Evolution of CER, WER, and CTC Loss per Epoch and Phase', fontsize=18)

# Define colors and line styles
colores = {'lines': '#1f77b4', 'semi': '#ff7f0e', 'full': '#2ca02c'}
estilos = {'CER (%)': '-', 'WER (%)': '--', 'CTC': ':'}
phases = ['lines', 'semi', 'full']

for i, phase in enumerate(phases):
    ax1 = axes[i]
    df_fase = df[df['Fase'] == phase].sort_values('Epoca')
    
    if not df_fase.empty:
        # Plot CER and WER on the primary y-axis
        l1 = ax1.plot(df_fase['Epoca'], df_fase['CER (%)'], 
                 color=colores[phase], linestyle=estilos['CER (%)'], linewidth=2, label=f'CER')
        l2 = ax1.plot(df_fase['Epoca'], df_fase['WER (%)'], 
                 color=colores[phase], linestyle=estilos['WER (%)'], linewidth=2, label=f'WER')
        
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Error (%)', fontsize=12)
        ax1.set_title(f'{phase.capitalize()}', fontsize=14)
        ax1.grid(True, linestyle=':', alpha=0.7)
        
        # Create a secondary y-axis for the CTC Loss because its scale is much smaller
        ax2 = ax1.twinx()
        # Using a dark grey/black color for the CTC loss to make it pop against the phase color
        l3 = ax2.plot(df_fase['Epoca'], df_fase['Train Loss'], 
                 color='#444444', linestyle=estilos['CTC'], linewidth=2, alpha=0.8, label=f'CTC Loss')
        ax2.set_ylabel('CTC Loss', fontsize=12)
        
        # Combine legends from both axes
        lines = l1 + l2 + l3
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper right')

plt.tight_layout()

# Save the image
img_path = '../results/errors_cer_wer_subplots.png'
plt.savefig(img_path, dpi=150)
plt.show()

print(f"Image saved as: {img_path}")


